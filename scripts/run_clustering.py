#!/usr/bin/env python3
"""Clustering script — Phase 4 of the desert patterns pipeline.

Discovers natural pattern groupings via UMAP dimensionality reduction →
HDBSCAN clustering, then characterizes clusters with texture profiles,
feature importance, temporal dynamics, and continuous space analysis.

Usage:
    python scripts/run_clustering.py --config configs/clustering_config.yaml
    python scripts/run_clustering.py --config configs/clustering_config.yaml --ablation fused_default
    python scripts/run_clustering.py --config configs/clustering_config.yaml --skip-bootstrap --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering.cluster_characterization import ClusterCharacterizer
from src.clustering.cluster_discovery import (
    ClusterConfig,
    FittedPipeline,
    PatternClusterDiscovery,
    load_cluster_config,
    prepare_ablation_features,
    run_ablation,
)
from src.clustering.continuous_space import ContinuousPatternSpace, ContinuousSpaceConfig
from src.clustering.dimensionality_reduction import (
    DimensionalityReducer,
    load_reduction_config,
)
from src.clustering.temporal_analysis import TemporalPatternAnalysis
from src.features.feature_curation import (
    CurationResult,
    curate_features,
    load_curation_config,
)
from src.features.feature_store import FeatureStore
from src.features.fusion import FusionConfig, fuse_features

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Clustering & pattern discovery."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "clustering_config.yaml",
        help="Path to clustering configuration YAML file.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Run a single ablation by name (e.g. 'fused_default'). Default: run all.",
    )
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal analysis.",
    )
    parser.add_argument(
        "--skip-continuous",
        action="store_true",
        help="Skip continuous space analysis.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap stability analysis (faster).",
    )
    parser.add_argument(
        "--exclude-clusters",
        type=str,
        default=None,
        help="Comma-separated cluster IDs to exclude (e.g. '2,3,4,5').",
    )
    parser.add_argument(
        "--triage-from",
        type=str,
        default=None,
        help="Ablation name whose assignments define the clusters to exclude.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist fitted UMAP + HDBSCAN models via joblib for inference.",
    )
    parser.add_argument(
        "--scene-clusters",
        type=str,
        default=None,
        help=(
            "Comma-separated cluster IDs for scene direction removal "
            "(e.g. '6,7'). Requires --triage-from."
        ),
    )
    parser.add_argument(
        "--n-directions",
        type=int,
        default=None,
        help="Number of scene directions to project out (default: from config or 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Resolve paths
    store_path = PROJECT_ROOT / raw_config["feature_store"]
    catalog_path = PROJECT_ROOT / raw_config["catalog_path"]
    manifest_path = PROJECT_ROOT / raw_config["manifest_path"]
    output_dir = PROJECT_ROOT / raw_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = raw_config.get("random_seed", 42)
    np.random.seed(seed)

    # Load feature store
    store = FeatureStore(store_path)
    image_ids = store.load_image_ids()
    logger.info("Loaded %d image IDs from feature store", len(image_ids))

    # Load metadata — join manifest (patch_id → parent_image_id) with catalog
    metadata_df = _load_metadata(manifest_path, catalog_path, image_ids)
    logger.info("Metadata joined: %d rows", len(metadata_df))

    # Load texture feature names for characterization
    try:
        feature_names = store.load_texture_feature_names()
    except Exception:
        feature_names = [f"texture_{i}" for i in range(90)]

    # Parse configs
    reduction_configs = load_reduction_config(raw_config.get("dimensionality_reduction", {}))
    cluster_config = load_cluster_config(raw_config.get("clustering", {}))

    # --- Triage filtering ---
    exclude_clusters = args.exclude_clusters
    triage_from = args.triage_from

    # Fall back to config values if CLI flags not provided
    filter_cfg = raw_config.get("filtering", None) or {}
    if exclude_clusters is None and filter_cfg.get("excluded_clusters"):
        exclude_clusters = ",".join(str(c) for c in filter_cfg["excluded_clusters"])
    if triage_from is None and filter_cfg.get("triage_source"):
        triage_from = filter_cfg["triage_source"]

    filter_mask = None
    filter_suffix = ""
    if exclude_clusters is not None:
        if triage_from is None:
            logger.error("--exclude-clusters requires --triage-from")
            sys.exit(1)
        excluded_ids = [int(x.strip()) for x in exclude_clusters.split(",")]
        source_csv = output_dir / f"{triage_from}_assignments.csv"
        if not source_csv.exists():
            logger.error("Triage source not found: %s", source_csv)
            sys.exit(1)
        source_df = pd.read_csv(source_csv)
        excluded_image_ids = set(
            source_df[source_df["cluster_id"].isin(excluded_ids)]["image_id"]
        )
        filter_mask = np.array([iid not in excluded_image_ids for iid in image_ids])
        n_excluded = int(np.sum(~filter_mask))
        n_remaining = int(np.sum(filter_mask))
        logger.info(
            "Triage filter: excluding clusters %s from %s — "
            "removing %d patches, keeping %d",
            excluded_ids, triage_from, n_excluded, n_remaining,
        )
        filter_suffix = "_filtered"

        # Apply filter to image_ids and metadata
        image_ids = [iid for iid, keep in zip(image_ids, filter_mask) if keep]
        metadata_df = metadata_df[metadata_df["image_id"].isin(image_ids)]

        # Save filter log
        filter_log = {
            "triage_source": triage_from,
            "excluded_clusters": excluded_ids,
            "n_excluded": n_excluded,
            "n_remaining": n_remaining,
            "excluded_image_ids": sorted(excluded_image_ids),
        }
        log_path = output_dir / f"{triage_from}_filter_log.json"
        with open(log_path, "w") as f:
            json.dump(filter_log, f, indent=2)
        logger.info("Saved filter log: %s", log_path)

    # --- Feature curation (scene direction removal) ---
    scene_cluster_ids_str = args.scene_clusters
    curation_cfg = raw_config.get("curation", None) or {}
    if scene_cluster_ids_str is None and curation_cfg.get("scene_clusters"):
        scene_cluster_ids_str = ",".join(str(c) for c in curation_cfg["scene_clusters"])

    curated_dino = None
    curation_result: CurationResult | None = None
    if scene_cluster_ids_str is not None:
        if triage_from is None:
            logger.error("--scene-clusters requires --triage-from")
            sys.exit(1)
        scene_cluster_ids = [int(x.strip()) for x in scene_cluster_ids_str.split(",")]

        # Load Stage 1 labels from triage source assignments
        source_csv = output_dir / f"{triage_from}_assignments.csv"
        if not source_csv.exists():
            logger.error("Triage source not found: %s", source_csv)
            sys.exit(1)
        source_df = pd.read_csv(source_csv)
        stage1_labels = source_df["cluster_id"].values

        # Load raw DINO for ALL patches (before triage filter)
        raw_dino = store.load_dino()

        # Determine n_directions (CLI > config > default)
        n_dirs = args.n_directions
        if n_dirs is None:
            n_dirs = curation_cfg.get("n_directions", 3)

        curation_config = load_curation_config({
            "n_directions": n_dirs,
            "method": curation_cfg.get("method", "centroid_pca"),
            "min_eigenvalue": curation_cfg.get("min_eigenvalue", 1e-10),
        })

        curation_result = curate_features(
            raw_dino, stage1_labels, scene_cluster_ids, config=curation_config,
        )
        curated_dino = curation_result.curated_features

        # Log diagnostics
        scene_cluster_sizes = {}
        for sid in scene_cluster_ids:
            scene_cluster_sizes[str(sid)] = int((stage1_labels == sid).sum())

        curation_log = {
            "scene_clusters": scene_cluster_ids,
            "n_directions_removed": curation_result.n_directions_removed,
            "variance_explained_per_direction": curation_result.variance_explained.tolist(),
            "total_variance_fraction_removed": curation_result.total_variance_fraction,
            "scene_cluster_sizes": scene_cluster_sizes,
        }
        log_path = output_dir / f"{triage_from}_curation_log.json"
        with open(log_path, "w") as f:
            json.dump(curation_log, f, indent=2)
        logger.info("Saved curation log: %s", log_path)
        logger.info(
            "Curation: removed %d directions (%.1f%% variance)",
            curation_result.n_directions_removed,
            curation_result.total_variance_fraction * 100,
        )

    # Determine which ablations to run
    ablation_defs = raw_config.get("clustering", {}).get("feature_ablations", [])
    if args.ablation:
        ablation_defs = [a for a in ablation_defs if a["name"] == args.ablation]
        if not ablation_defs:
            logger.error("Ablation %r not found in config", args.ablation)
            sys.exit(1)

    # Run each ablation
    for abl_def in ablation_defs:
        name = abl_def["name"]
        output_name = name + filter_suffix
        logger.info("=" * 60)
        logger.info("Running ablation: %s", output_name)
        logger.info("=" * 60)

        # Load features — use curated DINO if curation was applied
        dino_w = abl_def.get("dino_weight", 0.7)
        tex_w = abl_def.get("texture_weight", 0.3)
        if curated_dino is not None:
            if name == "dino_only":
                features = curated_dino.copy()
            elif name in ("fused_default", "fused_equal"):
                texture = store.load_texture()
                fusion_config = FusionConfig(dino_weight=dino_w, texture_weight=tex_w)
                fusion_result = fuse_features(curated_dino, texture, fusion_config)
                features = fusion_result.fused
            else:  # texture_only — no DINO involvement
                features = prepare_ablation_features(name, store, dino_w, tex_w)
        else:
            features = prepare_ablation_features(name, store, dino_w, tex_w)

        # Apply triage filter to features
        if filter_mask is not None:
            features = features[filter_mask]
        logger.info("Features shape: %s", features.shape)

        # Run ablation pipeline
        result = run_ablation(
            name, features, cluster_config, reduction_configs,
            return_fitted=args.save_models,
        )

        # Bootstrap stability
        if not args.skip_bootstrap:
            bootstrap_n = raw_config.get("validation", {}).get("bootstrap_n", 100)
            discoverer = PatternClusterDiscovery(cluster_config)
            from src.clustering.dimensionality_reduction import DimensionalityReducer
            reducer = DimensionalityReducer()
            high_dim = reducer.fit_transform(features, reduction_configs["high_dim"])
            stability = discoverer.evaluate_stability(high_dim.embeddings, n_bootstrap=bootstrap_n)
            logger.info(
                "Stability: ARI=%.3f ± %.3f",
                stability.mean_ari,
                stability.std_ari,
            )

        # Cluster characterization
        try:
            texture_features = store.load_texture()
            if filter_mask is not None:
                texture_features = texture_features[filter_mask]
        except Exception:
            logger.warning("Texture features not available — using zeros for characterization")
            texture_features = np.zeros((len(image_ids), len(feature_names)), dtype=np.float64)
        characterizer = ClusterCharacterizer(
            labels=result.cluster_result.labels,
            features_texture=texture_features,
            image_ids=image_ids,
            feature_names=feature_names,
            metadata_df=metadata_df,
            features_fused=features,
        )
        profiles = characterizer.characterize_all()
        importance_df = characterizer.compute_feature_importance()

        # Temporal analysis
        temporal_result = None
        if not args.skip_temporal:
            temporal = TemporalPatternAnalysis(
                labels=result.cluster_result.labels,
                image_ids=image_ids,
                metadata_df=metadata_df,
                features_2d=result.umap_2d.embeddings,
            )
            temporal_result = temporal.run_all()

        # Continuous space analysis
        continuous_result = None
        if not args.skip_continuous:
            cps = ContinuousPatternSpace(
                embeddings_2d=result.umap_2d.embeddings,
                image_ids=image_ids,
                metadata_df=metadata_df,
            )
            continuous_result = cps.kernel_density()

        # Save outputs
        _save_outputs(
            output_dir,
            output_name,
            result,
            profiles,
            importance_df,
            temporal_result,
            continuous_result,
            stability if not args.skip_bootstrap else None,
            image_ids,
        )

        # Save fitted models for inference
        if args.save_models and result.fitted_pipeline is not None:
            models_dir = PROJECT_ROOT / "outputs" / "models" / output_name
            _save_models(
                result.fitted_pipeline, models_dir, output_name, name,
                raw_config, curation_result,
            )

    logger.info("Done.")


def _load_metadata(
    manifest_path: Path,
    catalog_path: Path,
    image_ids: list[str],
) -> pd.DataFrame:
    """Load and join manifest + catalog to get metadata for each image ID.

    The feature store's image_ids are patch_ids from the manifest. We join
    through the manifest's parent_image_id to look up catalog metadata.
    """
    manifest = pd.read_csv(manifest_path)
    catalog = pd.read_csv(catalog_path)

    # Manifest has patch_id → parent_image_id
    # Catalog has image_id as the identifier
    if "parent_image_id" in manifest.columns and "image_id" in catalog.columns:
        merged = manifest.merge(
            catalog,
            left_on="parent_image_id",
            right_on="image_id",
            how="left",
            suffixes=("_manifest", "_catalog"),
        )
        # Use patch_id as the main image_id
        merged["image_id"] = merged["patch_id"]
    else:
        merged = manifest.copy()
        merged["image_id"] = merged.get("patch_id", merged.index)

    # Filter to only IDs in the feature store
    merged = merged[merged["image_id"].isin(image_ids)]
    return merged


def _save_outputs(
    output_dir: Path,
    name: str,
    result,
    profiles: list,
    importance_df: pd.DataFrame,
    temporal_result,
    continuous_result,
    stability,
    image_ids: list[str],
) -> None:
    """Save all outputs for a single ablation."""

    # Assignments CSV
    assignments_path = output_dir / f"{name}_assignments.csv"
    assignments = pd.DataFrame({
        "image_id": image_ids,
        "cluster_id": result.cluster_result.labels,
        "probability": result.cluster_result.probabilities,
        "umap_x": result.umap_2d.embeddings[:, 0],
        "umap_y": result.umap_2d.embeddings[:, 1],
        "umap_3d_x": result.umap_3d.embeddings[:, 0],
        "umap_3d_y": result.umap_3d.embeddings[:, 1],
        "umap_3d_z": result.umap_3d.embeddings[:, 2],
    })
    assignments.to_csv(assignments_path, index=False)
    logger.info("Saved assignments: %s", assignments_path)

    # Profiles JSON
    profiles_path = output_dir / f"{name}_profiles.json"
    profiles_data = []
    for p in profiles:
        profiles_data.append({
            "cluster_id": p.cluster_id,
            "size": p.size,
            "texture_mean": p.texture_mean.tolist(),
            "texture_std": p.texture_std.tolist(),
            "distinguishing_features": p.distinguishing_features,
            "metadata_distribution": p.metadata_distribution,
            "representative_ids": p.representative_ids,
            "boundary_ids": p.boundary_ids,
            "description": p.description,
        })
    with open(profiles_path, "w") as f:
        json.dump(profiles_data, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved profiles: %s", profiles_path)

    # Quality metrics JSON
    quality_path = output_dir / f"{name}_quality_metrics.json"
    qm = result.quality_metrics
    quality_data = {
        "silhouette": qm.silhouette if not np.isnan(qm.silhouette) else None,
        "davies_bouldin": qm.davies_bouldin if not np.isnan(qm.davies_bouldin) else None,
        "calinski_harabasz": qm.calinski_harabasz if not np.isnan(qm.calinski_harabasz) else None,
        "dbcv": qm.dbcv if not np.isnan(qm.dbcv) else None,
        "per_cluster_silhouette": {str(k): v for k, v in qm.per_cluster_silhouette.items()},
        "n_clusters": result.cluster_result.n_clusters,
        "noise_count": result.cluster_result.noise_count,
        "noise_fraction": result.cluster_result.noise_fraction,
    }
    if stability is not None:
        quality_data["bootstrap"] = {
            "mean_ari": stability.mean_ari if not np.isnan(stability.mean_ari) else None,
            "std_ari": stability.std_ari if not np.isnan(stability.std_ari) else None,
            "per_cluster_recovery": {
                str(k): v for k, v in stability.per_cluster_recovery.items()
            },
            "n_bootstrap": stability.n_bootstrap,
        }
    with open(quality_path, "w") as f:
        json.dump(quality_data, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved quality metrics: %s", quality_path)

    # Feature importance CSV
    importance_path = output_dir / f"{name}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info("Saved feature importance: %s", importance_path)

    # Temporal JSON
    if temporal_result is not None:
        temporal_path = output_dir / f"{name}_temporal.json"
        temporal_data = {
            "phase_distributions": (
                temporal_result.phase_distributions.to_dict()
                if not temporal_result.phase_distributions.empty
                else {}
            ),
            "transition_matrix": (
                temporal_result.transition_matrix.tolist()
                if temporal_result.transition_matrix is not None
                else None
            ),
            "embedding_drift": temporal_result.embedding_drift,
            "test_results": (
                temporal_result.test_results.to_dict(orient="records")
                if not temporal_result.test_results.empty
                else []
            ),
        }
        with open(temporal_path, "w") as f:
            json.dump(temporal_data, f, indent=2, cls=_NumpyEncoder)
        logger.info("Saved temporal: %s", temporal_path)

    # Continuous space CSV
    if continuous_result is not None:
        cont_path = output_dir / f"{name}_continuous_space.csv"
        cont_df = pd.DataFrame({
            "image_id": image_ids,
            "pattern_x": continuous_result.coordinates[:, 0],
            "pattern_y": continuous_result.coordinates[:, 1],
        })
        cont_df.to_csv(cont_path, index=False)
        logger.info("Saved continuous space: %s", cont_path)


def _save_models(
    pipeline: FittedPipeline,
    model_dir: Path,
    output_name: str,
    feature_type: str,
    raw_config: dict,
    curation_result: CurationResult | None = None,
) -> None:
    """Save fitted models to disk via joblib for later inference.

    Args:
        pipeline: Fitted pipeline with UMAP reducers and HDBSCAN discoverer.
        model_dir: Directory to write model files into.
        output_name: Ablation output name (for the config file).
        feature_type: Base ablation name (e.g. ``"dino_only"``).
        raw_config: Raw clustering config dict for recording parameters.
        curation_result: If curation was applied, persist the removed
            directions so inference can apply the same projection.
    """
    import joblib

    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the underlying fitted UMAP/HDBSCAN objects
    joblib.dump(pipeline.umap_high_dim._fitted_model, model_dir / "umap_high_dim.joblib")
    joblib.dump(pipeline.umap_2d._fitted_model, model_dir / "umap_2d.joblib")
    joblib.dump(pipeline.umap_3d._fitted_model, model_dir / "umap_3d.joblib")
    joblib.dump(pipeline.hdbscan_discoverer._clusterer, model_dir / "hdbscan.joblib")

    # Save curation directions if curation was applied
    if curation_result is not None and curation_result.n_directions_removed > 0:
        joblib.dump(curation_result.removed_directions, model_dir / "curation.joblib")
        logger.info(
            "Saved curation directions (%d) to %s",
            curation_result.n_directions_removed,
            model_dir / "curation.joblib",
        )

    # Save config for reproducibility
    config_data = {
        "ablation_name": output_name,
        "feature_type": feature_type,
        "clustering": raw_config.get("clustering", {}),
        "dimensionality_reduction": raw_config.get("dimensionality_reduction", {}),
    }
    config_path = model_dir / "pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    logger.info("Saved fitted models to %s", model_dir)


if __name__ == "__main__":
    main()

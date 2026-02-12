#!/usr/bin/env python3
"""Figure generation script — Phase 5 of the desert patterns pipeline.

Reads Phase 4 clustering outputs and generates publication-quality figures
including UMAP scatter plots, cluster galleries, silhouette analysis,
temporal phase distributions, and density contours.

Usage:
    python scripts/generate_figures.py --config configs/visualization_config.yaml
    python scripts/generate_figures.py --config configs/visualization_config.yaml --ablation fused_default --verbose
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

from src.visualization.style import FigureStyle, load_style_config, setup_matplotlib_style
from src.visualization.embedding_plots import (
    plot_umap_clusters,
    plot_umap_by_metadata,
    plot_silhouette,
)
from src.visualization.cluster_galleries import (
    plot_cluster_gallery,
    plot_cluster_comparison_strip,
    plot_noise_gallery,
)
from src.visualization.temporal_plots import (
    plot_phase_distribution,
    plot_density_contours,
    plot_embedding_drift,
)

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None if it doesn't exist."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path) -> pd.DataFrame | None:
    """Load a CSV file, returning None if it doesn't exist."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    return pd.read_csv(path)


def _build_patch_paths(manifest_df: pd.DataFrame) -> dict[str, Path]:
    """Build a mapping from patch_id to output file path."""
    paths: dict[str, Path] = {}
    path_col = "patch_path" if "patch_path" in manifest_df.columns else "output_path"
    for _, row in manifest_df.iterrows():
        pid = row.get("patch_id", row.get("image_id"))
        out = row.get(path_col)
        if pid and out:
            paths[str(pid)] = Path(out)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Generate visualization figures."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "visualization_config.yaml",
        help="Path to visualization configuration YAML file.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Ablation name to visualize (default: from config).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

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
    cluster_dir = PROJECT_ROOT / raw_config["cluster_dir"]
    manifest_path = PROJECT_ROOT / raw_config["manifest_path"]
    catalog_path = PROJECT_ROOT / raw_config["catalog_path"]
    output_dir = PROJECT_ROOT / raw_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = args.ablation or raw_config.get("ablation_name", "fused_default")

    # Load style
    style = load_style_config(raw_config.get("style", {}))
    setup_matplotlib_style(style)
    logger.info("Style: dpi=%d, font=%s", style.dpi, style.font_family)

    # Load Phase 4 outputs
    assignments_df = _load_csv(cluster_dir / f"{ablation}_assignments.csv")
    if assignments_df is None:
        logger.error("Assignments file not found for ablation '%s'", ablation)
        sys.exit(1)

    profiles = _load_json(cluster_dir / f"{ablation}_profiles.json")
    quality_metrics = _load_json(cluster_dir / f"{ablation}_quality_metrics.json")
    temporal_data = _load_json(cluster_dir / f"{ablation}_temporal.json")
    continuous_df = _load_csv(cluster_dir / f"{ablation}_continuous_space.csv")
    feature_importance = _load_csv(cluster_dir / f"{ablation}_feature_importance.csv")

    # Load manifest + catalog for metadata
    manifest_df = _load_csv(manifest_path)
    catalog_df = _load_csv(catalog_path)

    metadata_df = None
    patch_paths: dict[str, Path] = {}
    if manifest_df is not None:
        patch_paths = _build_patch_paths(manifest_df)
        if catalog_df is not None:
            metadata_df = manifest_df.merge(
                catalog_df,
                left_on="parent_image_id",
                right_on="image_id",
                how="left",
                suffixes=("_manifest", "_catalog"),
            )
            metadata_df["image_id"] = metadata_df["patch_id"]
        else:
            metadata_df = manifest_df.copy()
            if "patch_id" in metadata_df.columns:
                metadata_df["image_id"] = metadata_df["patch_id"]

    # ---- Generate figures ----

    logger.info("Generating UMAP cluster scatter...")
    plot_umap_clusters(assignments_df, output_dir / "umap_clusters", style)

    # Metadata-coloured UMAPs
    if metadata_df is not None:
        for col in ("temporal_phase", "altitude_group", "source_type"):
            if col in metadata_df.columns:
                logger.info("Generating UMAP by %s...", col)
                plot_umap_by_metadata(
                    assignments_df, metadata_df, col,
                    output_dir / f"umap_{col}", style,
                )

    # Silhouette
    if quality_metrics is not None:
        logger.info("Generating silhouette plot...")
        try:
            feature_store_path = PROJECT_ROOT / raw_config["feature_store"]
            from src.features.feature_store import FeatureStore
            store = FeatureStore(feature_store_path)
            try:
                fused = store.load_fused()
            except Exception:
                fused = store.load_dino()
            plot_silhouette(
                fused, assignments_df["cluster_id"].values,
                output_dir / "silhouette", style,
            )
        except Exception as e:
            logger.warning("Could not generate silhouette plot: %s", e)

    # Cluster galleries
    if profiles:
        gallery_config = raw_config.get("cluster_gallery", {})
        n_cols = gallery_config.get("grid_cols", 5)
        thumb_size = gallery_config.get("thumbnail_size", 128)

        for profile in profiles:
            cid = profile.get("cluster_id", "unknown")
            if cid == -1:
                continue
            logger.info("Generating gallery for cluster %s...", cid)
            plot_cluster_gallery(
                profile, patch_paths,
                output_dir / f"cluster_{cid}_gallery",
                n_cols=n_cols, thumbnail_size=thumb_size, style=style,
            )

        logger.info("Generating cluster comparison strip...")
        plot_cluster_comparison_strip(
            profiles, patch_paths,
            output_dir / "cluster_comparison_strip",
            n_per_cluster=gallery_config.get("n_per_cluster", 8),
            thumbnail_size=thumb_size, style=style,
        )

        logger.info("Generating noise gallery...")
        plot_noise_gallery(
            assignments_df, patch_paths,
            output_dir / "noise_gallery",
            thumbnail_size=thumb_size, n_cols=n_cols, style=style,
        )

    # Temporal plots
    if temporal_data:
        logger.info("Generating phase distribution plot...")
        plot_phase_distribution(temporal_data, output_dir / "phase_distribution", style)

        if continuous_df is not None and metadata_df is not None and "temporal_phase" in metadata_df.columns:
            logger.info("Generating density contours...")
            phase_order = raw_config.get("temporal", {}).get("phase_order")
            plot_density_contours(
                continuous_df, metadata_df,
                output_dir / "density_contours",
                phases=phase_order, style=style,
            )
        elif continuous_df is not None:
            logger.info("Skipping density contours — no temporal_phase metadata")

        logger.info("Generating embedding drift plot...")
        plot_embedding_drift(
            temporal_data, assignments_df,
            output_dir / "embedding_drift", style,
        )

    logger.info("Done — figures saved to %s", output_dir)


if __name__ == "__main__":
    main()

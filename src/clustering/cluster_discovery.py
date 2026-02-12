"""HDBSCAN clustering, quality metrics, and bootstrap stability.

Discovers natural pattern groupings without pre-specifying the number of
clusters.  Supports multiple feature ablations (DINOv3-only, texture-only,
fused with varying weights) for comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import hdbscan
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from src.clustering.dimensionality_reduction import (
    DimensionalityReducer,
    ReductionResult,
    UMAPConfig,
    load_reduction_config,
)
from src.features.feature_store import FeatureStore
from src.features.fusion import FusionConfig, fuse_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ClusterConfig:
    """Configuration for HDBSCAN clustering."""

    method: str = "hdbscan"
    min_cluster_size: int = 15
    min_samples: int = 5
    cluster_selection_method: str = "eom"


def load_cluster_config(config_dict: dict) -> ClusterConfig:
    """Create a ClusterConfig from a config dictionary.

    Args:
        config_dict: Dictionary with clustering configuration.

    Returns:
        Populated ClusterConfig.
    """
    return ClusterConfig(
        method=config_dict.get("method", "hdbscan"),
        min_cluster_size=config_dict.get("min_cluster_size", 15),
        min_samples=config_dict.get("min_samples", 5),
        cluster_selection_method=config_dict.get("cluster_selection_method", "eom"),
    )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Output of HDBSCAN clustering."""

    labels: np.ndarray  # (N,) int, -1 = noise
    probabilities: np.ndarray  # (N,) float, membership confidence
    outlier_scores: np.ndarray  # (N,) float, GLOSH outlier score
    n_clusters: int
    noise_count: int
    noise_fraction: float


@dataclass
class QualityMetrics:
    """Cluster validation metrics."""

    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    dbcv: float
    per_cluster_silhouette: dict[int, float]


@dataclass
class StabilityResult:
    """Bootstrap stability analysis result."""

    mean_ari: float
    std_ari: float
    per_cluster_recovery: dict[int, float]  # fraction of bootstraps recovering each cluster
    n_bootstrap: int


@dataclass
class AblationResult:
    """Result of a single feature ablation run."""

    name: str
    features_used: str
    cluster_result: ClusterResult
    quality_metrics: QualityMetrics
    umap_2d: ReductionResult
    umap_3d: ReductionResult
    fitted_pipeline: FittedPipeline | None = None


@dataclass
class FittedPipeline:
    """Fitted model objects for inference on new data."""

    umap_high_dim: DimensionalityReducer
    umap_2d: DimensionalityReducer
    umap_3d: DimensionalityReducer
    hdbscan_discoverer: PatternClusterDiscovery


# ---------------------------------------------------------------------------
# Core clustering class
# ---------------------------------------------------------------------------


class PatternClusterDiscovery:
    """Discover natural pattern groupings using HDBSCAN.

    HDBSCAN advantages over k-means:
    - No need to pre-specify number of clusters
    - Identifies noise/outlier images
    - Handles clusters of varying density
    - Provides soft cluster membership probabilities
    """

    def __init__(self, config: ClusterConfig | None = None) -> None:
        self.config = config or ClusterConfig()
        self._clusterer: hdbscan.HDBSCAN | None = None

    def fit(self, features_reduced: np.ndarray) -> ClusterResult:
        """Run HDBSCAN on reduced features.

        Args:
            features_reduced: Feature matrix of shape ``(N, D)``
                (typically after UMAP reduction to 10-20 dims).

        Returns:
            ClusterResult with labels, probabilities, and summary stats.
        """
        logger.info(
            "HDBSCAN: %d samples, %d dims (min_cluster_size=%d, min_samples=%d, "
            "selection=%s)",
            features_reduced.shape[0],
            features_reduced.shape[1],
            self.config.min_cluster_size,
            self.config.min_samples,
            self.config.cluster_selection_method,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            cluster_selection_method=self.config.cluster_selection_method,
            prediction_data=True,
        )
        clusterer.fit(features_reduced)
        self._clusterer = clusterer

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        outlier_scores = clusterer.outlier_scores_

        n_clusters = int(labels.max() + 1) if labels.max() >= 0 else 0
        noise_count = int((labels == -1).sum())
        noise_fraction = noise_count / len(labels) if len(labels) > 0 else 0.0

        logger.info(
            "Found %d clusters, %d noise points (%.1f%%)",
            n_clusters,
            noise_count,
            noise_fraction * 100,
        )
        if n_clusters > 0:
            for cid in range(n_clusters):
                count = int((labels == cid).sum())
                logger.info("  Cluster %d: %d members", cid, count)

        return ClusterResult(
            labels=labels,
            probabilities=probabilities,
            outlier_scores=outlier_scores,
            n_clusters=n_clusters,
            noise_count=noise_count,
            noise_fraction=noise_fraction,
        )

    def compute_quality_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> QualityMetrics:
        """Compute cluster validation metrics.

        Args:
            features: Feature matrix used for clustering ``(N, D)``.
            labels: Cluster labels from HDBSCAN (may include -1 noise).

        Returns:
            QualityMetrics with silhouette, Davies-Bouldin, etc.
        """
        # Mask out noise for metrics that require ≥2 clusters with ≥1 member
        non_noise = labels >= 0
        n_clusters = int(labels[non_noise].max() + 1) if non_noise.any() else 0

        if n_clusters < 2 or non_noise.sum() < n_clusters + 1:
            logger.warning(
                "Cannot compute quality metrics: %d clusters, %d non-noise points",
                n_clusters,
                int(non_noise.sum()),
            )
            return QualityMetrics(
                silhouette=float("nan"),
                davies_bouldin=float("nan"),
                calinski_harabasz=float("nan"),
                dbcv=float("nan"),
                per_cluster_silhouette={},
            )

        feat_nn = features[non_noise]
        labels_nn = labels[non_noise]

        # Silhouette
        sil = float(silhouette_score(feat_nn, labels_nn))
        sil_samples = silhouette_samples(feat_nn, labels_nn)
        per_cluster_sil: dict[int, float] = {}
        for cid in range(n_clusters):
            mask = labels_nn == cid
            if mask.any():
                per_cluster_sil[cid] = float(sil_samples[mask].mean())

        # Davies-Bouldin
        db = float(davies_bouldin_score(feat_nn, labels_nn))

        # Calinski-Harabasz
        ch = float(calinski_harabasz_score(feat_nn, labels_nn))

        # DBCV via hdbscan.validity_index
        try:
            dbcv = float(hdbscan.validity_index(feat_nn.astype(np.float64), labels_nn))
        except Exception:
            logger.warning("DBCV computation failed, setting to NaN")
            dbcv = float("nan")

        logger.info(
            "Quality metrics: silhouette=%.3f, DB=%.3f, CH=%.1f, DBCV=%.3f",
            sil,
            db,
            ch,
            dbcv,
        )

        return QualityMetrics(
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            dbcv=dbcv,
            per_cluster_silhouette=per_cluster_sil,
        )

    def evaluate_stability(
        self,
        features: np.ndarray,
        n_bootstrap: int = 100,
    ) -> StabilityResult:
        """Bootstrap stability analysis.

        Resamples data, reclusters, and measures adjusted Rand index
        against the full-data labels.

        Args:
            features: Feature matrix ``(N, D)``.
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            StabilityResult with ARI statistics and per-cluster recovery.
        """
        from sklearn.metrics import adjusted_rand_score

        # Fit on full data first
        full_result = self.fit(features)
        full_labels = full_result.labels
        n = len(features)
        rng = np.random.RandomState(self.config.min_cluster_size)

        ari_scores: list[float] = []
        cluster_ids = [cid for cid in range(full_result.n_clusters)]
        recovery_counts: dict[int, int] = {cid: 0 for cid in cluster_ids}

        logger.info("Bootstrap stability: %d iterations", n_bootstrap)

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n, size=n, replace=True)
            boot_features = features[indices]
            boot_full_labels = full_labels[indices]

            # Recluster
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_method=self.config.cluster_selection_method,
            )
            boot_labels = clusterer.fit_predict(boot_features)

            # ARI on non-noise points (both assignments must be non-noise)
            valid = (boot_full_labels >= 0) & (boot_labels >= 0)
            if valid.sum() > 1:
                ari = adjusted_rand_score(boot_full_labels[valid], boot_labels[valid])
                ari_scores.append(ari)

            # Check which original clusters are recovered
            boot_unique = set(boot_labels[boot_labels >= 0])
            for cid in cluster_ids:
                original_mask = boot_full_labels == cid
                if original_mask.sum() == 0:
                    continue
                boot_for_cluster = boot_labels[original_mask]
                # A cluster is "recovered" if >50% of its original members
                # are assigned to the same boot cluster
                boot_for_cluster_nn = boot_for_cluster[boot_for_cluster >= 0]
                if len(boot_for_cluster_nn) > 0:
                    counts = np.bincount(boot_for_cluster_nn)
                    if counts.max() > len(boot_for_cluster_nn) * 0.5:
                        recovery_counts[cid] += 1

        mean_ari = float(np.mean(ari_scores)) if ari_scores else float("nan")
        std_ari = float(np.std(ari_scores)) if ari_scores else float("nan")
        per_cluster_recovery = {
            cid: count / n_bootstrap for cid, count in recovery_counts.items()
        }

        logger.info(
            "Bootstrap stability: ARI=%.3f ± %.3f (%d valid iterations)",
            mean_ari,
            std_ari,
            len(ari_scores),
        )

        return StabilityResult(
            mean_ari=mean_ari,
            std_ari=std_ari,
            per_cluster_recovery=per_cluster_recovery,
            n_bootstrap=n_bootstrap,
        )


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------


def prepare_ablation_features(
    name: str,
    feature_store: FeatureStore,
    dino_weight: float = 0.7,
    texture_weight: float = 0.3,
) -> np.ndarray:
    """Load features for a named ablation.

    Args:
        name: Ablation name (``"dino_only"``, ``"texture_only"``,
            ``"fused_default"``, ``"fused_equal"``).
        feature_store: Feature store to load from.
        dino_weight: DINOv3 weight for fused ablations.
        texture_weight: Texture weight for fused ablations.

    Returns:
        Feature matrix ``(N, D)``.
    """
    if name == "dino_only":
        return feature_store.load_dino()
    elif name == "texture_only":
        return feature_store.load_texture()
    elif name == "fused_default":
        return feature_store.load_fused()
    elif name == "fused_equal":
        # Re-run fusion with custom weights
        dino = feature_store.load_dino()
        texture = feature_store.load_texture()
        config = FusionConfig(dino_weight=dino_weight, texture_weight=texture_weight)
        result = fuse_features(dino, texture, config)
        return result.fused
    else:
        raise ValueError(f"Unknown ablation name: {name!r}")


def run_ablation(
    name: str,
    features: np.ndarray,
    cluster_config: ClusterConfig,
    reduction_configs: dict[str, UMAPConfig],
    return_fitted: bool = False,
) -> AblationResult:
    """Run a complete ablation: UMAP high-d → HDBSCAN → quality → UMAP 2d/3d.

    Args:
        name: Ablation name for logging and output prefixing.
        features: Raw feature matrix ``(N, D)``.
        cluster_config: HDBSCAN configuration.
        reduction_configs: Must contain ``"high_dim"``, ``"viz_2d"``, ``"viz_3d"`` keys.
        return_fitted: If True, attach a ``FittedPipeline`` with the fitted
            reducers and clusterer for later inference.

    Returns:
        AblationResult with cluster result, quality metrics, and UMAP embeddings.
    """
    logger.info("=== Ablation: %s (%d samples, %d features) ===", name, *features.shape)

    reducer = DimensionalityReducer()

    # High-dimensional UMAP for clustering
    high_dim_result = reducer.fit_transform(features, reduction_configs["high_dim"])

    # Cluster
    discoverer = PatternClusterDiscovery(cluster_config)
    cluster_result = discoverer.fit(high_dim_result.embeddings)

    # Quality metrics on the high-dim UMAP space
    quality = discoverer.compute_quality_metrics(
        high_dim_result.embeddings, cluster_result.labels
    )

    # 2D and 3D UMAP for visualization
    reducer_2d = DimensionalityReducer()
    umap_2d = reducer_2d.fit_transform(features, reduction_configs["viz_2d"])

    reducer_3d = DimensionalityReducer()
    umap_3d = reducer_3d.fit_transform(features, reduction_configs["viz_3d"])

    pipeline = None
    if return_fitted:
        pipeline = FittedPipeline(
            umap_high_dim=reducer,
            umap_2d=reducer_2d,
            umap_3d=reducer_3d,
            hdbscan_discoverer=discoverer,
        )

    return AblationResult(
        name=name,
        features_used=name,
        cluster_result=cluster_result,
        quality_metrics=quality,
        umap_2d=umap_2d,
        umap_3d=umap_3d,
        fitted_pipeline=pipeline,
    )

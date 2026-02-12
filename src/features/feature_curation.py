"""Feature curation: remove scene-content directions from DINO embeddings.

Uses orthogonal projection to remove directions in DINO feature space that
encode scene content (e.g. tape measures, sky, horizon) rather than desert
surface patterns.  This is a pre-clustering step for the two-stage curated
clustering pipeline.

Approach:
  1. Compute centroids of "scene" clusters and "pattern" clusters from
     Stage 1 (triage) clustering.
  2. Difference vectors ``d_i = c_scene_i - c_pattern`` encode what makes
     each scene cluster distinct.
  3. PCA on the difference matrix yields the top orthonormal *scene
     directions* V.
  4. Project out:  ``X_curated = X - (X @ V^T) @ V``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CurationConfig:
    """Configuration for feature curation."""

    n_directions: int = 3
    """Number of scene directions to remove (clamped to n_scene_clusters)."""

    method: str = "centroid_pca"
    """Direction computation method (only ``centroid_pca`` currently)."""

    min_eigenvalue: float = 1e-10
    """Skip directions with eigenvalue below this threshold."""


def load_curation_config(config_dict: dict) -> CurationConfig:
    """Create a CurationConfig from a config dictionary.

    Args:
        config_dict: Dictionary with curation configuration values.

    Returns:
        Populated CurationConfig.
    """
    return CurationConfig(
        n_directions=config_dict.get("n_directions", CurationConfig.n_directions),
        method=config_dict.get("method", CurationConfig.method),
        min_eigenvalue=config_dict.get("min_eigenvalue", CurationConfig.min_eigenvalue),
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CurationResult:
    """Output of feature curation."""

    curated_features: np.ndarray
    """(N, D) projected features with scene directions removed."""

    removed_directions: np.ndarray
    """(n_dirs, D) orthonormal directions that were removed."""

    variance_explained: np.ndarray
    """(n_dirs,) variance along each removed direction."""

    total_variance_fraction: float
    """Fraction of total feature variance removed."""

    n_directions_removed: int
    """Actual count of directions removed (may be < requested)."""

    scene_centroid: np.ndarray
    """(D,) mean centroid of scene clusters."""

    pattern_centroid: np.ndarray
    """(D,) mean centroid of kept (pattern) clusters."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_scene_directions(
    features: np.ndarray,
    labels: np.ndarray,
    scene_cluster_ids: list[int],
    keep_cluster_ids: list[int] | None = None,
    n_directions: int = 3,
    min_eigenvalue: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal directions encoding scene content.

    Args:
        features: Raw DINO embeddings ``(N, D)``.
        labels: Stage 1 cluster labels ``(N,)``.  ``-1`` = noise.
        scene_cluster_ids: Cluster IDs identified as scene-contaminated.
        keep_cluster_ids: Cluster IDs to treat as pattern clusters.
            If ``None``, all non-noise, non-scene clusters are used.
        n_directions: Maximum number of directions to extract.
        min_eigenvalue: Skip directions below this eigenvalue.

    Returns:
        Tuple of ``(directions, eigenvalues, scene_centroid, pattern_centroid)``
        where *directions* is ``(K, D)`` orthonormal and *eigenvalues* is
        ``(K,)`` with ``K <= n_directions``.

    Raises:
        ValueError: If ``scene_cluster_ids`` is empty or if all non-noise
            labels are scene clusters.
    """
    if not scene_cluster_ids:
        raise ValueError("scene_cluster_ids must not be empty")

    scene_set = set(scene_cluster_ids)
    unique_labels = set(labels[labels >= 0])

    # Determine keep IDs
    if keep_cluster_ids is not None:
        keep_set = set(keep_cluster_ids)
    else:
        keep_set = unique_labels - scene_set

    if not keep_set:
        raise ValueError(
            "All non-noise clusters are scene clusters — cannot compute "
            "pattern centroid"
        )

    # Validate scene cluster IDs exist in labels
    valid_scene_ids = []
    for sid in scene_cluster_ids:
        if sid not in unique_labels:
            logger.warning(
                "Scene cluster ID %d not found in labels (present: %s) — skipping",
                sid,
                sorted(unique_labels),
            )
        elif (labels == sid).sum() == 0:
            logger.warning("Scene cluster %d has 0 members — skipping", sid)
        else:
            valid_scene_ids.append(sid)

    if not valid_scene_ids:
        raise ValueError("No valid scene cluster IDs remain after filtering")

    n_scene = len(valid_scene_ids)

    # Clamp n_directions
    if n_directions > n_scene:
        logger.warning(
            "n_directions=%d > n_scene_clusters=%d — clamping to %d",
            n_directions,
            n_scene,
            n_scene,
        )
        n_directions = n_scene

    # Compute centroids
    pattern_mask = np.isin(labels, list(keep_set))
    pattern_centroid = features[pattern_mask].mean(axis=0)

    scene_centroids = []
    for sid in valid_scene_ids:
        mask = labels == sid
        scene_centroids.append(features[mask].mean(axis=0))

    # Scene centroid (mean of all scene cluster centroids)
    scene_centroid = np.mean(scene_centroids, axis=0)

    # Difference vectors: what makes each scene cluster distinct
    diff_matrix = np.array([sc - pattern_centroid for sc in scene_centroids])
    # diff_matrix shape: (n_scene, D)

    if n_scene == 1:
        # Single scene cluster → single direction = normalized difference
        d = diff_matrix[0]
        norm = np.linalg.norm(d)
        if norm < min_eigenvalue:
            logger.warning(
                "Scene direction has near-zero norm (%.2e) — no directions removed",
                norm,
            )
            return (
                np.empty((0, features.shape[1])),
                np.empty((0,)),
                scene_centroid,
                pattern_centroid,
            )
        direction = (d / norm).reshape(1, -1)
        # Variance along this direction
        variance = float(np.var(features @ direction.T))
        return direction, np.array([variance]), scene_centroid, pattern_centroid

    # Multiple scene clusters → PCA on the difference matrix
    # Center the difference vectors (important for PCA)
    diff_mean = diff_matrix.mean(axis=0)
    diff_centered = diff_matrix - diff_mean

    # SVD to get principal directions
    # diff_centered is (n_scene, D) — SVD gives us directions in D-space
    U, S, Vt = np.linalg.svd(diff_centered, full_matrices=False)
    # Vt rows are principal directions; S are singular values
    # Eigenvalues of covariance = S^2 / (n-1)
    eigenvalues = (S ** 2) / max(n_scene - 1, 1)

    # Select directions above min_eigenvalue threshold
    directions = []
    kept_eigenvalues = []
    for i in range(min(n_directions, len(eigenvalues))):
        if eigenvalues[i] < min_eigenvalue:
            logger.warning(
                "Direction %d eigenvalue %.2e < min_eigenvalue %.2e — skipping",
                i,
                eigenvalues[i],
                min_eigenvalue,
            )
            break
        directions.append(Vt[i])
        # Compute actual variance of features along this direction
        proj = features @ Vt[i]
        kept_eigenvalues.append(float(np.var(proj)))

    if not directions:
        return (
            np.empty((0, features.shape[1])),
            np.empty((0,)),
            scene_centroid,
            pattern_centroid,
        )

    return (
        np.array(directions),
        np.array(kept_eigenvalues),
        scene_centroid,
        pattern_centroid,
    )


def project_out_directions(
    features: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Remove specified directions from features via orthogonal projection.

    Computes ``X_curated = X - (X @ V^T) @ V`` where V is the matrix of
    directions to remove.  This zeroes the component of each feature vector
    along each removed direction while preserving orthogonal components.

    Args:
        features: Feature matrix ``(N, D)``.
        directions: Orthonormal directions ``(K, D)`` to project out.

    Returns:
        Curated feature matrix ``(N, D)`` — same shape, scene signal removed.
    """
    if directions.shape[0] == 0:
        return features.copy()

    # X_curated = X - (X @ V^T) @ V
    # V is (K, D), V^T is (D, K)
    # (X @ V^T) is (N, K) — projection coefficients
    # ((X @ V^T) @ V) is (N, D) — reconstruction of scene components
    projections = features @ directions.T  # (N, K)
    scene_component = projections @ directions  # (N, D)
    return features - scene_component


def curate_features(
    features: np.ndarray,
    labels: np.ndarray,
    scene_cluster_ids: list[int],
    keep_cluster_ids: list[int] | None = None,
    config: CurationConfig | None = None,
) -> CurationResult:
    """Main entry point: compute scene directions and project them out.

    Args:
        features: Raw DINO embeddings ``(N, D)``.
        labels: Stage 1 cluster labels ``(N,)``.
        scene_cluster_ids: Cluster IDs identified as scene-contaminated.
        keep_cluster_ids: Cluster IDs to treat as pattern clusters.
            If ``None``, all non-noise, non-scene clusters are used.
        config: Curation configuration.

    Returns:
        CurationResult with curated features and diagnostics.
    """
    config = config or CurationConfig()

    logger.info(
        "Feature curation: %d samples, %d dims, scene_clusters=%s, "
        "n_directions=%d",
        features.shape[0],
        features.shape[1],
        scene_cluster_ids,
        config.n_directions,
    )

    directions, eigenvalues, scene_centroid, pattern_centroid = (
        compute_scene_directions(
            features,
            labels,
            scene_cluster_ids,
            keep_cluster_ids=keep_cluster_ids,
            n_directions=config.n_directions,
            min_eigenvalue=config.min_eigenvalue,
        )
    )

    n_removed = directions.shape[0]
    logger.info("Removing %d scene directions", n_removed)

    # Compute total variance fraction being removed
    total_var = float(np.var(features, axis=0).sum())
    removed_var = float(eigenvalues.sum()) if len(eigenvalues) > 0 else 0.0
    fraction = removed_var / total_var if total_var > 0 else 0.0

    logger.info(
        "Variance removed: %.4f (%.1f%% of total)",
        removed_var,
        fraction * 100,
    )

    curated = project_out_directions(features, directions)

    return CurationResult(
        curated_features=curated,
        removed_directions=directions,
        variance_explained=eigenvalues,
        total_variance_fraction=fraction,
        n_directions_removed=n_removed,
        scene_centroid=scene_centroid,
        pattern_centroid=pattern_centroid,
    )

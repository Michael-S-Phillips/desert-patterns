"""Feature fusion: L2-normalize, PCA, weighted concatenation.

Combines DINOv3 embeddings (768-d) and texture descriptors (~90-d) into a
single fused feature vector for downstream clustering.

Fusion pipeline:
  1. L2-normalize each feature set independently
  2. PCA on each (retain 95% variance by default)
  3. Weighted concatenation: ``[w_dino * dino_pca | w_tex * texture_pca]``
  4. Final L2-normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Configuration for feature fusion."""

    dino_weight: float = 0.7
    texture_weight: float = 0.3
    pca_variance_threshold: float = 0.95
    normalize: bool = True


def load_fusion_config(config_dict: dict) -> FusionConfig:
    """Create a FusionConfig from a config dictionary.

    Args:
        config_dict: Dictionary with fusion configuration values.

    Returns:
        Populated FusionConfig.
    """
    return FusionConfig(
        dino_weight=config_dict.get("dino_weight", FusionConfig.dino_weight),
        texture_weight=config_dict.get("texture_weight", FusionConfig.texture_weight),
        pca_variance_threshold=config_dict.get(
            "pca_variance_threshold", FusionConfig.pca_variance_threshold
        ),
        normalize=config_dict.get("normalize", FusionConfig.normalize),
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    """Output of the feature fusion pipeline."""

    dino_pca: np.ndarray  # (N, D1) PCA-reduced DINOv3 embeddings
    texture_pca: np.ndarray  # (N, D2) PCA-reduced texture features
    fused: np.ndarray  # (N, D1+D2) final fused vectors
    dino_pca_model: PCA  # Fitted PCA model for DINOv3
    texture_pca_model: PCA  # Fitted PCA model for texture
    dino_variance_explained: float  # Cumulative variance captured
    texture_variance_explained: float  # Cumulative variance captured
    dino_n_components: int  # Number of PCA components kept
    texture_n_components: int  # Number of PCA components kept


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.

    Zero-norm rows are left as zeros (not NaN).

    Args:
        features: 2-D array of shape ``(N, D)``.

    Returns:
        L2-normalized array of the same shape.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return features / norms


def fuse_features(
    dino: np.ndarray,
    texture: np.ndarray,
    config: FusionConfig | None = None,
) -> FusionResult:
    """Fuse DINOv3 and texture features via PCA + weighted concatenation.

    Args:
        dino: DINOv3 embeddings of shape ``(N, 768)``.
        texture: Texture features of shape ``(N, 90)``.
        config: Fusion configuration.

    Returns:
        FusionResult with PCA-reduced and fused arrays.
    """
    config = config or FusionConfig()
    n = dino.shape[0]
    assert texture.shape[0] == n, (
        f"Sample count mismatch: dino={n}, texture={texture.shape[0]}"
    )

    logger.info("Fusing features: %d samples, dino=%s, texture=%s", n, dino.shape, texture.shape)

    # 1. L2-normalize each set
    dino_norm = l2_normalize(dino.astype(np.float64))
    texture_norm = l2_normalize(texture.astype(np.float64))

    # 2. PCA on each
    dino_pca_model = PCA(
        n_components=config.pca_variance_threshold, random_state=42, svd_solver="full"
    )
    dino_pca = dino_pca_model.fit_transform(dino_norm)
    dino_var = float(dino_pca_model.explained_variance_ratio_.sum())
    logger.info(
        "DINOv3 PCA: %d → %d components (%.1f%% variance)",
        dino.shape[1],
        dino_pca.shape[1],
        dino_var * 100,
    )

    texture_pca_model = PCA(
        n_components=config.pca_variance_threshold, random_state=42, svd_solver="full"
    )
    texture_pca = texture_pca_model.fit_transform(texture_norm)
    texture_var = float(texture_pca_model.explained_variance_ratio_.sum())
    logger.info(
        "Texture PCA: %d → %d components (%.1f%% variance)",
        texture.shape[1],
        texture_pca.shape[1],
        texture_var * 100,
    )

    # 3. Weighted concatenation
    fused = np.hstack([
        config.dino_weight * dino_pca,
        config.texture_weight * texture_pca,
    ])

    # 4. Final L2-normalization
    if config.normalize:
        fused = l2_normalize(fused)

    logger.info("Fused feature dimension: %d", fused.shape[1])

    return FusionResult(
        dino_pca=dino_pca.astype(np.float32),
        texture_pca=texture_pca.astype(np.float32),
        fused=fused.astype(np.float32),
        dino_pca_model=dino_pca_model,
        texture_pca_model=texture_pca_model,
        dino_variance_explained=dino_var,
        texture_variance_explained=texture_var,
        dino_n_components=dino_pca.shape[1],
        texture_n_components=texture_pca.shape[1],
    )

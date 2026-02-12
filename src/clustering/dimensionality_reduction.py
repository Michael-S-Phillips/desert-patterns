"""Dimensionality reduction via UMAP and t-SNE.

Provides two reduction pathways:
  1. **High-dimensional reduction** (features → 10-20 dims) — input to HDBSCAN.
  2. **2D/3D reduction** (features → 2-3 dims) — for visualization.

Also supports t-SNE with multiple perplexity values for method-independent
verification of cluster structure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UMAPConfig:
    """Configuration for a single UMAP reduction."""

    n_components: int = 2
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class TSNEConfig:
    """Configuration for t-SNE reductions."""

    perplexities: list[int] = field(default_factory=lambda: [15, 30, 50])
    n_components: int = 2
    random_state: int = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ReductionResult:
    """Output of a single dimensionality reduction run."""

    embeddings: np.ndarray  # (N, n_components)
    method: str  # "umap" or "tsne"
    config: UMAPConfig | TSNEConfig
    fit_time_seconds: float


# ---------------------------------------------------------------------------
# Reducer class
# ---------------------------------------------------------------------------


class DimensionalityReducer:
    """UMAP and t-SNE dimensionality reduction.

    Stores the fitted UMAP model so that new points can be projected
    into the same embedding space via :meth:`transform`.
    """

    def __init__(self) -> None:
        self._fitted_model: UMAP | None = None

    def fit_transform(
        self,
        features: np.ndarray,
        config: UMAPConfig | None = None,
    ) -> ReductionResult:
        """Run UMAP dimensionality reduction.

        Args:
            features: Feature matrix of shape ``(N, D)``.
            config: UMAP configuration.  Defaults to standard 2-D settings.

        Returns:
            ReductionResult with the low-dimensional embeddings.
        """
        config = config or UMAPConfig()
        logger.info(
            "UMAP: %d samples, %d → %d dims (n_neighbors=%d, min_dist=%.2f, metric=%s)",
            features.shape[0],
            features.shape[1],
            config.n_components,
            config.n_neighbors,
            config.min_dist,
            config.metric,
        )

        model = UMAP(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
        )

        t0 = time.perf_counter()
        embeddings = model.fit_transform(features)
        elapsed = time.perf_counter() - t0

        self._fitted_model = model
        logger.info("UMAP completed in %.1f s → shape %s", elapsed, embeddings.shape)

        return ReductionResult(
            embeddings=embeddings.astype(np.float32),
            method="umap",
            config=config,
            fit_time_seconds=elapsed,
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project new points into the fitted UMAP space.

        Args:
            features: Feature matrix of shape ``(M, D)``.

        Returns:
            Projected embeddings of shape ``(M, n_components)``.

        Raises:
            RuntimeError: If no UMAP model has been fitted yet.
        """
        if self._fitted_model is None:
            raise RuntimeError("No UMAP model fitted yet — call fit_transform first")
        return self._fitted_model.transform(features).astype(np.float32)

    def fit_transform_tsne(
        self,
        features: np.ndarray,
        perplexity: int,
        config: TSNEConfig | None = None,
    ) -> ReductionResult:
        """Run t-SNE with a single perplexity value.

        Args:
            features: Feature matrix of shape ``(N, D)``.
            perplexity: Perplexity for this run.
            config: t-SNE configuration.

        Returns:
            ReductionResult with the 2-D embeddings.
        """
        config = config or TSNEConfig()
        logger.info(
            "t-SNE: %d samples, %d → %d dims (perplexity=%d)",
            features.shape[0],
            features.shape[1],
            config.n_components,
            perplexity,
        )

        model = TSNE(
            n_components=config.n_components,
            perplexity=perplexity,
            random_state=config.random_state,
        )

        t0 = time.perf_counter()
        embeddings = model.fit_transform(features)
        elapsed = time.perf_counter() - t0

        logger.info("t-SNE (perplexity=%d) completed in %.1f s", perplexity, elapsed)

        return ReductionResult(
            embeddings=embeddings.astype(np.float32),
            method="tsne",
            config=config,
            fit_time_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_reduction_config(config_dict: dict) -> dict[str, UMAPConfig | TSNEConfig]:
    """Parse dimensionality reduction config sections.

    Expects keys: ``high_dim``, ``viz_2d``, ``viz_3d``, ``tsne``.

    Args:
        config_dict: Raw dictionary from YAML config.

    Returns:
        Mapping from section name to config dataclass.
    """
    configs: dict[str, UMAPConfig | TSNEConfig] = {}

    for key in ("high_dim", "viz_2d", "viz_3d"):
        section = config_dict.get(key, {})
        configs[key] = UMAPConfig(
            n_components=section.get("n_components", 2 if key != "high_dim" else 15),
            n_neighbors=section.get("n_neighbors", 30),
            min_dist=section.get("min_dist", 0.1),
            metric=section.get("metric", "cosine"),
            random_state=section.get("random_state", 42),
        )

    tsne_section = config_dict.get("tsne", {})
    configs["tsne"] = TSNEConfig(
        perplexities=tsne_section.get("perplexities", [15, 30, 50]),
        n_components=tsne_section.get("n_components", 2),
        random_state=tsne_section.get("random_state", 42),
    )

    return configs

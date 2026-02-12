"""Continuous pattern space analysis: KDE density + measurement correlation.

Instead of forcing discrete categories, this module treats the 2-D UMAP
embedding as a continuous pattern space and provides:
- Kernel density estimation to find pattern "hotspots"
- Correlation with external measurements (Spearman + random forest)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ContinuousSpaceConfig:
    """Configuration for continuous pattern space analysis."""

    kde_bandwidth: str | float = "scott"
    n_grid: int = 100


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ContinuousSpaceResult:
    """Output of kernel density estimation."""

    coordinates: np.ndarray  # (N, 2) original UMAP coordinates
    kde_values: np.ndarray  # (n_grid, n_grid) evaluated density
    density_grid_x: np.ndarray  # (n_grid,) x-axis grid points
    density_grid_y: np.ndarray  # (n_grid,) y-axis grid points


@dataclass
class CorrelationResult:
    """Output of measurement correlation analysis."""

    spearman_dim1: tuple[float, float]  # (r, p) for UMAP dim 1
    spearman_dim2: tuple[float, float]  # (r, p) for UMAP dim 2
    combined_r2: float  # R² from random forest


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class ContinuousPatternSpace:
    """Continuous 2-D pattern space analysis.

    Args:
        embeddings_2d: UMAP 2-D coordinates ``(N, 2)``.
        image_ids: Image/patch identifiers ``(N,)``.
        metadata_df: DataFrame with at least ``image_id``; may include
            ``temporal_phase``, ``lat``, ``lon``.
    """

    def __init__(
        self,
        embeddings_2d: np.ndarray,
        image_ids: list[str],
        metadata_df: pd.DataFrame,
    ) -> None:
        self.embeddings_2d = embeddings_2d
        self.image_ids = image_ids
        self.metadata_df = metadata_df

        self._id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

        # Find the ID column in metadata
        self._id_col = "image_id"
        if self._id_col not in metadata_df.columns:
            for candidate in ("patch_id", "id"):
                if candidate in metadata_df.columns:
                    self._id_col = candidate
                    break

    def kernel_density(
        self,
        phase: str | None = None,
        config: ContinuousSpaceConfig | None = None,
    ) -> ContinuousSpaceResult:
        """Compute 2-D Gaussian KDE on pattern space coordinates.

        Args:
            phase: If provided, restrict to images with this temporal_phase.
            config: KDE configuration.

        Returns:
            ContinuousSpaceResult with density grid.
        """
        config = config or ContinuousSpaceConfig()

        coords = self.embeddings_2d
        if phase is not None:
            indices = self._get_phase_indices(phase)
            if len(indices) < 2:
                logger.warning(
                    "Phase %r has %d points — too few for KDE, using all points",
                    phase,
                    len(indices),
                )
                indices = list(range(len(self.image_ids)))
            coords = self.embeddings_2d[indices]

        logger.info("KDE: %d points, bandwidth=%s, grid=%d", len(coords),
                     config.kde_bandwidth, config.n_grid)

        # Fit KDE
        xy = coords.T  # (2, N)
        kde = gaussian_kde(xy, bw_method=config.kde_bandwidth)

        # Evaluate on grid
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        # Add margin
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        grid_x = np.linspace(x_min - margin_x, x_max + margin_x, config.n_grid)
        grid_y = np.linspace(y_min - margin_y, y_max + margin_y, config.n_grid)
        xx, yy = np.meshgrid(grid_x, grid_y)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(config.n_grid, config.n_grid)

        return ContinuousSpaceResult(
            coordinates=coords.astype(np.float32),
            kde_values=density.astype(np.float32),
            density_grid_x=grid_x.astype(np.float32),
            density_grid_y=grid_y.astype(np.float32),
        )

    def correlate_with_measurements(
        self,
        measurement_df: pd.DataFrame,
        measurement_col: str,
        lat_col: str = "lat",
        lon_col: str = "lon",
    ) -> CorrelationResult:
        """Correlate pattern space coordinates with external measurements.

        Matches measurement locations to the nearest image by GPS, then
        computes Spearman correlation per UMAP dimension and a combined
        R² from a random forest.

        Args:
            measurement_df: DataFrame with measurement values and GPS.
            measurement_col: Column name for the measurement values.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.

        Returns:
            CorrelationResult with Spearman and RF R² values.
        """
        from sklearn.ensemble import RandomForestRegressor

        # Match measurements to nearest images
        if lat_col not in measurement_df.columns or lon_col not in measurement_df.columns:
            logger.warning("Missing GPS columns in measurement data")
            return CorrelationResult(
                spearman_dim1=(0.0, 1.0),
                spearman_dim2=(0.0, 1.0),
                combined_r2=0.0,
            )

        # Get image GPS from metadata
        if "lat" not in self.metadata_df.columns or "lon" not in self.metadata_df.columns:
            logger.warning("No GPS in image metadata — cannot correlate with measurements")
            return CorrelationResult(
                spearman_dim1=(0.0, 1.0),
                spearman_dim2=(0.0, 1.0),
                combined_r2=0.0,
            )

        # Build array of image GPS
        meta_with_gps = self.metadata_df.dropna(subset=["lat", "lon"])
        if meta_with_gps.empty:
            logger.warning("No images with GPS — cannot correlate")
            return CorrelationResult(
                spearman_dim1=(0.0, 1.0),
                spearman_dim2=(0.0, 1.0),
                combined_r2=0.0,
            )

        matched_umap = []
        matched_values = []

        for _, mrow in measurement_df.iterrows():
            mlat, mlon = mrow[lat_col], mrow[lon_col]
            if pd.isna(mlat) or pd.isna(mlon):
                continue

            # Find nearest image
            best_dist = float("inf")
            best_idx = -1
            for _, irow in meta_with_gps.iterrows():
                iid = irow[self._id_col]
                if iid not in self._id_to_idx:
                    continue
                dist = _haversine_m(mlat, mlon, irow["lat"], irow["lon"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = self._id_to_idx[iid]

            if best_idx >= 0:
                matched_umap.append(self.embeddings_2d[best_idx])
                matched_values.append(mrow[measurement_col])

        if len(matched_umap) < 3:
            logger.warning("Only %d matched points — too few for correlation", len(matched_umap))
            return CorrelationResult(
                spearman_dim1=(0.0, 1.0),
                spearman_dim2=(0.0, 1.0),
                combined_r2=0.0,
            )

        umap_arr = np.array(matched_umap)
        values = np.array(matched_values, dtype=np.float64)

        # Spearman per dimension
        r1, p1 = spearmanr(umap_arr[:, 0], values)
        r2, p2 = spearmanr(umap_arr[:, 1], values)

        # Random forest R²
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(umap_arr, values)
        r2_score = float(rf.score(umap_arr, values))

        logger.info(
            "Measurement correlation: Spearman dim1=(%.3f, p=%.4f), "
            "dim2=(%.3f, p=%.4f), RF R²=%.3f",
            r1, p1, r2, p2, r2_score,
        )

        return CorrelationResult(
            spearman_dim1=(float(r1), float(p1)),
            spearman_dim2=(float(r2), float(p2)),
            combined_r2=r2_score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_phase_indices(self, phase: str) -> list[int]:
        """Get indices of images with a given temporal phase."""
        if "temporal_phase" not in self.metadata_df.columns:
            return []

        phase_ids = self.metadata_df[self.metadata_df["temporal_phase"] == phase][
            self._id_col
        ].tolist()
        return [self._id_to_idx[iid] for iid in phase_ids if iid in self._id_to_idx]


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two GPS points in metres."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

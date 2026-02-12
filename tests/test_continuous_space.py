"""Tests for continuous pattern space analysis."""

import numpy as np
import pandas as pd
import pytest

from src.clustering.continuous_space import (
    ContinuousPatternSpace,
    ContinuousSpaceConfig,
    ContinuousSpaceResult,
    CorrelationResult,
)


@pytest.fixture
def pattern_space(mock_metadata_df: pd.DataFrame) -> ContinuousPatternSpace:
    """ContinuousPatternSpace with mock data."""
    rng = np.random.RandomState(42)
    embeddings_2d = rng.randn(200, 2).astype(np.float32)
    image_ids = [f"img_{i:04d}" for i in range(200)]
    return ContinuousPatternSpace(embeddings_2d, image_ids, mock_metadata_df)


class TestKernelDensity:
    """Tests for KDE density estimation."""

    def test_kde_output_shape(self, pattern_space: ContinuousPatternSpace):
        config = ContinuousSpaceConfig(n_grid=50)
        result = pattern_space.kernel_density(config=config)
        assert isinstance(result, ContinuousSpaceResult)
        assert result.kde_values.shape == (50, 50)
        assert result.density_grid_x.shape == (50,)
        assert result.density_grid_y.shape == (50,)

    def test_kde_values_nonnegative(self, pattern_space: ContinuousPatternSpace):
        result = pattern_space.kernel_density(config=ContinuousSpaceConfig(n_grid=30))
        assert (result.kde_values >= 0).all()

    def test_kde_coordinates_match_input(self, pattern_space: ContinuousPatternSpace):
        result = pattern_space.kernel_density()
        assert result.coordinates.shape == (200, 2)

    def test_kde_phase_filtered(self, pattern_space: ContinuousPatternSpace):
        """Phase-filtered KDE may use fewer points."""
        result = pattern_space.kernel_density(
            phase="during_rain", config=ContinuousSpaceConfig(n_grid=30)
        )
        assert isinstance(result, ContinuousSpaceResult)
        # The coordinates may be a subset or full (if too few for filter)
        assert result.kde_values.shape == (30, 30)

    def test_kde_unknown_phase_fallback(self, pattern_space: ContinuousPatternSpace):
        """Phase with no images should fall back to all points."""
        result = pattern_space.kernel_density(
            phase="nonexistent_phase", config=ContinuousSpaceConfig(n_grid=20)
        )
        assert result.coordinates.shape == (200, 2)


class TestMeasurementCorrelation:
    """Tests for measurement correlation."""

    def test_correlation_returns_result(self, pattern_space: ContinuousPatternSpace):
        rng = np.random.RandomState(42)
        measurement_df = pd.DataFrame({
            "lat": rng.uniform(-24.09, -24.08, 20),
            "lon": rng.uniform(-69.92, -69.91, 20),
            "atp_count": rng.uniform(100, 1000, 20),
        })
        result = pattern_space.correlate_with_measurements(
            measurement_df, measurement_col="atp_count"
        )
        assert isinstance(result, CorrelationResult)
        assert -1.0 <= result.spearman_dim1[0] <= 1.0
        assert -1.0 <= result.spearman_dim2[0] <= 1.0

    def test_correlation_missing_gps_graceful(self):
        """No GPS in metadata — should return default values."""
        embeddings = np.random.randn(10, 2).astype(np.float32)
        image_ids = [f"img_{i}" for i in range(10)]
        meta = pd.DataFrame({"image_id": image_ids})
        cps = ContinuousPatternSpace(embeddings, image_ids, meta)

        measurement_df = pd.DataFrame({
            "lat": [1.0], "lon": [1.0], "val": [42.0],
        })
        result = cps.correlate_with_measurements(measurement_df, "val")
        assert result.combined_r2 == 0.0

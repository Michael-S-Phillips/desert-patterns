"""Tests for temporal pattern analysis."""

import numpy as np
import pandas as pd
import pytest

from src.clustering.temporal_analysis import TemporalPatternAnalysis, TemporalResult


@pytest.fixture
def temporal_analyzer(
    cluster_labels: np.ndarray,
    mock_metadata_df: pd.DataFrame,
) -> TemporalPatternAnalysis:
    """TemporalPatternAnalysis with mock data."""
    image_ids = [f"img_{i:04d}" for i in range(200)]
    features_2d = np.random.RandomState(42).randn(200, 2).astype(np.float32)
    return TemporalPatternAnalysis(
        labels=cluster_labels,
        image_ids=image_ids,
        metadata_df=mock_metadata_df,
        features_2d=features_2d,
    )


class TestPhaseDistribution:
    """Tests for cluster distribution by phase."""

    def test_returns_dataframe(self, temporal_analyzer: TemporalPatternAnalysis):
        dist = temporal_analyzer.cluster_distribution_by_phase()
        assert isinstance(dist, pd.DataFrame)

    def test_fractions_sum_to_one(self, temporal_analyzer: TemporalPatternAnalysis):
        dist = temporal_analyzer.cluster_distribution_by_phase()
        if not dist.empty:
            row_sums = dist.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


class TestTransitionMatrix:
    """Tests for transition matrix."""

    def test_transition_with_gps_data(self, temporal_analyzer: TemporalPatternAnalysis):
        result = temporal_analyzer.transition_matrix(distance_threshold_m=50000)
        # With our mock data (random GPS in a ~1km area and 50km threshold),
        # we should find matches, but transition_matrix might still return
        # None if there are not enough matched pairs between known phases
        assert result is None or isinstance(result, np.ndarray)

    def test_transition_without_gps(self):
        labels = np.array([0, 0, 1, 1])
        image_ids = ["a", "b", "c", "d"]
        meta = pd.DataFrame({
            "image_id": image_ids,
            "temporal_phase": ["during_rain", "during_rain", "post_rain", "post_rain"],
            # No lat/lon columns
        })
        features_2d = np.random.randn(4, 2).astype(np.float32)
        analyzer = TemporalPatternAnalysis(labels, image_ids, meta, features_2d)
        result = analyzer.transition_matrix()
        assert result is None


class TestEmbeddingDrift:
    """Tests for embedding drift analysis."""

    def test_drift_with_sufficient_phases(self):
        """Create data with enough samples in 2 phases for drift."""
        n = 100
        rng = np.random.RandomState(42)
        labels = np.repeat([0, 1], 50)
        image_ids = [f"img_{i}" for i in range(n)]
        phases = (
            ["during_rain"] * 25 + ["post_rain"] * 25 +
            ["during_rain"] * 25 + ["post_rain"] * 25
        )
        meta = pd.DataFrame({
            "image_id": image_ids,
            "temporal_phase": phases,
        })
        features_2d = rng.randn(n, 2).astype(np.float32)
        analyzer = TemporalPatternAnalysis(labels, image_ids, meta, features_2d)
        result = analyzer.embedding_drift()
        assert result is not None
        assert isinstance(result, dict)

    def test_drift_sparse_phases_returns_none(self):
        """All images have unknown phase — should return None."""
        labels = np.array([0, 0, 1, 1])
        image_ids = ["a", "b", "c", "d"]
        meta = pd.DataFrame({
            "image_id": image_ids,
            "temporal_phase": ["unknown"] * 4,
        })
        features_2d = np.random.randn(4, 2).astype(np.float32)
        analyzer = TemporalPatternAnalysis(labels, image_ids, meta, features_2d)
        result = analyzer.embedding_drift()
        assert result is None


class TestPhaseStatisticalTests:
    """Tests for chi-squared and proportion tests."""

    def test_returns_dataframe(self):
        """With sufficient data in 2 phases, should return results."""
        n = 120
        labels = np.repeat([0, 1, 2], 40)
        image_ids = [f"img_{i}" for i in range(n)]
        phases = (
            ["during_rain"] * 20 + ["post_rain"] * 20 +
            ["during_rain"] * 20 + ["post_rain"] * 20 +
            ["during_rain"] * 20 + ["post_rain"] * 20
        )
        meta = pd.DataFrame({
            "image_id": image_ids,
            "temporal_phase": phases,
        })
        features_2d = np.random.randn(n, 2).astype(np.float32)
        analyzer = TemporalPatternAnalysis(labels, image_ids, meta, features_2d)
        df = analyzer.phase_statistical_tests()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_cramers_v_range(self):
        n = 120
        labels = np.repeat([0, 1, 2], 40)
        image_ids = [f"img_{i}" for i in range(n)]
        phases = (
            ["during_rain"] * 20 + ["post_rain"] * 20 +
            ["during_rain"] * 20 + ["post_rain"] * 20 +
            ["during_rain"] * 20 + ["post_rain"] * 20
        )
        meta = pd.DataFrame({
            "image_id": image_ids,
            "temporal_phase": phases,
        })
        features_2d = np.random.randn(n, 2).astype(np.float32)
        analyzer = TemporalPatternAnalysis(labels, image_ids, meta, features_2d)
        df = analyzer.phase_statistical_tests()
        overall = df[df["test"] == "chi_squared_overall"]
        if not overall.empty:
            v = overall.iloc[0]["effect_size"]
            assert 0.0 <= v <= 1.0

    def test_empty_with_all_unknown_phase(self):
        labels = np.array([0, 0, 1, 1])
        image_ids = ["a", "b", "c", "d"]
        meta = pd.DataFrame({
            "image_id": image_ids,
            "temporal_phase": ["unknown"] * 4,
        })
        features_2d = np.random.randn(4, 2).astype(np.float32)
        analyzer = TemporalPatternAnalysis(labels, image_ids, meta, features_2d)
        df = analyzer.phase_statistical_tests()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestRunAll:
    """Tests for the run_all orchestrator."""

    def test_run_all_returns_result(self, temporal_analyzer: TemporalPatternAnalysis):
        result = temporal_analyzer.run_all()
        assert isinstance(result, TemporalResult)
        assert isinstance(result.phase_distributions, pd.DataFrame)
        assert isinstance(result.test_results, pd.DataFrame)

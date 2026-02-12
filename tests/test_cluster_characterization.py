"""Tests for cluster characterization and feature importance."""

import numpy as np
import pandas as pd
import pytest

from src.clustering.cluster_characterization import ClusterCharacterizer, ClusterProfile


@pytest.fixture
def characterizer(
    cluster_labels: np.ndarray,
    mock_texture_features: np.ndarray,
    mock_metadata_df: pd.DataFrame,
) -> ClusterCharacterizer:
    """ClusterCharacterizer initialized with standard fixtures."""
    image_ids = [f"img_{i:04d}" for i in range(200)]
    feature_names = [f"texture_{i}" for i in range(90)]
    return ClusterCharacterizer(
        labels=cluster_labels,
        features_texture=mock_texture_features,
        image_ids=image_ids,
        feature_names=feature_names,
        metadata_df=mock_metadata_df,
    )


class TestClusterProfile:
    """Tests for individual cluster profiling."""

    def test_profile_has_expected_fields(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert isinstance(profile, ClusterProfile)
        assert profile.cluster_id == 0
        assert profile.size > 0

    def test_texture_mean_shape(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert profile.texture_mean.shape == (90,)

    def test_texture_std_shape(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert profile.texture_std.shape == (90,)

    def test_distinguishing_features_count(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert len(profile.distinguishing_features) == 10
        for f in profile.distinguishing_features:
            assert "name" in f
            assert "value" in f
            assert "z_score" in f

    def test_representative_ids_count(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert len(profile.representative_ids) <= 25
        assert len(profile.representative_ids) > 0

    def test_boundary_ids_count(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert len(profile.boundary_ids) <= 10

    def test_description_is_nonempty_string(self, characterizer: ClusterCharacterizer):
        profile = characterizer.characterize(0)
        assert isinstance(profile.description, str)
        assert len(profile.description) > 0

    def test_characterize_all_returns_list(self, characterizer: ClusterCharacterizer):
        profiles = characterizer.characterize_all()
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        for p in profiles:
            assert isinstance(p, ClusterProfile)


class TestFeatureImportance:
    """Tests for Kruskal-Wallis feature importance."""

    def test_returns_dataframe(self, characterizer: ClusterCharacterizer):
        df = characterizer.compute_feature_importance()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "h_statistic" in df.columns
        assert "p_value" in df.columns
        assert "p_adjusted" in df.columns
        assert "effect_size" in df.columns

    def test_p_values_in_range(self, characterizer: ClusterCharacterizer):
        df = characterizer.compute_feature_importance()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1).all()
        assert (df["p_adjusted"] >= 0).all()
        assert (df["p_adjusted"] <= 1).all()

    def test_sorted_by_h_statistic(self, characterizer: ClusterCharacterizer):
        df = characterizer.compute_feature_importance()
        h_vals = df["h_statistic"].values
        assert all(h_vals[i] >= h_vals[i + 1] for i in range(len(h_vals) - 1))

    def test_effect_size_nonnegative(self, characterizer: ClusterCharacterizer):
        df = characterizer.compute_feature_importance()
        assert (df["effect_size"] >= 0).all()


class TestSingleCluster:
    """Edge case: only one non-noise cluster."""

    def test_feature_importance_with_single_cluster(self, mock_metadata_df: pd.DataFrame):
        labels = np.zeros(200, dtype=int)  # all cluster 0
        texture = np.random.RandomState(42).randn(200, 90).astype(np.float32)
        image_ids = [f"img_{i:04d}" for i in range(200)]
        feature_names = [f"texture_{i}" for i in range(90)]

        char = ClusterCharacterizer(
            labels=labels,
            features_texture=texture,
            image_ids=image_ids,
            feature_names=feature_names,
            metadata_df=mock_metadata_df,
        )
        df = char.compute_feature_importance()
        # Should return empty or a single-cluster warning
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

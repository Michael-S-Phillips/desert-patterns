"""Tests for feature fusion (L2 normalize + PCA + weighted concat)."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.fusion import FusionConfig, FusionResult, fuse_features, l2_normalize


# ---------------------------------------------------------------------------
# l2_normalize tests
# ---------------------------------------------------------------------------


class TestL2Normalize:
    """Tests for row-wise L2 normalization."""

    def test_unit_norm(self):
        data = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
        normed = l2_normalize(data)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-7)

    def test_zero_vector_stays_zero(self):
        data = np.array([[0.0, 0.0], [1.0, 0.0]])
        normed = l2_normalize(data)
        # Zero row should remain zero (not NaN)
        np.testing.assert_array_equal(normed[0], [0.0, 0.0])
        assert np.linalg.norm(normed[1]) == pytest.approx(1.0, abs=1e-7)

    def test_preserves_direction(self):
        data = np.array([[3.0, 4.0]])
        normed = l2_normalize(data)
        assert normed[0, 0] / normed[0, 1] == pytest.approx(3.0 / 4.0, abs=1e-7)

    def test_already_normalized(self):
        data = np.array([[0.6, 0.8]])
        normed = l2_normalize(data)
        np.testing.assert_allclose(normed, data, atol=1e-7)


# ---------------------------------------------------------------------------
# fuse_features tests
# ---------------------------------------------------------------------------


class TestFuseFeatures:
    """Tests for the full fusion pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample DINOv3 and texture features."""
        rng = np.random.RandomState(42)
        n = 50
        dino = rng.randn(n, 768).astype(np.float32)
        texture = rng.randn(n, 90).astype(np.float32)
        return dino, texture

    def test_returns_fusion_result(self, sample_data):
        dino, texture = sample_data
        result = fuse_features(dino, texture)
        assert isinstance(result, FusionResult)

    def test_pca_reduces_dimensionality(self, sample_data):
        dino, texture = sample_data
        result = fuse_features(dino, texture)
        # PCA should reduce dimensions
        assert result.dino_pca.shape[1] < 768
        assert result.texture_pca.shape[1] < 90
        # But keep the same number of samples
        assert result.dino_pca.shape[0] == 50
        assert result.texture_pca.shape[0] == 50

    def test_fused_shape(self, sample_data):
        dino, texture = sample_data
        result = fuse_features(dino, texture)
        expected_d = result.dino_n_components + result.texture_n_components
        assert result.fused.shape == (50, expected_d)

    def test_fused_is_normalized(self, sample_data):
        dino, texture = sample_data
        result = fuse_features(dino, texture)
        norms = np.linalg.norm(result.fused, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_fused_not_normalized_when_disabled(self, sample_data):
        dino, texture = sample_data
        config = FusionConfig(normalize=False)
        result = fuse_features(dino, texture, config)
        norms = np.linalg.norm(result.fused, axis=1)
        # At least some should differ from 1.0
        assert not np.allclose(norms, 1.0, atol=0.01)

    def test_deterministic_with_seed(self, sample_data):
        dino, texture = sample_data
        r1 = fuse_features(dino, texture)
        r2 = fuse_features(dino, texture)
        np.testing.assert_array_equal(r1.fused, r2.fused)

    def test_variance_explained(self, sample_data):
        dino, texture = sample_data
        result = fuse_features(dino, texture)
        assert result.dino_variance_explained >= 0.95
        assert result.texture_variance_explained >= 0.95

    def test_custom_weights(self, sample_data):
        dino, texture = sample_data
        config = FusionConfig(dino_weight=0.5, texture_weight=0.5)
        result = fuse_features(dino, texture, config)
        assert result.fused.shape[0] == 50

    def test_sample_count_mismatch_raises(self):
        dino = np.random.randn(10, 768).astype(np.float32)
        texture = np.random.randn(5, 90).astype(np.float32)
        with pytest.raises(AssertionError, match="mismatch"):
            fuse_features(dino, texture)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for fusion."""

    def test_single_sample(self):
        """Fusion should work with a single sample (PCA needs n >= n_components)."""
        # With 1 sample, PCA can only keep 1 component
        dino = np.random.RandomState(42).randn(1, 768).astype(np.float32)
        texture = np.random.RandomState(43).randn(1, 90).astype(np.float32)
        # PCA with n_components=0.95 on 1 sample → keeps 1 component
        config = FusionConfig(pca_variance_threshold=0.95)
        result = fuse_features(dino, texture, config)
        assert result.fused.shape[0] == 1

    def test_two_samples(self):
        """Fusion with 2 samples (minimum for meaningful PCA)."""
        rng = np.random.RandomState(42)
        dino = rng.randn(2, 768).astype(np.float32)
        texture = rng.randn(2, 90).astype(np.float32)
        result = fuse_features(dino, texture)
        assert result.fused.shape[0] == 2

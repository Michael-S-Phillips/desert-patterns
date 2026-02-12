"""Tests for dimensionality reduction (UMAP + t-SNE)."""

import numpy as np
import pytest

from src.clustering.dimensionality_reduction import (
    DimensionalityReducer,
    ReductionResult,
    TSNEConfig,
    UMAPConfig,
    load_reduction_config,
)


class TestUMAPReduction:
    """Tests for UMAP dimensionality reduction."""

    def test_umap_output_shape_2d(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        config = UMAPConfig(n_components=2, n_neighbors=15)
        result = reducer.fit_transform(clusterable_features, config)
        assert result.embeddings.shape == (200, 2)

    def test_umap_output_shape_3d(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        config = UMAPConfig(n_components=3, n_neighbors=15)
        result = reducer.fit_transform(clusterable_features, config)
        assert result.embeddings.shape == (200, 3)

    def test_umap_output_shape_high_dim(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        config = UMAPConfig(n_components=10, n_neighbors=15)
        result = reducer.fit_transform(clusterable_features, config)
        assert result.embeddings.shape == (200, 10)

    def test_umap_result_type(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        config = UMAPConfig(n_components=2, n_neighbors=15)
        result = reducer.fit_transform(clusterable_features, config)
        assert isinstance(result, ReductionResult)
        assert result.method == "umap"
        assert result.fit_time_seconds > 0

    def test_umap_reproducibility(self, clusterable_features: np.ndarray):
        config = UMAPConfig(n_components=2, n_neighbors=15, random_state=42)

        r1 = DimensionalityReducer()
        result1 = r1.fit_transform(clusterable_features, config)

        r2 = DimensionalityReducer()
        result2 = r2.fit_transform(clusterable_features, config)

        np.testing.assert_array_almost_equal(result1.embeddings, result2.embeddings)

    def test_umap_transform_new_points(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        config = UMAPConfig(n_components=2, n_neighbors=15)
        reducer.fit_transform(clusterable_features, config)

        new_points = clusterable_features[:10]
        projected = reducer.transform(new_points)
        assert projected.shape == (10, 2)
        assert projected.dtype == np.float32

    def test_umap_transform_without_fit_raises(self):
        reducer = DimensionalityReducer()
        with pytest.raises(RuntimeError, match="No UMAP model fitted"):
            reducer.transform(np.random.randn(5, 10).astype(np.float32))

    def test_umap_embeddings_are_finite(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        config = UMAPConfig(n_components=2, n_neighbors=15)
        result = reducer.fit_transform(clusterable_features, config)
        assert np.all(np.isfinite(result.embeddings))


class TestTSNE:
    """Tests for t-SNE reduction."""

    def test_tsne_output_shape(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        result = reducer.fit_transform_tsne(clusterable_features, perplexity=30)
        assert result.embeddings.shape == (200, 2)

    def test_tsne_result_method(self, clusterable_features: np.ndarray):
        reducer = DimensionalityReducer()
        result = reducer.fit_transform_tsne(clusterable_features, perplexity=15)
        assert result.method == "tsne"
        assert result.fit_time_seconds > 0


class TestConfigLoading:
    """Tests for configuration parsing."""

    def test_load_reduction_config_defaults(self):
        configs = load_reduction_config({})
        assert "high_dim" in configs
        assert "viz_2d" in configs
        assert "viz_3d" in configs
        assert "tsne" in configs
        assert isinstance(configs["high_dim"], UMAPConfig)
        assert isinstance(configs["tsne"], TSNEConfig)

    def test_load_reduction_config_high_dim_components(self):
        configs = load_reduction_config({
            "high_dim": {"n_components": 20},
        })
        assert configs["high_dim"].n_components == 20

    def test_load_reduction_config_tsne_perplexities(self):
        configs = load_reduction_config({
            "tsne": {"perplexities": [10, 25]},
        })
        assert configs["tsne"].perplexities == [10, 25]

    def test_load_reduction_config_viz_2d_always_2d(self):
        configs = load_reduction_config({
            "viz_2d": {"n_components": 2, "metric": "euclidean"},
        })
        assert configs["viz_2d"].n_components == 2
        assert configs["viz_2d"].metric == "euclidean"

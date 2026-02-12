"""Tests for HDBSCAN cluster discovery and validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.clustering.cluster_discovery import (
    AblationResult,
    ClusterConfig,
    ClusterResult,
    PatternClusterDiscovery,
    QualityMetrics,
    StabilityResult,
    load_cluster_config,
    prepare_ablation_features,
    run_ablation,
)
from src.clustering.dimensionality_reduction import UMAPConfig


class TestHDBSCAN:
    """Tests for HDBSCAN clustering."""

    def test_finds_clusters_on_blobs(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters >= 2  # make_blobs with 4 centers

    def test_noise_label_is_minus_one(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        # Noise points (if any) should be labeled -1
        if result.noise_count > 0:
            assert -1 in result.labels

    def test_labels_shape(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        assert result.labels.shape == (200,)

    def test_probabilities_shape_and_range(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        assert result.probabilities.shape == (200,)
        assert result.probabilities.min() >= 0.0
        assert result.probabilities.max() <= 1.0

    def test_outlier_scores_shape(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        assert result.outlier_scores.shape == (200,)

    def test_noise_fraction_in_01(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        assert 0.0 <= result.noise_fraction <= 1.0

    def test_cluster_count_matches_labels(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        unique_non_noise = set(result.labels[result.labels >= 0])
        assert result.n_clusters == len(unique_non_noise)


class TestQualityMetrics:
    """Tests for cluster quality metrics."""

    def test_metrics_computed(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        metrics = discoverer.compute_quality_metrics(
            clusterable_features, result.labels
        )
        assert isinstance(metrics, QualityMetrics)

    def test_silhouette_range(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        metrics = discoverer.compute_quality_metrics(
            clusterable_features, result.labels
        )
        if not np.isnan(metrics.silhouette):
            assert -1.0 <= metrics.silhouette <= 1.0

    def test_davies_bouldin_positive(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        metrics = discoverer.compute_quality_metrics(
            clusterable_features, result.labels
        )
        if not np.isnan(metrics.davies_bouldin):
            assert metrics.davies_bouldin >= 0

    def test_calinski_harabasz_positive(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        metrics = discoverer.compute_quality_metrics(
            clusterable_features, result.labels
        )
        if not np.isnan(metrics.calinski_harabasz):
            assert metrics.calinski_harabasz > 0

    def test_per_cluster_silhouette(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.fit(clusterable_features)
        metrics = discoverer.compute_quality_metrics(
            clusterable_features, result.labels
        )
        for cid, sil in metrics.per_cluster_silhouette.items():
            assert -1.0 <= sil <= 1.0

    def test_metrics_with_insufficient_clusters(self):
        # All same label — only 1 cluster
        features = np.random.randn(50, 10).astype(np.float32)
        labels = np.zeros(50, dtype=int)
        discoverer = PatternClusterDiscovery()
        metrics = discoverer.compute_quality_metrics(features, labels)
        assert np.isnan(metrics.silhouette)

    def test_metrics_all_noise(self):
        features = np.random.randn(50, 10).astype(np.float32)
        labels = np.full(50, -1, dtype=int)
        discoverer = PatternClusterDiscovery()
        metrics = discoverer.compute_quality_metrics(features, labels)
        assert np.isnan(metrics.silhouette)


class TestBootstrapStability:
    """Tests for bootstrap stability analysis."""

    def test_stability_returns_result(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.evaluate_stability(
            clusterable_features, n_bootstrap=5
        )
        assert isinstance(result, StabilityResult)
        assert result.n_bootstrap == 5

    def test_ari_range(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.evaluate_stability(
            clusterable_features, n_bootstrap=5
        )
        if not np.isnan(result.mean_ari):
            assert -1.0 <= result.mean_ari <= 1.0

    def test_recovery_rates(self, clusterable_features: np.ndarray):
        discoverer = PatternClusterDiscovery(
            ClusterConfig(min_cluster_size=10, min_samples=3)
        )
        result = discoverer.evaluate_stability(
            clusterable_features, n_bootstrap=5
        )
        for cid, rate in result.per_cluster_recovery.items():
            assert 0.0 <= rate <= 1.0


class TestConfigLoading:
    """Tests for cluster config loading."""

    def test_default_config(self):
        config = load_cluster_config({})
        assert config.method == "hdbscan"
        assert config.min_cluster_size == 15
        assert config.min_samples == 5
        assert config.cluster_selection_method == "eom"

    def test_custom_config(self):
        config = load_cluster_config({
            "method": "hdbscan",
            "min_cluster_size": 20,
            "min_samples": 10,
            "cluster_selection_method": "leaf",
        })
        assert config.min_cluster_size == 20
        assert config.min_samples == 10
        assert config.cluster_selection_method == "leaf"


class TestAblation:
    """Tests for ablation feature loading and run."""

    def test_prepare_ablation_dino_only(self, tmp_path):
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "test.h5")
        dino = np.random.randn(50, 768).astype(np.float32)
        texture = np.random.randn(50, 90).astype(np.float32)
        store.save_image_ids([f"img_{i}" for i in range(50)])
        store.save_dino(dino)
        store.save_texture(texture)

        result = prepare_ablation_features("dino_only", store)
        assert result.shape == (50, 768)

    def test_prepare_ablation_texture_only(self, tmp_path):
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "test.h5")
        dino = np.random.randn(50, 768).astype(np.float32)
        texture = np.random.randn(50, 90).astype(np.float32)
        store.save_image_ids([f"img_{i}" for i in range(50)])
        store.save_dino(dino)
        store.save_texture(texture)

        result = prepare_ablation_features("texture_only", store)
        assert result.shape == (50, 90)

    def test_prepare_ablation_unknown_raises(self, tmp_path):
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / "test.h5")
        with pytest.raises(ValueError, match="Unknown ablation"):
            prepare_ablation_features("bogus", store)

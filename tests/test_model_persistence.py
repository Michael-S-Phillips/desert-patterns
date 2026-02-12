"""Tests for model persistence: FittedPipeline and save/load roundtrip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.clustering.cluster_discovery import (
    AblationResult,
    FittedPipeline,
    run_ablation,
)
from src.clustering.dimensionality_reduction import UMAPConfig


@pytest.fixture
def reduction_configs() -> dict[str, UMAPConfig]:
    """Minimal UMAP configs for fast testing."""
    return {
        "high_dim": UMAPConfig(n_components=5, n_neighbors=15, random_state=42),
        "viz_2d": UMAPConfig(n_components=2, n_neighbors=15, random_state=42),
        "viz_3d": UMAPConfig(n_components=3, n_neighbors=15, random_state=42),
    }


class TestFittedPipeline:
    """Tests for FittedPipeline and run_ablation return_fitted param."""

    def test_run_ablation_returns_fitted(
        self,
        clusterable_features: np.ndarray,
        reduction_configs: dict[str, UMAPConfig],
    ):
        """When return_fitted=True, AblationResult has a FittedPipeline."""
        from src.clustering.cluster_discovery import ClusterConfig

        result = run_ablation(
            "test_ablation",
            clusterable_features,
            ClusterConfig(min_cluster_size=10, min_samples=3),
            reduction_configs,
            return_fitted=True,
        )

        assert result.fitted_pipeline is not None
        assert isinstance(result.fitted_pipeline, FittedPipeline)
        assert result.fitted_pipeline.umap_high_dim._fitted_model is not None
        assert result.fitted_pipeline.umap_2d._fitted_model is not None
        assert result.fitted_pipeline.umap_3d._fitted_model is not None
        assert result.fitted_pipeline.hdbscan_discoverer._clusterer is not None

    def test_run_ablation_without_fitted(
        self,
        clusterable_features: np.ndarray,
        reduction_configs: dict[str, UMAPConfig],
    ):
        """Default run_ablation does not attach a FittedPipeline."""
        from src.clustering.cluster_discovery import ClusterConfig

        result = run_ablation(
            "test_ablation",
            clusterable_features,
            ClusterConfig(min_cluster_size=10, min_samples=3),
            reduction_configs,
        )

        assert result.fitted_pipeline is None

    def test_save_and_load_models_roundtrip(
        self,
        clusterable_features: np.ndarray,
        reduction_configs: dict[str, UMAPConfig],
        tmp_path,
    ):
        """Save models via joblib, then load with PatternPredictor and predict."""
        import joblib

        from src.clustering.cluster_discovery import ClusterConfig

        result = run_ablation(
            "test_ablation",
            clusterable_features,
            ClusterConfig(min_cluster_size=10, min_samples=3),
            reduction_configs,
            return_fitted=True,
        )
        pipeline = result.fitted_pipeline
        assert pipeline is not None

        # Save models
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        joblib.dump(pipeline.umap_high_dim._fitted_model, model_dir / "umap_high_dim.joblib")
        joblib.dump(pipeline.umap_2d._fitted_model, model_dir / "umap_2d.joblib")
        joblib.dump(pipeline.umap_3d._fitted_model, model_dir / "umap_3d.joblib")
        joblib.dump(pipeline.hdbscan_discoverer._clusterer, model_dir / "hdbscan.joblib")

        config = {
            "ablation_name": "test_ablation",
            "feature_type": "test",
            "clustering": {},
            "dimensionality_reduction": {},
        }
        with open(model_dir / "pipeline_config.json", "w") as f:
            json.dump(config, f)

        # Load with PatternPredictor and predict
        from src.inference.predict import PatternPredictor

        predictor = PatternPredictor(model_dir)
        sample = clusterable_features[0]
        pred = predictor.predict_from_features(sample)

        assert pred.cluster_id >= -1
        assert 0.0 <= pred.cluster_probability <= 1.0
        assert pred.umap_2d.shape == (2,)
        assert pred.umap_3d.shape == (3,)
        assert pred.umap_high_dim.shape == (5,)

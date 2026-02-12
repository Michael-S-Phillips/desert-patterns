"""Tests for PatternPredictor — single-image inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.clustering.cluster_discovery import ClusterConfig, run_ablation
from src.clustering.dimensionality_reduction import UMAPConfig
from src.inference.predict import PatternPredictor, PredictionResult


@pytest.fixture
def reduction_configs() -> dict[str, UMAPConfig]:
    """Minimal UMAP configs for fast testing."""
    return {
        "high_dim": UMAPConfig(n_components=5, n_neighbors=15, random_state=42),
        "viz_2d": UMAPConfig(n_components=2, n_neighbors=15, random_state=42),
        "viz_3d": UMAPConfig(n_components=3, n_neighbors=15, random_state=42),
    }


@pytest.fixture
def mock_model_dir(
    clusterable_features: np.ndarray,
    reduction_configs: dict[str, UMAPConfig],
    tmp_path,
) -> Path:
    """Create a model directory from fitted models."""
    result = run_ablation(
        "test",
        clusterable_features,
        ClusterConfig(min_cluster_size=10, min_samples=3),
        reduction_configs,
        return_fitted=True,
    )
    pipeline = result.fitted_pipeline
    assert pipeline is not None

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(pipeline.umap_high_dim._fitted_model, model_dir / "umap_high_dim.joblib")
    joblib.dump(pipeline.umap_2d._fitted_model, model_dir / "umap_2d.joblib")
    joblib.dump(pipeline.umap_3d._fitted_model, model_dir / "umap_3d.joblib")
    joblib.dump(pipeline.hdbscan_discoverer._clusterer, model_dir / "hdbscan.joblib")

    config = {
        "ablation_name": "test",
        "feature_type": "test",
        "clustering": {},
        "dimensionality_reduction": {},
    }
    with open(model_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f)

    return model_dir


class TestPatternPredictor:
    """Tests for single-image prediction."""

    def test_load_models(self, mock_model_dir: Path):
        """PatternPredictor loads without error."""
        predictor = PatternPredictor(mock_model_dir)
        assert predictor.config["ablation_name"] == "test"

    def test_predict_from_features(
        self,
        mock_model_dir: Path,
        clusterable_features: np.ndarray,
    ):
        """predict_from_features returns a valid PredictionResult."""
        predictor = PatternPredictor(mock_model_dir)
        sample = clusterable_features[0]
        result = predictor.predict_from_features(sample)

        assert isinstance(result, PredictionResult)
        assert result.cluster_id >= -1
        assert 0.0 <= result.cluster_probability <= 1.0
        assert result.umap_2d.shape == (2,)
        assert result.umap_3d.shape == (3,)
        assert result.umap_high_dim.shape == (5,)
        assert result.features_raw.shape == (50,)

    def test_find_nearest_neighbors(
        self,
        mock_model_dir: Path,
        clusterable_features: np.ndarray,
    ):
        """find_nearest_neighbors returns the correct number of IDs."""
        predictor = PatternPredictor(mock_model_dir)
        sample = clusterable_features[0]
        result = predictor.predict_from_features(sample)

        # Build a mock assignments DataFrame
        from src.clustering.dimensionality_reduction import DimensionalityReducer, UMAPConfig

        reducer = DimensionalityReducer()
        umap_result = reducer.fit_transform(
            clusterable_features, UMAPConfig(n_components=2, n_neighbors=15, random_state=42)
        )
        assignments_df = pd.DataFrame({
            "image_id": [f"img_{i:04d}" for i in range(len(clusterable_features))],
            "umap_x": umap_result.embeddings[:, 0],
            "umap_y": umap_result.embeddings[:, 1],
        })

        neighbors = predictor.find_nearest_neighbors(result, assignments_df, n=5)
        assert len(neighbors) == 5
        assert all(isinstance(n, str) for n in neighbors)

    def test_missing_model_dir_raises(self, tmp_path):
        """FileNotFoundError on non-existent model directory."""
        with pytest.raises(FileNotFoundError):
            PatternPredictor(tmp_path / "nonexistent")

    def test_missing_config_raises(self, tmp_path):
        """FileNotFoundError when pipeline_config.json is missing."""
        model_dir = tmp_path / "empty_models"
        model_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            PatternPredictor(model_dir)

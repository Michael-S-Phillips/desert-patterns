"""Tests for BatchPredictor — multi-image inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.clustering.cluster_discovery import ClusterConfig, run_ablation
from src.clustering.dimensionality_reduction import UMAPConfig
from src.inference.batch_predict import BatchPredictor, BatchResult
from src.inference.predict import PatternPredictor


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


class TestBatchPredictor:
    """Tests for batch prediction."""

    def test_predict_directory_empty(self, mock_model_dir: Path, tmp_path):
        """Empty directory yields an empty BatchResult."""
        predictor = PatternPredictor(mock_model_dir)
        batch = BatchPredictor(predictor)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = batch.predict_directory(empty_dir)

        assert isinstance(result, BatchResult)
        assert len(result.predictions) == 0
        assert len(result.failed_paths) == 0
        assert result.assignments_df.empty

    def test_batch_result_df_schema(self, mock_model_dir: Path, tmp_path):
        """Output DataFrame has the expected columns even when empty."""
        predictor = PatternPredictor(mock_model_dir)
        batch = BatchPredictor(predictor)

        empty_dir = tmp_path / "empty2"
        empty_dir.mkdir()
        result = batch.predict_directory(empty_dir)

        expected_cols = {
            "image_path", "image_name", "cluster_id",
            "cluster_probability", "umap_x", "umap_y",
        }
        assert expected_cols == set(result.assignments_df.columns)

    def test_predict_paths_empty(self, mock_model_dir: Path):
        """Empty path list yields an empty BatchResult."""
        predictor = PatternPredictor(mock_model_dir)
        batch = BatchPredictor(predictor)

        result = batch.predict_paths([])

        assert isinstance(result, BatchResult)
        assert len(result.predictions) == 0
        assert result.assignments_df.empty

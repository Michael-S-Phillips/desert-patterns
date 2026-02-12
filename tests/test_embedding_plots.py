"""Tests for embedding plot generation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.style import FigureStyle, setup_matplotlib_style
from src.visualization.embedding_plots import (
    create_interactive_umap,
    plot_ablation_comparison,
    plot_silhouette,
    plot_umap_by_feature,
    plot_umap_by_metadata,
    plot_umap_clusters,
)


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPlotUmapClusters:
    """Tests for the main UMAP cluster scatter."""

    def test_returns_figure(self, mock_assignments_df, tmp_path):
        fig = plot_umap_clusters(mock_assignments_df, tmp_path / "umap")
        assert isinstance(fig, plt.Figure)

    def test_creates_output_files(self, mock_assignments_df, tmp_path):
        style = FigureStyle(export_formats=["png"])
        plot_umap_clusters(mock_assignments_df, tmp_path / "umap", style)
        assert (tmp_path / "umap.png").exists()

    def test_all_noise(self, tmp_path):
        df = pd.DataFrame({
            "image_id": [f"p_{i}" for i in range(20)],
            "cluster_id": [-1] * 20,
            "umap_x": np.random.randn(20),
            "umap_y": np.random.randn(20),
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_umap_clusters(df, tmp_path / "noise_only", style)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "noise_only.png").exists()


class TestPlotUmapByMetadata:
    """Tests for metadata-coloured UMAP."""

    def test_creates_output(self, mock_assignments_df, tmp_path):
        meta = pd.DataFrame({
            "image_id": mock_assignments_df["image_id"],
            "source_type": np.random.choice(["drone", "ground"], len(mock_assignments_df)),
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_umap_by_metadata(
            mock_assignments_df, meta, "source_type",
            tmp_path / "by_source", style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "by_source.png").exists()


class TestPlotUmapByFeature:
    """Tests for feature-coloured UMAP."""

    def test_creates_output(self, mock_assignments_df, tmp_path):
        values = np.random.randn(len(mock_assignments_df))
        style = FigureStyle(export_formats=["png"])
        fig = plot_umap_by_feature(
            mock_assignments_df, values, "test_feature",
            tmp_path / "by_feat", style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "by_feat.png").exists()


class TestPlotAblationComparison:
    """Tests for multi-ablation comparison panels."""

    def test_creates_panels(self, mock_assignments_df, tmp_path):
        ablations = {
            "dino_only": mock_assignments_df.copy(),
            "texture_only": mock_assignments_df.copy(),
        }
        style = FigureStyle(export_formats=["png"])
        fig = plot_ablation_comparison(ablations, tmp_path / "ablation", style)
        assert isinstance(fig, plt.Figure)
        # Should have 2 subplots
        assert len(fig.axes) == 2
        assert (tmp_path / "ablation.png").exists()


class TestPlotSilhouette:
    """Tests for silhouette analysis plot."""

    def test_creates_output(self, tmp_path):
        rng = np.random.RandomState(42)
        features = rng.randn(100, 10).astype(np.float32)
        labels = np.repeat([0, 1, 2, 3], 25)
        style = FigureStyle(export_formats=["png"])
        fig = plot_silhouette(features, labels, tmp_path / "sil", style)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "sil.png").exists()

    def test_all_noise_handled(self, tmp_path):
        features = np.random.randn(20, 5).astype(np.float32)
        labels = np.full(20, -1)
        style = FigureStyle(export_formats=["png"])
        fig = plot_silhouette(features, labels, tmp_path / "sil_noise", style)
        assert isinstance(fig, plt.Figure)


class TestInteractiveUmap:
    """Tests for the plotly interactive UMAP."""

    def test_returns_plotly_figure(self, mock_assignments_df):
        fig = create_interactive_umap(mock_assignments_df)
        assert isinstance(fig, go.Figure)

    def test_with_metadata(self, mock_assignments_df):
        meta = pd.DataFrame({
            "image_id": mock_assignments_df["image_id"],
            "source_type": np.random.choice(["drone", "ground"], len(mock_assignments_df)),
        })
        fig = create_interactive_umap(mock_assignments_df, meta, "source_type")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

"""Tests for cluster gallery generation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.style import FigureStyle
from src.visualization.cluster_galleries import (
    _load_and_resize,
    plot_cluster_comparison_strip,
    plot_cluster_gallery,
    plot_noise_gallery,
)


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestLoadAndResize:
    """Tests for the thumbnail loader."""

    def test_loads_existing_image(self, mock_patch_paths):
        path = list(mock_patch_paths.values())[0]
        img = _load_and_resize(path, 64)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_missing_file_returns_placeholder(self):
        img = _load_and_resize(Path("/nonexistent/file.png"), 64)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert np.all(img == 180)  # gray placeholder

    def test_custom_size(self, mock_patch_paths):
        path = list(mock_patch_paths.values())[0]
        img = _load_and_resize(path, 256)
        assert img.shape == (256, 256, 3)


class TestPlotClusterGallery:
    """Tests for single cluster gallery."""

    def test_creates_output(self, mock_profiles, mock_patch_paths, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_cluster_gallery(
            mock_profiles[0], mock_patch_paths,
            tmp_path / "gallery_0", style=style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "gallery_0.png").exists()

    def test_empty_representatives(self, mock_patch_paths, tmp_path):
        profile = {"cluster_id": 99, "size": 0, "representative_ids": []}
        style = FigureStyle(export_formats=["png"])
        fig = plot_cluster_gallery(profile, mock_patch_paths, tmp_path / "empty", style=style)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "empty.png").exists()

    def test_missing_images_get_placeholder(self, tmp_path):
        profile = {
            "cluster_id": 0,
            "size": 3,
            "representative_ids": ["missing_a", "missing_b", "missing_c"],
        }
        style = FigureStyle(export_formats=["png"])
        fig = plot_cluster_gallery(profile, {}, tmp_path / "missing", style=style)
        assert isinstance(fig, plt.Figure)


class TestPlotClusterComparisonStrip:
    """Tests for multi-cluster comparison strip."""

    def test_creates_output(self, mock_profiles, mock_patch_paths, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_cluster_comparison_strip(
            mock_profiles, mock_patch_paths,
            tmp_path / "strip", style=style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "strip.png").exists()
        # Should have rows × cols axes
        assert len(fig.axes) > 0

    def test_no_clusters(self, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_cluster_comparison_strip([], {}, tmp_path / "empty_strip", style=style)
        assert isinstance(fig, plt.Figure)


class TestPlotNoiseGallery:
    """Tests for noise image gallery."""

    def test_creates_output(self, mock_assignments_df, mock_patch_paths, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_noise_gallery(
            mock_assignments_df, mock_patch_paths,
            tmp_path / "noise", style=style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "noise.png").exists()

    def test_no_noise_points(self, tmp_path):
        df = pd.DataFrame({
            "image_id": ["a", "b"],
            "cluster_id": [0, 1],
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_noise_gallery(df, {}, tmp_path / "no_noise", style=style)
        assert isinstance(fig, plt.Figure)

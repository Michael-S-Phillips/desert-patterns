"""Tests for temporal dynamics plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.style import FigureStyle
from src.visualization.temporal_plots import (
    plot_density_contours,
    plot_embedding_drift,
    plot_phase_distribution,
)


@pytest.fixture(autouse=True)
def _mpl_cleanup():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestPlotPhaseDistribution:
    """Tests for phase distribution stacked bar chart."""

    def test_creates_output(self, mock_temporal_data, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_phase_distribution(mock_temporal_data, tmp_path / "phase", style)
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "phase.png").exists()

    def test_empty_distribution(self, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_phase_distribution(
            {"phase_distributions": {}}, tmp_path / "empty_phase", style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "empty_phase.png").exists()

    def test_single_phase(self, tmp_path):
        data = {"phase_distributions": {"post_rain": {"0": 0.5, "1": 0.5}}}
        style = FigureStyle(export_formats=["png"])
        fig = plot_phase_distribution(data, tmp_path / "single", style)
        assert isinstance(fig, plt.Figure)


class TestPlotDensityContours:
    """Tests for density contour plots."""

    def test_creates_output(self, mock_continuous_space_df, tmp_path):
        meta = pd.DataFrame({
            "image_id": mock_continuous_space_df["image_id"],
            "temporal_phase": np.random.choice(
                ["during_rain", "post_rain"], len(mock_continuous_space_df),
            ),
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_density_contours(
            mock_continuous_space_df, meta, tmp_path / "density", style=style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "density.png").exists()

    def test_missing_phases(self, tmp_path):
        cs_df = pd.DataFrame({
            "image_id": ["a", "b"],
            "pattern_x": [0.0, 1.0],
            "pattern_y": [0.0, 1.0],
        })
        meta = pd.DataFrame({
            "image_id": ["a", "b"],
            "temporal_phase": [None, None],
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_density_contours(cs_df, meta, tmp_path / "no_phase", style=style)
        assert isinstance(fig, plt.Figure)

    def test_few_points_fallback(self, tmp_path):
        """Fewer than 10 points should use scatter instead of KDE."""
        cs_df = pd.DataFrame({
            "image_id": [f"p_{i}" for i in range(5)],
            "pattern_x": np.random.randn(5),
            "pattern_y": np.random.randn(5),
        })
        meta = pd.DataFrame({
            "image_id": [f"p_{i}" for i in range(5)],
            "temporal_phase": ["post_rain"] * 5,
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_density_contours(cs_df, meta, tmp_path / "few", style=style)
        assert isinstance(fig, plt.Figure)

    def test_explicit_phase_order(self, mock_continuous_space_df, tmp_path):
        meta = pd.DataFrame({
            "image_id": mock_continuous_space_df["image_id"],
            "temporal_phase": np.random.choice(
                ["during_rain", "post_rain"], len(mock_continuous_space_df),
            ),
        })
        style = FigureStyle(export_formats=["png"])
        fig = plot_density_contours(
            mock_continuous_space_df, meta, tmp_path / "ordered",
            phases=["during_rain", "post_rain"], style=style,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotEmbeddingDrift:
    """Tests for embedding drift arrow plot."""

    def test_creates_output(self, mock_temporal_data, mock_assignments_df, tmp_path):
        style = FigureStyle(export_formats=["png"])
        fig = plot_embedding_drift(
            mock_temporal_data, mock_assignments_df, tmp_path / "drift", style,
        )
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "drift.png").exists()

    def test_no_drift_returns_none(self, mock_assignments_df, tmp_path):
        data = {"embedding_drift": None}
        style = FigureStyle(export_formats=["png"])
        result = plot_embedding_drift(data, mock_assignments_df, tmp_path / "no_drift", style)
        assert result is None

    def test_empty_drift_returns_none(self, mock_assignments_df, tmp_path):
        data = {"embedding_drift": {}}
        style = FigureStyle(export_formats=["png"])
        result = plot_embedding_drift(data, mock_assignments_df, tmp_path / "empty_drift", style)
        assert result is None

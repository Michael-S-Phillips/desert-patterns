"""Tests for visualization style module."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.visualization.style import (
    NOISE_COLOR,
    TOL_QUALITATIVE,
    WONG_PALETTE,
    FigureStyle,
    get_cluster_colors,
    get_color_for_label,
    load_style_config,
    save_figure,
    setup_matplotlib_style,
)


class TestWongPalette:
    """Tests for the Wong colourblind-safe palette."""

    def test_palette_has_8_colours(self):
        assert len(WONG_PALETTE) == 8

    def test_palette_entries_are_hex(self):
        for colour in WONG_PALETTE:
            assert colour.startswith("#")
            assert len(colour) == 7

    def test_noise_color_is_gray(self):
        assert NOISE_COLOR == "#999999"


class TestFigureStyle:
    """Tests for FigureStyle configuration."""

    def test_defaults(self):
        style = FigureStyle()
        assert style.dpi == 300
        assert style.font_family == "Arial"
        assert style.font_size_base == 10
        assert style.figure_width == 7.0
        assert "png" in style.export_formats
        assert "svg" in style.export_formats
        assert style.continuous_cmap == "viridis"

    def test_load_from_dict(self):
        cfg = {"dpi": 150, "font_size_base": 12, "continuous_cmap": "cividis"}
        style = load_style_config(cfg)
        assert style.dpi == 150
        assert style.font_size_base == 12
        assert style.continuous_cmap == "cividis"
        assert style.font_family == "Arial"  # unchanged default

    def test_figure_width_double_alias(self):
        cfg = {"figure_width_double": 8.5}
        style = load_style_config(cfg)
        assert style.figure_width == 8.5


class TestTolPalette:
    """Tests for the Tol qualitative palette."""

    def test_palette_has_12_colours(self):
        assert len(TOL_QUALITATIVE) == 12

    def test_palette_entries_are_hex(self):
        for colour in TOL_QUALITATIVE:
            assert colour.startswith("#")
            assert len(colour) == 7

    def test_palette_all_distinct(self):
        assert len(set(TOL_QUALITATIVE)) == 12


class TestGetClusterColors:
    """Tests for colour generation."""

    def test_returns_n_colours(self):
        assert len(get_cluster_colors(3)) == 3

    def test_wong_for_8_or_fewer(self):
        colors = get_cluster_colors(8)
        assert colors == WONG_PALETTE[:8]

    def test_tol_for_9_to_12(self):
        colors = get_cluster_colors(10)
        assert len(colors) == 10
        assert colors == TOL_QUALITATIVE[:10]
        # All distinct
        assert len(set(colors)) == 10

    def test_tol_at_12(self):
        colors = get_cluster_colors(12)
        assert len(colors) == 12
        assert len(set(colors)) == 12
        assert colors == TOL_QUALITATIVE[:12]

    def test_tab20_fallback_above_12(self):
        colors = get_cluster_colors(15)
        assert len(colors) == 15
        # All should be hex strings
        for c in colors:
            assert c.startswith("#")
        # All distinct
        assert len(set(colors)) == 15

    def test_zero_colours(self):
        assert get_cluster_colors(0) == []


class TestGetColorForLabel:
    """Tests for label-to-colour mapping."""

    def test_noise_returns_gray(self):
        colors = get_cluster_colors(4)
        assert get_color_for_label(-1, colors) == NOISE_COLOR

    def test_valid_label(self):
        colors = get_cluster_colors(4)
        assert get_color_for_label(0, colors) == colors[0]
        assert get_color_for_label(2, colors) == colors[2]


class TestSetupMatplotlibStyle:
    """Tests for matplotlib style configuration."""

    def test_sets_rcparams(self):
        setup_matplotlib_style(FigureStyle(dpi=150, font_size_base=12))
        assert plt.rcParams["figure.dpi"] == 150
        assert plt.rcParams["font.size"] == 12

    def test_default_style(self):
        setup_matplotlib_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False


class TestSaveFigure:
    """Tests for multi-format figure saving."""

    def test_saves_png_and_svg(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "test_fig"
        saved = save_figure(fig, out, formats=["png", "svg"], dpi=72)
        assert len(saved) == 2
        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.svg").exists()

    def test_creates_directories(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "sub" / "dir" / "fig"
        saved = save_figure(fig, out, formats=["png"], dpi=72)
        assert len(saved) == 1
        assert saved[0].exists()

    def test_closes_figure(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig_num = fig.number
        save_figure(fig, tmp_path / "test", formats=["png"], dpi=72)
        assert fig_num not in plt.get_fignums()

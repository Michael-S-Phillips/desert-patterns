"""Shared style configuration for publication-quality figures.

Provides colourblind-safe palettes (Wong 2011), matplotlib rcParams setup,
figure saving in multiple formats, and consistent style across all plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Wong (2011) colourblind-safe palette — 8 distinct colours
WONG_PALETTE: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

NOISE_COLOR: str = "#999999"

# Tol qualitative palette — colourblind-safe, up to 12 distinct colours
# From Paul Tol's colour schemes (https://personal.sron.nl/~pault/)
TOL_QUALITATIVE: list[str] = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#000000",  # black
    "#EE7733",  # orange
    "#0077BB",  # dark blue
    "#33BBEE",  # light blue
    "#EE3377",  # magenta
]


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------


@dataclass
class FigureStyle:
    """Configuration for figure aesthetics."""

    dpi: int = 300
    font_family: str = "Arial"
    font_size_base: int = 10
    figure_width: float = 7.0
    export_formats: list[str] = field(default_factory=lambda: ["png", "svg"])
    continuous_cmap: str = "viridis"


def load_style_config(config_dict: dict) -> FigureStyle:
    """Create a FigureStyle from a configuration dictionary.

    Args:
        config_dict: Dictionary with optional keys matching FigureStyle fields.

    Returns:
        Configured FigureStyle instance.
    """
    kwargs = {}
    for fld in ("dpi", "font_family", "font_size_base", "figure_width", "continuous_cmap"):
        if fld in config_dict:
            kwargs[fld] = config_dict[fld]
    # Handle alternate name from config
    if "figure_width_double" in config_dict and "figure_width" not in config_dict:
        kwargs["figure_width"] = config_dict["figure_width_double"]
    if "export_formats" in config_dict:
        kwargs["export_formats"] = list(config_dict["export_formats"])
    return FigureStyle(**kwargs)


def setup_matplotlib_style(style: FigureStyle | None = None) -> None:
    """Configure matplotlib rcParams for publication-quality figures.

    Sets Agg backend when no display is available (headless / CI).
    Falls back to DejaVu Sans if the requested font is not available.

    Args:
        style: Optional style configuration. Uses defaults if None.
    """
    if style is None:
        style = FigureStyle()

    # Use non-interactive backend when no display is available
    try:
        display = matplotlib.get_backend()
        if display in ("agg", "Agg"):
            pass  # already non-interactive
        else:
            # Try to detect headless environment
            import os
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                matplotlib.use("Agg")
    except Exception:
        matplotlib.use("Agg")

    # Font setup with fallback
    font_family = style.font_family
    try:
        from matplotlib.font_manager import findfont, FontProperties
        fp = FontProperties(family=font_family)
        found = findfont(fp, fallback_to_default=False)
        if "dejavu" in found.lower() and font_family.lower() != "dejavu sans":
            logger.warning(
                "Font '%s' not found, falling back to DejaVu Sans", font_family
            )
            font_family = "DejaVu Sans"
    except Exception:
        logger.warning("Font '%s' not found, falling back to DejaVu Sans", font_family)
        font_family = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "DejaVu Sans"],
        "font.size": style.font_size_base,
        "axes.titlesize": style.font_size_base + 2,
        "axes.labelsize": style.font_size_base,
        "xtick.labelsize": style.font_size_base - 1,
        "ytick.labelsize": style.font_size_base - 1,
        "legend.fontsize": style.font_size_base - 1,
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
        "figure.figsize": (style.figure_width, style.figure_width * 0.75),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_cluster_colors(n: int) -> list[str]:
    """Return *n* colourblind-safe colours.

    Palette selection:
    - n <= 8: Wong (2011) palette (8 colours)
    - 8 < n <= 12: Tol qualitative palette (12 colours)
    - n > 12: matplotlib tab20 (20 colours, less CVD-safe but necessary)

    Args:
        n: Number of colours needed.

    Returns:
        List of hex colour strings.
    """
    if n <= 8:
        return [WONG_PALETTE[i] for i in range(n)]
    if n <= 12:
        return [TOL_QUALITATIVE[i] for i in range(n)]
    # Fall back to tab20 for large cluster counts
    cmap = plt.cm.tab20
    return [matplotlib.colors.rgb2hex(cmap(i / n)) for i in range(n)]


def get_color_for_label(label: int, colors: list[str]) -> str:
    """Map a cluster label to a colour, using NOISE_COLOR for -1.

    Args:
        label: Cluster label (-1 for noise).
        colors: Colour list from get_cluster_colors.

    Returns:
        Hex colour string.
    """
    if label == -1:
        return NOISE_COLOR
    return colors[label % len(colors)]


def save_figure(
    fig: plt.Figure,
    path: Path,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Save figure in multiple formats, creating directories as needed.

    Args:
        fig: Matplotlib figure to save.
        path: Base path (extension ignored; each format appended).
        formats: List of format extensions (e.g. ["png", "svg"]).
        dpi: Resolution for raster formats.

    Returns:
        List of saved file paths.
    """
    if formats is None:
        formats = ["png", "svg"]

    base = path.parent / path.stem
    base.parent.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", format=fmt)
        saved.append(out)
        logger.debug("Saved figure: %s", out)

    plt.close(fig)
    return saved

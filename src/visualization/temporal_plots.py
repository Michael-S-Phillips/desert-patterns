"""Temporal dynamics visualizations.

Phase distribution stacked bars, density contour comparisons across
temporal phases, and embedding drift arrows on 2-D UMAP space.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    FigureStyle,
    get_cluster_colors,
    save_figure,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)


def plot_phase_distribution(
    temporal_data: dict,
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Stacked bar chart of cluster counts per temporal phase.

    Args:
        temporal_data: Dict loaded from ``{name}_temporal.json``, must
            contain ``phase_distributions`` as a nested dict convertible
            to a DataFrame (cluster × phase fractions).
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    dist = temporal_data.get("phase_distributions", {})
    if not dist:
        fig, ax = plt.subplots(figsize=(style.figure_width * 0.7, style.figure_width * 0.5))
        ax.text(0.5, 0.5, "No phase distribution data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    df = pd.DataFrame(dist)
    # Rows are phases, columns are cluster IDs
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty distribution", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    # Transpose so clusters are rows, phases are columns for stacked bar
    df = df.T
    cluster_ids = df.index.tolist()
    phases = df.columns.tolist()
    colors = get_cluster_colors(len(phases))

    fig, ax = plt.subplots(figsize=(style.figure_width * 0.8, style.figure_width * 0.6))

    bottoms = np.zeros(len(cluster_ids))
    x = np.arange(len(cluster_ids))

    for i, phase in enumerate(phases):
        vals = df[phase].values.astype(float)
        ax.bar(x, vals, bottom=bottoms, color=colors[i], label=phase, edgecolor="white",
               linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cluster_ids])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction")
    ax.set_title("Phase Distribution by Cluster")
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=style.font_size_base - 2, frameon=False,
    )

    fig.tight_layout()
    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_density_contours(
    continuous_space_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_path: Path,
    phases: list[str] | None = None,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Side-by-side KDE contour plots per temporal phase.

    Falls back to scatter plots when fewer than 10 points are available
    for a phase.

    Args:
        continuous_space_df: DataFrame with ``image_id``, ``pattern_x``,
            ``pattern_y``.
        metadata_df: DataFrame with ``image_id`` and ``temporal_phase``.
        output_path: Base path for saved figure files.
        phases: Ordered list of phase names to plot. Defaults to unique
            values in metadata_df.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    merged = continuous_space_df.merge(
        metadata_df[["image_id", "temporal_phase"]].drop_duplicates(),
        on="image_id",
        how="left",
    )

    if phases is None:
        phases = sorted(merged["temporal_phase"].dropna().unique())
    # Filter to phases with data
    phases = [p for p in phases if (merged["temporal_phase"] == p).sum() > 0]

    if not phases:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No temporal phases with data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    n = len(phases)
    fig, axes = plt.subplots(
        1, n,
        figsize=(style.figure_width, style.figure_width / max(n, 1) * 0.85),
        squeeze=False,
    )

    # Shared axis limits
    x_all = merged["pattern_x"].values
    y_all = merged["pattern_y"].values
    xlim = (np.nanmin(x_all) - 0.5, np.nanmax(x_all) + 0.5)
    ylim = (np.nanmin(y_all) - 0.5, np.nanmax(y_all) + 0.5)

    for i, phase in enumerate(phases):
        ax = axes[0, i]
        mask = merged["temporal_phase"] == phase
        px = merged.loc[mask, "pattern_x"].values
        py = merged.loc[mask, "pattern_y"].values
        n_pts = len(px)

        if n_pts >= 10:
            # KDE contour
            from scipy.stats import gaussian_kde
            try:
                xy = np.vstack([px, py])
                kde = gaussian_kde(xy)
                xg = np.linspace(xlim[0], xlim[1], 80)
                yg = np.linspace(ylim[0], ylim[1], 80)
                X, Y = np.meshgrid(xg, yg)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contourf(X, Y, Z, levels=8, cmap=style.continuous_cmap, alpha=0.7)
                ax.contour(X, Y, Z, levels=8, colors="white", linewidths=0.3)
            except np.linalg.LinAlgError:
                # Singular matrix — fall back to scatter
                ax.scatter(px, py, s=8, alpha=0.5, c="steelblue")
        else:
            # Scatter fallback
            ax.scatter(px, py, s=12, alpha=0.6, c="steelblue")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{phase}\n(n={n_pts})", fontsize=style.font_size_base)
        ax.set_xlabel("Pattern dim 1")
        if i == 0:
            ax.set_ylabel("Pattern dim 2")
        else:
            ax.set_yticklabels([])

    fig.suptitle("Density by Temporal Phase", fontsize=style.font_size_base + 2, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_embedding_drift(
    temporal_data: dict,
    assignments_df: pd.DataFrame,
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure | None:
    """Arrows on UMAP showing centroid drift between phases.

    Returns None if no drift data is available in temporal_data.

    Args:
        temporal_data: Dict loaded from ``{name}_temporal.json``, checked
            for ``embedding_drift`` key with per-cluster centroid data.
        assignments_df: DataFrame with ``umap_x``, ``umap_y``, ``cluster_id``.
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure, or None if no drift data.
    """
    drift = temporal_data.get("embedding_drift")
    if not drift:
        logger.info("No embedding drift data available — skipping drift plot")
        return None

    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))

    # Background scatter
    labels = assignments_df["cluster_id"].values
    ax.scatter(
        assignments_df["umap_x"],
        assignments_df["umap_y"],
        c="#dddddd", s=4, alpha=0.3, zorder=1,
    )

    cluster_ids = sorted(set(labels) - {-1})
    colors = get_cluster_colors(len(cluster_ids))

    # Draw drift arrows for each cluster
    for cid_str, drift_info in drift.items():
        cid = int(cid_str)
        if cid not in cluster_ids:
            continue
        color = colors[cluster_ids.index(cid)]

        centroids = drift_info if isinstance(drift_info, list) else drift_info.get("centroids", [])
        if len(centroids) < 2:
            continue

        for j in range(len(centroids) - 1):
            c0 = centroids[j]
            c1 = centroids[j + 1]
            if len(c0) >= 2 and len(c1) >= 2:
                ax.annotate(
                    "",
                    xy=(c1[0], c1[1]),
                    xytext=(c0[0], c0[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                    zorder=3,
                )

        # Mark start centroid
        ax.scatter(centroids[0][0], centroids[0][1], c=color, s=60,
                   edgecolors="black", linewidth=0.5, zorder=4, marker="o")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embedding Drift by Cluster")

    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig

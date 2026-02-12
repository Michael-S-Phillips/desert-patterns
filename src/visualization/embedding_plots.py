"""UMAP embedding scatter plots and silhouette analysis.

Produces publication-quality 2-D scatter plots coloured by cluster,
metadata fields, or continuous feature values. Also generates interactive
plotly figures for the Gradio explorer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.style import (
    NOISE_COLOR,
    FigureStyle,
    get_cluster_colors,
    get_color_for_label,
    save_figure,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)


def plot_umap_clusters(
    assignments_df: pd.DataFrame,
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Scatter plot of 2-D UMAP coloured by cluster label.

    Noise points (cluster_id == -1) are rendered in gray below other points.
    Legend shows cluster IDs with sample counts.

    Args:
        assignments_df: DataFrame with ``umap_x``, ``umap_y``, ``cluster_id``.
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    labels = assignments_df["cluster_id"].values
    unique_labels = sorted(set(labels))
    cluster_labels = [l for l in unique_labels if l != -1]
    colors = get_cluster_colors(len(cluster_labels))

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))

    # Plot noise first (behind)
    if -1 in unique_labels:
        mask = labels == -1
        count = mask.sum()
        ax.scatter(
            assignments_df.loc[mask, "umap_x"],
            assignments_df.loc[mask, "umap_y"],
            c=NOISE_COLOR,
            s=8,
            alpha=0.3,
            label=f"Noise (n={count})",
            zorder=1,
        )

    # Plot each cluster
    for i, cid in enumerate(cluster_labels):
        mask = labels == cid
        count = mask.sum()
        ax.scatter(
            assignments_df.loc[mask, "umap_x"],
            assignments_df.loc[mask, "umap_y"],
            c=colors[i],
            s=12,
            alpha=0.7,
            label=f"Cluster {cid} (n={count})",
            zorder=2,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Pattern Space — UMAP Clusters")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=style.font_size_base - 2,
        frameon=False,
    )

    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_umap_by_metadata(
    assignments_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_col: str,
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Scatter plot of 2-D UMAP coloured by a categorical metadata field.

    Args:
        assignments_df: DataFrame with ``umap_x``, ``umap_y``, ``image_id``.
        metadata_df: DataFrame with ``image_id`` and *color_col*.
        color_col: Column name to use for colouring.
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    # Merge metadata
    merged = assignments_df.merge(metadata_df[["image_id", color_col]], on="image_id", how="left")
    categories = sorted(merged[color_col].dropna().unique())
    colors = get_cluster_colors(len(categories))

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))

    for i, cat in enumerate(categories):
        mask = merged[color_col] == cat
        count = mask.sum()
        ax.scatter(
            merged.loc[mask, "umap_x"],
            merged.loc[mask, "umap_y"],
            c=colors[i],
            s=12,
            alpha=0.7,
            label=f"{cat} (n={count})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Pattern Space — {color_col}")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=style.font_size_base - 2,
        frameon=False,
    )

    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_umap_by_feature(
    assignments_df: pd.DataFrame,
    feature_values: np.ndarray,
    feature_name: str,
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Scatter plot of 2-D UMAP coloured by a continuous feature value.

    Args:
        assignments_df: DataFrame with ``umap_x``, ``umap_y``.
        feature_values: 1-D array of feature values (same length as df).
        feature_name: Name of the feature for labelling.
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))

    sc = ax.scatter(
        assignments_df["umap_x"],
        assignments_df["umap_y"],
        c=feature_values,
        cmap=style.continuous_cmap,
        s=12,
        alpha=0.7,
    )
    fig.colorbar(sc, ax=ax, label=feature_name, shrink=0.8)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Pattern Space — {feature_name}")

    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_ablation_comparison(
    ablation_assignments: dict[str, pd.DataFrame],
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Side-by-side UMAP scatter for multiple ablations.

    Creates a 1×N subplot layout with panel labels (a), (b), (c), etc.

    Args:
        ablation_assignments: Mapping from ablation name to assignments DataFrame.
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    n = len(ablation_assignments)
    panel_w = style.figure_width / max(n, 1)
    fig, axes = plt.subplots(
        1, n, figsize=(style.figure_width, panel_w * 0.85), squeeze=False
    )
    panel_labels = "abcdefghij"

    for idx, (name, adf) in enumerate(ablation_assignments.items()):
        ax = axes[0, idx]
        labels = adf["cluster_id"].values
        cluster_ids = sorted(set(labels) - {-1})
        colors = get_cluster_colors(len(cluster_ids))

        # Noise
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(
                adf.loc[noise_mask, "umap_x"],
                adf.loc[noise_mask, "umap_y"],
                c=NOISE_COLOR, s=4, alpha=0.3, zorder=1,
            )

        for i, cid in enumerate(cluster_ids):
            mask = labels == cid
            ax.scatter(
                adf.loc[mask, "umap_x"],
                adf.loc[mask, "umap_y"],
                c=colors[i], s=6, alpha=0.7, zorder=2,
            )

        ax.set_title(name, fontsize=style.font_size_base)
        ax.set_xlabel("UMAP 1")
        if idx == 0:
            ax.set_ylabel("UMAP 2")
        else:
            ax.set_yticklabels([])

        # Panel label
        if idx < len(panel_labels):
            ax.text(
                0.02, 0.98, f"({panel_labels[idx]})",
                transform=ax.transAxes,
                fontsize=style.font_size_base + 1,
                fontweight="bold",
                va="top",
            )

    fig.tight_layout()
    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Silhouette plot with bars grouped by cluster.

    Args:
        features: Feature matrix ``(N, D)`` used for silhouette computation.
        labels: Cluster labels ``(N,)`` from HDBSCAN (-1 = noise, excluded).
        output_path: Base path for saved figure files.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    from sklearn.metrics import silhouette_samples

    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    # Exclude noise for silhouette
    non_noise = labels != -1
    if non_noise.sum() < 2:
        logger.warning("Insufficient non-noise points for silhouette plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    feats = features[non_noise]
    labs = labels[non_noise]
    sil_vals = silhouette_samples(feats, labs)
    avg_sil = sil_vals.mean()

    cluster_ids = sorted(set(labs))
    colors = get_cluster_colors(len(cluster_ids))

    fig, ax = plt.subplots(figsize=(style.figure_width * 0.6, style.figure_width))
    y_lower = 0

    for i, cid in enumerate(cluster_ids):
        cluster_sil = sil_vals[labs == cid]
        cluster_sil.sort()
        y_upper = y_lower + len(cluster_sil)
        ax.barh(
            range(y_lower, y_upper),
            cluster_sil,
            height=1.0,
            color=colors[i],
            edgecolor="none",
            alpha=0.8,
        )
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(cid),
                fontsize=style.font_size_base - 1, va="center", ha="right")
        y_lower = y_upper + 2  # gap between clusters

    ax.axvline(avg_sil, color="red", linestyle="--", linewidth=1,
               label=f"Mean = {avg_sil:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Analysis")
    ax.set_yticks([])
    ax.legend(loc="lower right")

    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def create_interactive_umap(
    assignments_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    color_col: str = "cluster_id",
) -> go.Figure:
    """Create an interactive plotly scatter for the Gradio explorer.

    Uses ``scattergl`` for performance with large point counts.

    Args:
        assignments_df: DataFrame with ``umap_x``, ``umap_y``, ``cluster_id``,
            and ``image_id``.
        metadata_df: Optional metadata to merge for hover info.
        color_col: Column to colour by (default ``cluster_id``).

    Returns:
        Plotly Figure object.
    """
    df = assignments_df.copy()

    if metadata_df is not None and color_col != "cluster_id":
        if color_col in metadata_df.columns:
            df = df.merge(
                metadata_df[["image_id", color_col]].drop_duplicates(),
                on="image_id",
                how="left",
            )

    fig = go.Figure()

    if color_col == "cluster_id":
        labels = df["cluster_id"].values
        unique_labels = sorted(set(labels))
        cluster_labels = [l for l in unique_labels if l != -1]
        colors = get_cluster_colors(len(cluster_labels))

        # Noise
        if -1 in unique_labels:
            mask = labels == -1
            fig.add_trace(go.Scattergl(
                x=df.loc[mask, "umap_x"],
                y=df.loc[mask, "umap_y"],
                mode="markers",
                marker=dict(color=NOISE_COLOR, size=4, opacity=0.3),
                name="Noise",
                text=df.loc[mask, "image_id"],
                hovertemplate="ID: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}",
            ))

        for i, cid in enumerate(cluster_labels):
            mask = labels == cid
            fig.add_trace(go.Scattergl(
                x=df.loc[mask, "umap_x"],
                y=df.loc[mask, "umap_y"],
                mode="markers",
                marker=dict(color=colors[i], size=5, opacity=0.7),
                name=f"Cluster {cid}",
                text=df.loc[mask, "image_id"],
                hovertemplate="ID: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}",
            ))
    else:
        categories = sorted(df[color_col].dropna().unique())
        colors = get_cluster_colors(len(categories))
        for i, cat in enumerate(categories):
            mask = df[color_col] == cat
            fig.add_trace(go.Scattergl(
                x=df.loc[mask, "umap_x"],
                y=df.loc[mask, "umap_y"],
                mode="markers",
                marker=dict(color=colors[i], size=5, opacity=0.7),
                name=str(cat),
                text=df.loc[mask, "image_id"],
                hovertemplate="ID: %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}",
            ))

    fig.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        legend=dict(orientation="v"),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    return fig

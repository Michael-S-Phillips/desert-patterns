"""Cluster image galleries for visual inspection.

Generates thumbnail grids for each cluster's representative images,
comparison strips across clusters, and noise image galleries.
Missing images are replaced with gray placeholders.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    FigureStyle,
    get_cluster_colors,
    save_figure,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)


def _load_and_resize(path: Path, size: int = 128) -> np.ndarray:
    """Load an image and resize to a square thumbnail.

    Returns a gray placeholder if the file does not exist or cannot be read.

    Args:
        path: Path to the image file.
        size: Target side length in pixels.

    Returns:
        RGB uint8 array of shape ``(size, size, 3)``.
    """
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            return img

    # Gray placeholder
    placeholder = np.full((size, size, 3), 180, dtype=np.uint8)
    return placeholder


def plot_cluster_gallery(
    profile_dict: dict,
    patch_paths: dict[str, Path],
    output_path: Path,
    n_cols: int = 5,
    thumbnail_size: int = 128,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Grid of representative images for a single cluster.

    Args:
        profile_dict: Cluster profile dict with ``cluster_id``,
            ``representative_ids``, ``size``, ``description``.
        patch_paths: Mapping from image_id to file path.
        output_path: Base path for saved figure files.
        n_cols: Number of columns in the grid.
        thumbnail_size: Side length of each thumbnail.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    rep_ids = profile_dict.get("representative_ids", [])
    cid = profile_dict.get("cluster_id", "?")
    size = profile_dict.get("size", 0)
    desc = profile_dict.get("description", "")

    if not rep_ids:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"Cluster {cid}: no representatives",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    n_rows = (len(rep_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.3, n_rows * 1.3 + 0.6),
        squeeze=False,
    )

    for i, img_id in enumerate(rep_ids):
        r, c = divmod(i, n_cols)
        path = patch_paths.get(img_id, Path("missing"))
        thumb = _load_and_resize(path, thumbnail_size)
        axes[r, c].imshow(thumb)
        axes[r, c].axis("off")

    # Turn off unused axes
    for i in range(len(rep_ids), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Cluster {cid} (n={size})", fontsize=style.font_size_base + 1)
    if desc:
        fig.text(0.5, 0.01, desc, ha="center", fontsize=style.font_size_base - 2,
                 style="italic", wrap=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_cluster_comparison_strip(
    profiles: list[dict],
    patch_paths: dict[str, Path],
    output_path: Path,
    n_per_cluster: int = 8,
    thumbnail_size: int = 128,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """One row per cluster of representative images.

    Args:
        profiles: List of cluster profile dicts.
        patch_paths: Mapping from image_id to file path.
        output_path: Base path for saved figure files.
        n_per_cluster: Max images per cluster row.
        thumbnail_size: Side length of each thumbnail.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    # Filter to non-noise clusters
    cluster_profiles = [p for p in profiles if p.get("cluster_id", -1) != -1]
    if not cluster_profiles:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No clusters to display", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    n_clusters = len(cluster_profiles)
    colors = get_cluster_colors(n_clusters)

    fig, axes = plt.subplots(
        n_clusters, n_per_cluster,
        figsize=(n_per_cluster * 1.2, n_clusters * 1.4),
        squeeze=False,
    )

    for row, profile in enumerate(cluster_profiles):
        cid = profile.get("cluster_id", row)
        rep_ids = profile.get("representative_ids", [])[:n_per_cluster]

        for col in range(n_per_cluster):
            ax = axes[row, col]
            if col < len(rep_ids):
                path = patch_paths.get(rep_ids[col], Path("missing"))
                thumb = _load_and_resize(path, thumbnail_size)
                ax.imshow(thumb)
            ax.axis("off")

        # Row label
        axes[row, 0].set_ylabel(
            f"C{cid}",
            rotation=0,
            labelpad=30,
            fontsize=style.font_size_base,
            color=colors[row],
            fontweight="bold",
        )
        axes[row, 0].yaxis.set_label_position("left")

    fig.suptitle("Cluster Comparison", fontsize=style.font_size_base + 2)
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig


def plot_noise_gallery(
    assignments_df,
    patch_paths: dict[str, Path],
    output_path: Path,
    max_images: int = 25,
    thumbnail_size: int = 128,
    n_cols: int = 5,
    style: FigureStyle | None = None,
) -> plt.Figure:
    """Gallery of noise-labelled images.

    Args:
        assignments_df: DataFrame with ``image_id`` and ``cluster_id``.
        patch_paths: Mapping from image_id to file path.
        output_path: Base path for saved figure files.
        max_images: Maximum number of noise images to show.
        thumbnail_size: Side length of each thumbnail.
        n_cols: Number of columns in the grid.
        style: Figure style configuration.

    Returns:
        The matplotlib Figure.
    """
    if style is None:
        style = FigureStyle()
    setup_matplotlib_style(style)

    noise_ids = assignments_df.loc[
        assignments_df["cluster_id"] == -1, "image_id"
    ].tolist()[:max_images]

    if not noise_ids:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No noise points", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        save_figure(fig, output_path, style.export_formats, style.dpi)
        return fig

    n_rows = (len(noise_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.3, n_rows * 1.3 + 0.4),
        squeeze=False,
    )

    for i, img_id in enumerate(noise_ids):
        r, c = divmod(i, n_cols)
        path = patch_paths.get(img_id, Path("missing"))
        thumb = _load_and_resize(path, thumbnail_size)
        axes[r, c].imshow(thumb)
        axes[r, c].axis("off")

    for i in range(len(noise_ids), n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Noise Points (n={len(noise_ids)})", fontsize=style.font_size_base + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_path, style.export_formats, style.dpi)
    return fig

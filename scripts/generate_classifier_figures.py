#!/usr/bin/env python3
"""Generate classifier analysis figures for DINOv3 embedding interpretability.

Produces figures in outputs/figures/classifier_analysis/:
  class_gallery_{name}.png/.svg          — top-12 highest-confidence images per class
  lr_coefficient_heatmap.png/.svg        — top-50 embedding dims x 3 classes
  lr_coefficient_bars.png/.svg           — top-30 coeff dims per class (bar chart)
  image_extremes.png/.svg                — images at high/low activation on top discriminative dims
  class_probability_histograms.png/.svg
  cosine_similarity_matrix.png/.svg
  pca_projection.png/.svg                — first 2 PCs colored by class
  pca_component_strips.png/.svg          — images at extremes of top 4 PCs
  pca_variance_explained.png/.svg        — scree plot
  patch_projection_overlays.png/.svg     — per-patch activation heatmaps on original images

Usage:
    python scripts/generate_classifier_figures.py
    python scripts/generate_classifier_figures.py --skip-patch-overlays  # skip slow inference step
    python scripts/generate_classifier_figures.py --verbose
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from src.classification.train import load_classifier_config, scan_labeled_images
from src.visualization.style import (
    WONG_PALETTE,
    FigureStyle,
    save_figure,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)

THUMB_LARGE = 224   # px for class galleries
THUMB_SMALL = 112   # px for image extremes / PCA strips


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(config) -> tuple[np.ndarray, list[str], list, object]:
    """Load embeddings, labels, image paths, and model.

    Returns:
        X: (N, 768) float32 embeddings
        y: list of N class label strings (authoritative from cache)
        image_list: list[tuple[Path, str]] aligned with X/y
        model: fitted LogisticRegression
    """
    X = np.load(config.embedding_cache)
    y = list(np.load(config.label_cache))

    image_list = scan_labeled_images(Path(config.image_dir), config.label_map)

    cached_labels = list(y)
    scanned_labels = [lbl for _, lbl in image_list]
    if scanned_labels != cached_labels:
        raise ValueError(
            f"Image path order does not match cached labels "
            f"({len(scanned_labels)} scanned vs {len(cached_labels)} cached). "
            "Re-run train_classifier.py --force to rebuild the cache."
        )

    model = joblib.load(Path(config.model_dir) / "lr_classifier.joblib")
    logger.info(
        "Loaded %d embeddings, %d labels, model classes: %s",
        len(X), len(y), list(model.classes_),
    )
    return X, y, image_list, model


def load_thumb(path: Path, size: int) -> np.ndarray:
    """Load an image as an RGB numpy array thumbnail."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Figure 1: per-class image galleries
# ---------------------------------------------------------------------------


def fig_class_galleries(
    X: np.ndarray,
    y: list[str],
    image_list: list,
    model,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    for cls in classes:
        col_idx = classes.index(cls)
        cls_mask = np.array([lbl == cls for lbl in y])
        cls_indices = np.where(cls_mask)[0]
        cls_probs = proba[cls_indices, col_idx]
        top_n = min(12, len(cls_indices))
        top_local = np.argsort(cls_probs)[::-1][:top_n]
        top_global = cls_indices[top_local]

        ncols, nrows = 4, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
        fig.suptitle(
            f"{cls} — top-{top_n} by confidence  (n={len(cls_indices)})",
            fontsize=style.font_size_base + 1,
        )
        for ax in axes.flat:
            ax.axis("off")

        for k, gi in enumerate(top_global):
            ax = axes[k // ncols, k % ncols]
            img_path, _ = image_list[gi]
            try:
                thumb = load_thumb(img_path, THUMB_LARGE)
                ax.imshow(thumb)
            except Exception as e:
                logger.warning("Could not load %s: %s", img_path, e)
            ax.set_title(f"{cls_probs[top_local[k]]:.3f}", fontsize=7)

        fig.tight_layout()
        out = out_dir / f"class_gallery_{cls}"
        save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
        logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 2: LR coefficient heatmap
# ---------------------------------------------------------------------------


def fig_lr_coefficient_heatmap(
    model,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    coef = model.coef_          # (3, 768)
    classes = list(model.classes_)
    top_n = 50
    max_abs = np.abs(coef).max(axis=0)
    top_dims = np.argsort(max_abs)[::-1][:top_n]
    coef_sub = coef[:, top_dims]

    vmax = np.abs(coef_sub).max()
    fig, ax = plt.subplots(figsize=(style.figure_width * 1.5, 2.5))
    im = ax.imshow(coef_sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=style.font_size_base)
    ax.set_xlabel(f"Embedding dimension (top-{top_n} by max |coef|)", fontsize=style.font_size_base)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_dims, rotation=90, fontsize=6)
    ax.set_title("LR coefficients — top-50 discriminative embedding dims", fontsize=style.font_size_base + 1)
    fig.colorbar(im, ax=ax, label="LR coefficient", shrink=0.8)
    fig.tight_layout()

    out = out_dir / "lr_coefficient_heatmap"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 3: per-class coefficient bar charts
# ---------------------------------------------------------------------------


def fig_lr_coefficient_bars(
    model,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    coef = model.coef_          # (3, 768)
    classes = list(model.classes_)
    top_n = 30

    fig, axes = plt.subplots(1, 3, figsize=(style.figure_width * 1.6, 3.5), sharey=False)
    wong_pos = WONG_PALETTE[0]   # orange
    wong_neg = WONG_PALETTE[4]   # blue

    for i, (cls, ax) in enumerate(zip(classes, axes)):
        row = coef[i]
        top_dims = np.argsort(np.abs(row))[::-1][:top_n]
        vals = row[top_dims]
        colors = [wong_pos if v > 0 else wong_neg for v in vals]
        ax.bar(range(top_n), vals, color=colors, width=0.8)
        ax.set_title(cls, fontsize=style.font_size_base)
        ax.set_xlabel("Rank", fontsize=style.font_size_base - 1)
        ax.set_ylabel("Coefficient" if i == 0 else "", fontsize=style.font_size_base - 1)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(top_dims, rotation=90, fontsize=5)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Top-30 LR coefficient dims per class", fontsize=style.font_size_base + 1)
    fig.tight_layout()

    out = out_dir / "lr_coefficient_bars"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 4: image extremes gallery
# ---------------------------------------------------------------------------


def _collect_discriminative_dims(model, n: int = 3) -> list[tuple[int, str]]:
    """Return up to n unique (dim_index, class_name) pairs.

    Iterates top-1, top-2, ... per class in round-robin until n unique dims
    accumulated.
    """
    coef = model.coef_
    classes = list(model.classes_)
    sorted_dims_per_class = [
        np.argsort(np.abs(coef[i]))[::-1].tolist() for i in range(len(classes))
    ]
    seen: set[int] = set()
    result: list[tuple[int, str]] = []
    rank = 0
    while len(result) < n:
        found_any = False
        for cls_idx, cls in enumerate(classes):
            if rank < len(sorted_dims_per_class[cls_idx]):
                dim = sorted_dims_per_class[cls_idx][rank]
                if dim not in seen:
                    seen.add(dim)
                    result.append((dim, cls))
                    found_any = True
                    if len(result) >= n:
                        break
        rank += 1
        if rank > 768:
            break
        if not found_any and rank > 10:
            break
    return result


def fig_image_extremes(
    X: np.ndarray,
    image_list: list,
    model,
    out_dir: Path,
    style: FigureStyle,
    n_dims: int = 3,
    n_images: int = 6,
) -> None:
    dim_info = _collect_discriminative_dims(model, n=n_dims)
    actual_n = len(dim_info)
    ncols = n_images * 2  # low | high

    fig, axes = plt.subplots(
        actual_n, ncols,
        figsize=(ncols * 1.4, actual_n * 1.6),
    )
    if actual_n == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (dim, cls) in enumerate(dim_info):
        activations = X[:, dim]
        low_idxs = np.argsort(activations)[:n_images]
        high_idxs = np.argsort(activations)[::-1][:n_images]

        for col_idx, gi in enumerate(low_idxs):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            try:
                ax.imshow(load_thumb(image_list[gi][0], THUMB_SMALL))
            except Exception:
                pass
            if col_idx == 0:
                ax.set_ylabel(f"Dim {dim}\n({cls})", fontsize=7, rotation=0, labelpad=50, va="center")

        for col_idx, gi in enumerate(high_idxs):
            ax = axes[row_idx, n_images + col_idx]
            ax.axis("off")
            try:
                ax.imshow(load_thumb(image_list[gi][0], THUMB_SMALL))
            except Exception:
                pass

        # divider label between low and high
        axes[row_idx, n_images - 1].set_title("← low", fontsize=7, pad=2)
        axes[row_idx, n_images].set_title("high →", fontsize=7, pad=2)

    fig.suptitle(
        "Image extremes for top discriminative embedding dims  (low ← | → high)",
        fontsize=style.font_size_base,
    )
    fig.tight_layout()

    out = out_dir / "image_extremes"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 5: class probability histograms
# ---------------------------------------------------------------------------


def fig_class_probability_histograms(
    X: np.ndarray,
    y: list[str],
    model,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    colors = {cls: WONG_PALETTE[i] for i, cls in enumerate(classes)}

    fig, axes = plt.subplots(1, 3, figsize=(style.figure_width * 1.4, 3.0), sharey=False)
    for ax, cls in zip(axes, classes):
        col_idx = classes.index(cls)
        cls_mask = np.array([lbl == cls for lbl in y])
        probs = proba[cls_mask, col_idx]
        median_p = np.median(probs)
        ax.hist(probs, bins=20, range=(0, 1), color=colors[cls], edgecolor="white", linewidth=0.4)
        ax.axvline(median_p, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{cls}\nmedian={median_p:.3f}", fontsize=style.font_size_base)
        ax.set_xlabel("Predicted probability", fontsize=style.font_size_base - 1)
        ax.set_ylabel("Count" if cls == classes[0] else "", fontsize=style.font_size_base - 1)
        ax.set_xlim(0, 1)

    fig.suptitle("Predicted class probability distributions", fontsize=style.font_size_base + 1)
    fig.tight_layout()

    out = out_dir / "class_probability_histograms"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 6: cosine similarity matrix
# ---------------------------------------------------------------------------


def fig_cosine_similarity_matrix(
    X: np.ndarray,
    y: list[str],
    model,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    classes = list(model.classes_)
    y_arr = np.array(y)
    n = len(classes)
    sim_matrix = np.zeros((n, n))

    for i, ci in enumerate(classes):
        Xi = X[y_arr == ci]
        # diagonal: mean intra-class cosine similarity
        intra = cosine_similarity(Xi)
        triu = intra[np.triu_indices(len(Xi), k=1)]
        sim_matrix[i, i] = float(np.mean(triu)) if len(triu) > 0 else 1.0
        for j, cj in enumerate(classes):
            if j <= i:
                continue
            Xj = X[y_arr == cj]
            inter = cosine_similarity(Xi, Xj).mean()
            sim_matrix[i, j] = inter
            sim_matrix[j, i] = inter

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=style.font_size_base)
    ax.set_yticklabels(classes, fontsize=style.font_size_base)
    for i in range(n):
        for j in range(n):
            label = "intra" if i == j else ""
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}\n{label}",
                    ha="center", va="center", fontsize=8,
                    color="white" if sim_matrix[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    ax.set_title("Pairwise cosine similarity (diagonal = mean intra-class)", fontsize=style.font_size_base)
    fig.tight_layout()

    out = out_dir / "cosine_similarity_matrix"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 7: PCA projection
# ---------------------------------------------------------------------------


def fig_pca_projection(
    X: np.ndarray,
    y: list[str],
    pca: PCA,
    model,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    proj = pca.transform(X)[:, :2]
    classes = list(model.classes_)
    colors = {cls: WONG_PALETTE[i] for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))
    for cls in classes:
        mask = np.array([lbl == cls for lbl in y])
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=colors[cls], label=cls, s=18, alpha=0.7, linewidths=0)

    # Loading vectors — top-5 dims by combined PC1+PC2 loading magnitude
    loadings = pca.components_[:2].T    # (768, 2)
    loading_mag = np.linalg.norm(loadings, axis=1)
    top5 = np.argsort(loading_mag)[::-1][:5]
    scale = np.abs(proj).max() * 0.5
    for dim in top5:
        lx, ly = loadings[dim] * scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.text(lx * 1.1, ly * 1.1, str(dim), fontsize=7, ha="center")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=style.font_size_base)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=style.font_size_base)
    ax.set_title("PCA projection colored by class", fontsize=style.font_size_base + 1)
    ax.legend(frameon=False, fontsize=style.font_size_base - 1)
    fig.tight_layout()

    out = out_dir / "pca_projection"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 8: PCA component image strips
# ---------------------------------------------------------------------------


def fig_pca_component_strips(
    X: np.ndarray,
    image_list: list,
    pca: PCA,
    out_dir: Path,
    style: FigureStyle,
    n_components: int = 4,
    n_images: int = 5,
) -> None:
    proj = pca.transform(X)[:, :n_components]
    var = pca.explained_variance_ratio_
    ncols = n_images * 2

    fig, axes = plt.subplots(
        n_components, ncols,
        figsize=(ncols * 1.4, n_components * 1.6),
    )

    for row_idx in range(n_components):
        scores = proj[:, row_idx]
        low_idxs = np.argsort(scores)[:n_images]
        high_idxs = np.argsort(scores)[::-1][:n_images]

        for col_idx, gi in enumerate(low_idxs):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            try:
                ax.imshow(load_thumb(image_list[gi][0], THUMB_SMALL))
            except Exception:
                pass

        for col_idx, gi in enumerate(high_idxs):
            ax = axes[row_idx, n_images + col_idx]
            ax.axis("off")
            try:
                ax.imshow(load_thumb(image_list[gi][0], THUMB_SMALL))
            except Exception:
                pass

        axes[row_idx, 0].set_ylabel(
            f"PC{row_idx+1}\n({var[row_idx]*100:.1f}% var)",
            fontsize=7, rotation=0, labelpad=55, va="center",
        )
        axes[row_idx, n_images - 1].set_title("← low", fontsize=7, pad=2)
        axes[row_idx, n_images].set_title("high →", fontsize=7, pad=2)

    fig.suptitle("PCA component image strips  (low ← | → high)", fontsize=style.font_size_base)
    fig.tight_layout()

    out = out_dir / "pca_component_strips"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 9: PCA variance explained
# ---------------------------------------------------------------------------


def fig_pca_variance_explained(
    pca: PCA,
    out_dir: Path,
    style: FigureStyle,
) -> None:
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    n = len(var)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(range(n), var, color=WONG_PALETTE[4], alpha=0.7, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(range(n), cumvar, color=WONG_PALETTE[0], linewidth=1.5, label="Cumulative")

    for thresh, ls in [(0.90, "--"), (0.95, ":")]:
        idx = np.searchsorted(cumvar, thresh)
        ax2.axhline(thresh, color="gray", linestyle=ls, linewidth=0.8)
        ax2.text(n - 1, thresh + 0.005, f"{int(thresh*100)}%", fontsize=7, ha="right", color="gray")
        if idx < n:
            ax2.axvline(idx, color="gray", linestyle=ls, linewidth=0.8)

    ax.set_xlabel("Principal component", fontsize=style.font_size_base)
    ax.set_ylabel("Explained variance ratio", fontsize=style.font_size_base)
    ax2.set_ylabel("Cumulative explained variance", fontsize=style.font_size_base)
    ax.set_title("PCA variance explained (first 50 components)", fontsize=style.font_size_base + 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=style.font_size_base - 1)
    fig.tight_layout()

    out = out_dir / "pca_variance_explained"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 10: patch projection overlays
# ---------------------------------------------------------------------------


def fig_patch_projection_overlays(
    X: np.ndarray,
    y: list[str],
    image_list: list,
    model,
    config,
    out_dir: Path,
    style: FigureStyle,
    n_per_class: int = 5,
) -> None:
    """Overlay per-patch LR coefficient activation as a heatmap on original images.

    For each class, selects the top-n_per_class highest-confidence images, extracts
    DINOv3 patch tokens, projects each patch onto the class's LR coefficient vector,
    and overlays the resulting spatial activation map on the original image.

    Requires torch + transformers (pip install -e ".[ml]").
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.warning("torch not available — skipping patch projection overlays")
        return

    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    classes = list(model.classes_)
    proba = model.predict_proba(X)

    dino_cfg = DinoConfig(
        model_name=config.dino_model_name,
        input_size=config.dino_input_size,
        batch_size=1,
        device=config.dino_device,
    )
    extractor = DinoFeatureExtractor(dino_cfg)

    ncols = n_per_class
    nrows = len(classes)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.4, nrows * 2.6),
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]

    cmap = plt.get_cmap("hot")

    for row_idx, cls in enumerate(classes):
        cls_col = classes.index(cls)
        cls_coef = model.coef_[cls_col]           # (768,) — discriminative direction
        cls_mask = np.array([lbl == cls for lbl in y])
        cls_indices = np.where(cls_mask)[0]
        cls_probs = proba[cls_indices, cls_col]
        top_local = np.argsort(cls_probs)[::-1][:n_per_class]
        top_global = cls_indices[top_local]

        for col_idx, gi in enumerate(top_global):
            img_path, _ = image_list[gi]
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            try:
                # Load original image for display (resize to input_size for alignment)
                pil_img = Image.open(img_path).convert("RGB")
                display_img = np.asarray(
                    pil_img.resize(
                        (config.dino_input_size, config.dino_input_size),
                        Image.LANCZOS,
                    )
                )

                # Extract patch tokens
                patch_tokens = extractor.extract_patch_tokens(pil_img)  # (n_patches, 768)
                n_patches = patch_tokens.shape[0]
                grid_size = int(round(np.sqrt(n_patches)))

                # Project each patch onto the class coefficient vector
                activations = patch_tokens @ cls_coef               # (n_patches,)
                spatial = activations[:grid_size * grid_size].reshape(grid_size, grid_size)

                # Normalize per-image to [0, 1] for full colormap range
                vmin, vmax = spatial.min(), spatial.max()
                if vmax > vmin:
                    spatial_norm = (spatial - vmin) / (vmax - vmin)
                else:
                    spatial_norm = np.zeros_like(spatial)

                # Upsample to display size using PIL
                heat_pil = Image.fromarray((spatial_norm * 255).astype(np.uint8), mode="L")
                heat_up = np.asarray(
                    heat_pil.resize(
                        (config.dino_input_size, config.dino_input_size),
                        Image.BILINEAR,
                    ),
                    dtype=np.float32,
                ) / 255.0

                ax.imshow(display_img)
                ax.imshow(cmap(heat_up), alpha=0.5)
                ax.set_title(f"{cls_probs[top_local[col_idx]]:.3f}", fontsize=7)

            except Exception as e:
                logger.warning("Failed patch overlay for %s: %s", img_path, e)

        axes[row_idx, 0].set_ylabel(
            cls, fontsize=style.font_size_base, rotation=90, labelpad=4,
        )

    fig.suptitle(
        "Patch-level LR coefficient activation overlaid on images\n"
        "(brighter = patch token aligns more with the class discriminative direction)",
        fontsize=style.font_size_base,
    )
    fig.tight_layout()

    out = out_dir / "patch_projection_overlays"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate classifier analysis figures")
    parser.add_argument("--config", default="configs/classifier_config.yaml")
    parser.add_argument(
        "--skip-patch-overlays",
        action="store_true",
        help="Skip patch projection overlays (requires DINOv3 inference)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = load_classifier_config(config_dict)

    out_dir = Path(config.figures_dir) / "classifier_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    style = FigureStyle()
    setup_matplotlib_style(style)

    # Load data
    X, y, image_list, model = load_data(config)

    # Shared PCA (fit once, slice for figs 7/8/9)
    logger.info("Fitting PCA(n_components=50)…")
    pca = PCA(n_components=50, random_state=42)
    pca.fit(X)

    # Generate all figures
    logger.info("Fig 1: class galleries")
    fig_class_galleries(X, y, image_list, model, out_dir, style)

    logger.info("Fig 2: LR coefficient heatmap")
    fig_lr_coefficient_heatmap(model, out_dir, style)

    logger.info("Fig 3: LR coefficient bar charts")
    fig_lr_coefficient_bars(model, out_dir, style)

    logger.info("Fig 4: image extremes")
    fig_image_extremes(X, image_list, model, out_dir, style)

    logger.info("Fig 5: class probability histograms")
    fig_class_probability_histograms(X, y, model, out_dir, style)

    logger.info("Fig 6: cosine similarity matrix")
    fig_cosine_similarity_matrix(X, y, model, out_dir, style)

    logger.info("Fig 7: PCA projection")
    fig_pca_projection(X, y, pca, model, out_dir, style)

    logger.info("Fig 8: PCA component strips")
    fig_pca_component_strips(X, image_list, pca, out_dir, style)

    logger.info("Fig 9: PCA variance explained")
    fig_pca_variance_explained(pca, out_dir, style)

    if not args.skip_patch_overlays:
        logger.info("Fig 10: patch projection overlays (runs DINOv3 inference — use --skip-patch-overlays to skip)")
        fig_patch_projection_overlays(X, y, image_list, model, config, out_dir, style)
    else:
        logger.info("Skipping patch projection overlays")

    logger.info("All figures saved to %s", out_dir)


if __name__ == "__main__":
    main()

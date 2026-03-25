#!/usr/bin/env python3
"""Train a logistic regression classifier on DINOv3 embeddings.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --force        # re-extract + re-train
    python scripts/train_classifier.py --verbose
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.classification.train import (
    ClassifierTrainer,
    load_classifier_config,
    scan_labeled_images,
)
from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor
from src.visualization.style import (
    WONG_PALETTE,
    FigureStyle,
    save_figure,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_or_load_embeddings(
    config,
    image_list: list,
    force: bool,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, y) — from cache if available and force=False."""
    cache_emb = Path(config.embedding_cache)
    cache_lbl = Path(config.label_cache)

    if not force and cache_emb.exists() and cache_lbl.exists():
        logger.info("Loading cached embeddings from %s", cache_emb)
        X = np.load(cache_emb)
        y = list(np.load(cache_lbl))
        if len(X) != len(y):
            raise ValueError(
                f"Embedding cache shape mismatch: {len(X)} embeddings vs {len(y)} labels. "
                "Delete cache files or re-run with --force."
            )
        logger.info("Loaded %d cached embeddings", len(X))
        return X, y

    paths = [p for p, _ in image_list]
    labels = [lbl for _, lbl in image_list]

    dino_cfg = DinoConfig(
        model_name=config.dino_model_name,
        input_size=config.dino_input_size,
        batch_size=config.dino_batch_size,
        device=config.dino_device,
    )
    extractor = DinoFeatureExtractor(dino_cfg)
    X = extractor.extract_batch(paths, batch_size=config.dino_batch_size)
    y = labels

    cache_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_emb, X)
    np.save(cache_lbl, np.array(y))
    logger.info("Embeddings cached to %s (%d images)", cache_emb, len(X))

    return X, y


def _save_confusion_matrix(cv_result, figures_dir: Path, style: FigureStyle) -> None:
    """Save sum-of-folds row-normalised confusion matrix as PNG+SVG."""
    from sklearn.metrics import ConfusionMatrixDisplay

    cm_sum = sum(cv_result.confusion_matrices)
    row_sums = cm_sum.sum(axis=1, keepdims=True)
    cm_norm = cm_sum / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=cv_result.class_names,
    )
    disp.plot(ax=ax, cmap="viridis", values_format=".2f", colorbar=True)
    ax.set_title("Cross-validated confusion matrix (row-normalised)")

    out = figures_dir / "classifier_confusion_matrix"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Confusion matrix saved to %s", out)


def _save_umap_plot(
    X: np.ndarray,
    y: list[str],
    figures_dir: Path,
    style: FigureStyle,
) -> None:
    """Fit UMAP on embeddings and save scatter coloured by class (Wong palette)."""
    import umap

    logger.info("Fitting UMAP for visualisation…")
    reducer = umap.UMAP(
        metric="cosine",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(X)

    classes = sorted(set(y))
    colors = {cls: WONG_PALETTE[i % len(WONG_PALETTE)] for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))
    for cls in classes:
        mask = np.array([lbl == cls for lbl in y])
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=colors[cls],
            label=cls,
            s=20,
            alpha=0.7,
            linewidths=0,
        )
    ax.legend(frameon=False)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("DINOv3 embeddings by class")

    out = figures_dir / "classifier_umap"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("UMAP plot saved to %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pattern classifier")
    parser.add_argument(
        "--config",
        default="configs/classifier_config.yaml",
        help="Path to classifier config YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract embeddings and re-train model even if cached",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = load_classifier_config(config_dict)

    # 2. Scan images
    image_dir = Path(config.image_dir)
    image_list = scan_labeled_images(image_dir, config.label_map)
    class_counts: dict[str, int] = {}
    for _, lbl in image_list:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    logger.info("Found %d images: %s", len(image_list), class_counts)

    # 3. Extract / load embeddings
    X, y = _extract_or_load_embeddings(config, image_list, args.force)

    # 4. Evaluate with CV
    trainer = ClassifierTrainer(X, y, config)
    cv_result = trainer.evaluate()
    logger.info(
        "CV macro-F1: %.3f ± %.3f | Cohen κ: %.3f ± %.3f",
        cv_result.macro_f1_mean,
        cv_result.macro_f1_std,
        cv_result.cohen_kappa_mean,
        cv_result.cohen_kappa_std,
    )

    # 5. Figures
    style = FigureStyle()
    setup_matplotlib_style(style)
    figures_dir = Path(config.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_confusion_matrix(cv_result, figures_dir, style)
    _save_umap_plot(X, y, figures_dir, style)

    # 6. Fit + save final model
    model_path = Path(config.model_dir) / "lr_classifier.joblib"
    if not args.force and model_path.exists():
        logger.info(
            "Model already exists at %s — skipping fit (use --force to retrain)",
            model_path,
        )
    else:
        trainer.fit()
        trainer.save(Path(config.model_dir))

    logger.info(
        "Done. Figures → %s  |  Model → %s",
        config.figures_dir,
        config.model_dir,
    )


if __name__ == "__main__":
    main()

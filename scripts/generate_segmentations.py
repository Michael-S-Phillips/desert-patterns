#!/usr/bin/env python
"""Batch segmentation of desert pattern images using DINOv3 attention + SAM.

For each labeled image in checked-images/, computes a DINOv3 patch attention
map, thresholds it to a binary mask, and prompts SAM to generate instance masks.
Saves per-image overlays, binary masks, attention maps, and metadata JSON.
Also generates per-class gallery figures of the top-12 most-confident examples.

Usage:
    python scripts/generate_segmentations.py [--force] [--verbose]
    python scripts/generate_segmentations.py --image IMG_8346.jpeg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _colorize_attention(attention_map: np.ndarray) -> np.ndarray:
    """Render float32 [0,1] attention map as RGB uint8 using the viridis colormap."""
    from matplotlib import colormaps
    rgba = (colormaps["viridis"](attention_map) * 255).astype(np.uint8)
    return rgba[:, :, :3]


def make_overlay(image_np: np.ndarray, result) -> np.ndarray:
    """Composite original image with attention heatmap and instance mask overlays.

    Blends:
    - Attention heatmap at 30% alpha (underneath masks)
    - Instance masks at 50% alpha (Wong palette)
    - Instance boundary outlines (2px, same color)

    Args:
        image_np: uint8 RGB array, shape (H, W, 3)
        result: SegmentationResult

    Returns:
        uint8 RGB composite, same shape as image_np.
    """
    from src.visualization.style import WONG_PALETTE

    out = image_np.astype(np.float32) / 255.0

    # Attention heatmap at 30% alpha
    heat_rgb = _colorize_attention(result.attention_map).astype(np.float32) / 255.0
    out = out * 0.7 + heat_rgb * 0.3

    # Instance masks at 50% alpha
    for i, mask in enumerate(result.instance_masks):
        r, g, b = _hex_to_rgb(WONG_PALETTE[i % len(WONG_PALETTE)])
        color = np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
        out[mask] = out[mask] * 0.5 + color * 0.5

    out_uint8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)

    # Boundary outlines — convert RGB→BGR for OpenCV, draw, convert back
    out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
    for i, mask in enumerate(result.instance_masks):
        r, g, b = _hex_to_rgb(WONG_PALETTE[i % len(WONG_PALETTE)])
        bgr_color = (b, g, r)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out_bgr, contours, -1, bgr_color, 2)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def save_segmentation_outputs(result, class_dir: Path) -> None:
    """Write attention PNG, binary mask PNG, overlay PNG, and masks JSON.

    Args:
        result: SegmentationResult
        class_dir: Directory for this image's class (e.g. outputs/segmentations/mudcrack/)
    """
    stem = result.image_path.stem
    image_np = np.asarray(Image.open(result.image_path).convert("RGB"))

    # Soft attention heatmap blended with original (50/50)
    heat_rgb = _colorize_attention(result.attention_map)
    attention_overlay = (image_np * 0.5 + heat_rgb * 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(attention_overlay).save(class_dir / f"{stem}_attention.png")

    # Binary mask (0/255)
    Image.fromarray(result.binary_mask).save(class_dir / f"{stem}_mask.png")

    # Full composite overlay
    overlay = make_overlay(image_np, result)
    Image.fromarray(overlay).save(class_dir / f"{stem}_overlay.png")

    # Instance metadata JSON
    masks_data = []
    for mask, iou_score in zip(result.instance_masks, result.iou_scores):
        if mask.any():
            ys, xs = np.where(mask)
            masks_data.append({
                "area_px": int(mask.sum()),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
                "iou_score": float(iou_score),
            })

    with open(class_dir / f"{stem}_masks.json", "w") as f:
        json.dump(masks_data, f, indent=2)

    logger.info("Saved %s — %d instance(s)", stem, len(masks_data))


# ---------------------------------------------------------------------------
# Gallery figure
# ---------------------------------------------------------------------------


def make_gallery_figure(
    class_name: str,
    seg_output_dir: Path,
    model,
    X: np.ndarray,
    y: list[str],
    image_list: list,
    figures_dir: Path,
    style,
) -> None:
    """3×4 gallery of top-12 confidence images showing overlay thumbnails."""
    import matplotlib.pyplot as plt
    from src.visualization.style import save_figure

    class_idx = list(model.classes_).index(class_name)
    proba = model.predict_proba(X)[:, class_idx]

    cls_mask = np.array([lbl == class_name for lbl in y])
    cls_indices = np.where(cls_mask)[0]
    cls_probs = proba[cls_indices]

    n_show = min(12, len(cls_indices))
    top_local = np.argsort(cls_probs)[::-1][:n_show]
    top_global = cls_indices[top_local]

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.6))
    class_dir = seg_output_dir / class_name

    for ax_idx, gi in enumerate(top_global):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        ax.axis("off")
        img_path, _ = image_list[gi]
        overlay_path = class_dir / f"{img_path.stem}_overlay.png"
        if overlay_path.exists():
            thumb = Image.open(overlay_path).convert("RGB").resize((224, 224), Image.LANCZOS)
            ax.imshow(np.asarray(thumb))
            ax.set_title(f"{cls_probs[top_local[ax_idx]]:.3f}", fontsize=7)
        else:
            ax.set_facecolor("#cccccc")

    # Hide unused grid cells
    for ax_idx in range(len(top_global), nrows * ncols):
        axes[ax_idx // ncols, ax_idx % ncols].set_visible(False)

    fig.suptitle(
        f"{class_name} — top-{n_show} by confidence (instance overlay)",
        fontsize=style.font_size_base,
    )
    fig.tight_layout()
    out = figures_dir / f"segmentation_gallery_{class_name}"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved gallery for %s", class_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-segment desert pattern images with DINOv3 attention + SAM."
    )
    parser.add_argument("--config", default="configs/classifier_config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if _overlay.png already exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--image", nargs="+", default=None,
        help="Process only these image files (filename or stem). Gallery skipped.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    from src.classification.train import load_classifier_config, scan_labeled_images
    from src.features.dino_embeddings import load_dino_config
    from src.segmentation.segment import PatternSegmenter, load_segmentation_config
    from src.visualization.style import FigureStyle, setup_matplotlib_style

    clf_config = load_classifier_config(config_dict)
    seg_config = load_segmentation_config(config_dict)
    dino_config = load_dino_config(config_dict.get("dino", {}))

    setup_matplotlib_style()
    style = FigureStyle()

    # Load classifier model and cached data
    model_dir = Path(clf_config.model_dir)
    model = joblib.load(model_dir / "lr_classifier.joblib")
    X = np.load(clf_config.embedding_cache)
    y = list(np.load(clf_config.label_cache, allow_pickle=True))

    # Scan images and validate alignment with label cache
    image_list = scan_labeled_images(Path(clf_config.image_dir), clf_config.label_map)
    if [lbl for _, lbl in image_list] != y:
        raise ValueError(
            "Image list and label cache are misaligned — "
            "re-run scripts/train_classifier.py to regenerate the cache."
        )

    # Filter to --image subset if specified
    if args.image:
        targets = set(args.image)
        image_list_to_process = [
            (p, lbl) for p, lbl in image_list
            if p.name in targets or p.stem in targets
        ]
        if not image_list_to_process:
            raise ValueError(
                f"No images matched --image {args.image!r}. "
                "Check filenames against the checked-images/ directory."
            )
        run_gallery = False
        logger.info("Processing %d specific image(s)", len(image_list_to_process))
    else:
        image_list_to_process = image_list
        run_gallery = True
        logger.info("Processing all %d images", len(image_list))

    # Create output directories
    seg_output_dir = Path(seg_config.output_dir)
    figures_dir = Path(clf_config.figures_dir)
    for class_name in model.classes_:
        (seg_output_dir / class_name).mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialise segmenter (DINOv3 and SAM load lazily on first image)
    segmenter = PatternSegmenter(model, seg_config, dino_config)

    # Process images
    for img_path, class_name in image_list_to_process:
        class_dir = seg_output_dir / class_name
        overlay_path = class_dir / f"{img_path.stem}_overlay.png"
        if overlay_path.exists() and not args.force:
            logger.debug("Skipping %s (overlay exists; use --force to re-run)", img_path.stem)
            continue

        logger.info("Segmenting %s (%s)", img_path.stem, class_name)
        try:
            result = segmenter.segment(img_path, class_name)
            save_segmentation_outputs(result, class_dir)
        except Exception as exc:
            logger.error("Failed to segment %s: %s", img_path.stem, exc, exc_info=True)

    # Gallery figures
    if run_gallery:
        logger.info("Generating per-class gallery figures")
        for class_name in model.classes_:
            make_gallery_figure(
                class_name, seg_output_dir, model, X, y, image_list, figures_dir, style
            )

    logger.info("Done — segmentation outputs at %s", seg_output_dir)


if __name__ == "__main__":
    main()

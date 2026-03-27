#!/usr/bin/env python
"""Batch CLASP segmentation of desert pattern images using DINOv3 + spectral clustering.

For each labeled image in checked-images/, runs CLASP to produce segment labels,
saves per-image overlays (Wong palette), label map PNGs, and metadata JSON.

Usage:
    python scripts/generate_clasp_segmentations.py [--force] [--verbose]
    python scripts/generate_clasp_segmentations.py --image IMG_8346.jpeg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def make_clasp_overlay(image_np: np.ndarray, result) -> np.ndarray:
    """Blend original image with per-segment colors at 50% alpha (Wong palette).

    Args:
        image_np: uint8 RGB array, shape (H, W, 3)
        result: ClaspResult

    Returns:
        uint8 RGB composite, same shape as image_np.
    """
    from src.visualization.style import WONG_PALETTE

    out = image_np.astype(np.float32) / 255.0
    for seg_id in range(result.k_chosen):
        r, g, b = _hex_to_rgb(WONG_PALETTE[seg_id % len(WONG_PALETTE)])
        color = np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
        mask = result.label_map == seg_id
        out[mask] = out[mask] * 0.5 + color * 0.5

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def save_clasp_outputs(result, class_dir: Path) -> None:
    """Write overlay PNG, label map PNG, and segments JSON.

    Args:
        result: ClaspResult
        class_dir: Per-class output directory (e.g. outputs/segmentations_clasp/mudcrack/)
    """
    stem = result.image_path.stem
    image_np = np.asarray(Image.open(result.image_path).convert("RGB"))

    # Segment color overlay
    overlay = make_clasp_overlay(image_np, result)
    Image.fromarray(overlay).save(class_dir / f"{stem}_clasp_overlay.png")

    # Raw label map as indexed-color PNG
    Image.fromarray(result.label_map.astype(np.uint8)).save(
        class_dir / f"{stem}_clasp_labelmap.png"
    )

    # Metadata JSON
    with open(class_dir / f"{stem}_clasp_segments.json", "w") as f:
        json.dump(
            {
                "k": result.k_chosen,
                "segment_areas": result.segment_areas,
                "image_path": str(result.image_path),
            },
            f,
            indent=2,
        )

    logger.info("Saved %s — k=%d segments", stem, result.k_chosen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-segment desert pattern images with CLASP."
    )
    parser.add_argument("--config", default="configs/classifier_config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if _clasp_overlay.png already exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--image", nargs="+", default=None,
        help="Process only these image files (filename or stem).",
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
    from src.segmentation.clasp import ClaspSegmenter, load_clasp_config

    clf_config = load_classifier_config(config_dict)
    clasp_config = load_clasp_config(config_dict)
    dino_config = load_dino_config(config_dict.get("dino", {}))

    image_list = scan_labeled_images(Path(clf_config.image_dir), clf_config.label_map)

    if args.image:
        targets = set(args.image)
        image_list = [
            (p, lbl) for p, lbl in image_list
            if p.name in targets or p.stem in targets
        ]
        if not image_list:
            raise ValueError(
                f"No images matched --image {args.image!r}. "
                "Check filenames against the checked-images/ directory."
            )
        logger.info("Processing %d specific image(s)", len(image_list))
    else:
        logger.info("Processing all %d images", len(image_list))

    # Create output directories
    output_dir = Path(clasp_config.output_dir)
    for _, class_name in image_list:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    segmenter = ClaspSegmenter(clasp_config, dino_config)

    for img_path, class_name in image_list:
        class_dir = output_dir / class_name
        overlay_path = class_dir / f"{img_path.stem}_clasp_overlay.png"
        if overlay_path.exists() and not args.force:
            logger.debug("Skipping %s (overlay exists; use --force to re-run)", img_path.stem)
            continue

        logger.info("Segmenting %s (%s)", img_path.stem, class_name)
        try:
            result = segmenter.segment(img_path)
            save_clasp_outputs(result, class_dir)
        except Exception as exc:
            logger.error("Failed to segment %s: %s", img_path.stem, exc, exc_info=True)

    logger.info("Done — CLASP outputs at %s", output_dir)


if __name__ == "__main__":
    main()

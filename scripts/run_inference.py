#!/usr/bin/env python3
"""Inference script — classify new images into discovered pattern clusters.

Uses fitted UMAP + HDBSCAN models persisted by ``run_clustering.py --save-models``.

Usage:
    python scripts/run_inference.py --model-dir outputs/models/dino_only_filtered --image data/kim_2023/some_image.jpg
    python scripts/run_inference.py --model-dir outputs/models/dino_only_filtered --image-dir data/new_images/ --output predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: Classify new images into pattern clusters."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing fitted models (from --save-models).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a single image to classify.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of images to classify in batch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for batch results.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.image is None and args.image_dir is None:
        parser.error("Provide --image or --image-dir")

    model_dir = args.model_dir if args.model_dir.is_absolute() else PROJECT_ROOT / args.model_dir

    from src.inference.predict import PatternPredictor

    predictor = PatternPredictor(model_dir)

    if args.image is not None:
        image_path = args.image if args.image.is_absolute() else PROJECT_ROOT / args.image
        result = predictor.predict(image_path)
        logger.info(
            "Prediction: cluster=%d (prob=%.3f), UMAP 2D=(%.3f, %.3f)",
            result.cluster_id,
            result.cluster_probability,
            float(result.umap_2d[0]),
            float(result.umap_2d[1]),
        )

    if args.image_dir is not None:
        from src.inference.batch_predict import BatchPredictor

        image_dir = args.image_dir if args.image_dir.is_absolute() else PROJECT_ROOT / args.image_dir
        batch = BatchPredictor(predictor)
        batch_result = batch.predict_directory(image_dir)

        logger.info(
            "Batch: %d predictions, %d failures",
            len(batch_result.predictions),
            len(batch_result.failed_paths),
        )

        output_path = args.output
        if output_path is None:
            output_path = PROJECT_ROOT / "outputs" / "inference_predictions.csv"
        elif not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        batch_result.assignments_df.to_csv(output_path, index=False)
        logger.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()

"""Batch prediction — classify multiple images into pattern clusters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.predict import PatternPredictor, PredictionResult

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class BatchResult:
    """Result of batch prediction."""

    predictions: list[PredictionResult]
    assignments_df: pd.DataFrame
    failed_paths: list[Path] = field(default_factory=list)


class BatchPredictor:
    """Classify multiple images using a fitted pipeline.

    Args:
        predictor: A loaded :class:`PatternPredictor`.
    """

    def __init__(self, predictor: PatternPredictor) -> None:
        self.predictor = predictor

    def predict_directory(
        self,
        image_dir: Path,
        extensions: set[str] | None = None,
    ) -> BatchResult:
        """Classify all images in a directory.

        Args:
            image_dir: Directory to scan for images.
            extensions: File extensions to include (default: jpg, png, tif).

        Returns:
            BatchResult with predictions and a summary DataFrame.
        """
        extensions = extensions or DEFAULT_EXTENSIONS
        image_dir = Path(image_dir)
        paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        logger.info("Found %d images in %s", len(paths), image_dir)
        return self.predict_paths(paths)

    def predict_paths(self, image_paths: list[Path]) -> BatchResult:
        """Classify a list of image paths.

        Args:
            image_paths: Paths to image files.

        Returns:
            BatchResult with predictions and a summary DataFrame.
        """
        predictions: list[PredictionResult] = []
        failed: list[Path] = []
        records: list[dict] = []

        for i, path in enumerate(image_paths):
            try:
                result = self.predictor.predict(Path(path))
                predictions.append(result)
                records.append({
                    "image_path": str(path),
                    "image_name": Path(path).name,
                    "cluster_id": result.cluster_id,
                    "cluster_probability": result.cluster_probability,
                    "umap_x": float(result.umap_2d[0]),
                    "umap_y": float(result.umap_2d[1]),
                })
                if (i + 1) % 10 == 0:
                    logger.info("Predicted %d / %d images", i + 1, len(image_paths))
            except Exception as e:
                logger.warning("Failed to predict %s: %s", path, e)
                failed.append(Path(path))

        df = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["image_path", "image_name", "cluster_id",
                     "cluster_probability", "umap_x", "umap_y"]
        )

        logger.info(
            "Batch prediction complete: %d succeeded, %d failed",
            len(predictions), len(failed),
        )

        return BatchResult(
            predictions=predictions,
            assignments_df=df,
            failed_paths=failed,
        )

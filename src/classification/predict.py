"""Supervised pattern classifier predictor — loads persisted model and classifies images."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

if TYPE_CHECKING:
    from src.features.dino_embeddings import DinoFeatureExtractor

logger = logging.getLogger(__name__)


class ClassifierPredictor:
    """Classify new images using a persisted logistic regression model.

    Loads ``lr_classifier.joblib`` and ``label_encoder.joblib`` from
    ``model_dir`` on construction.  DINOv3 extraction is lazy: the model is
    not loaded until :meth:`predict` is first called.

    Args:
        model_dir: Directory produced by :meth:`ClassifierTrainer.save`.
    """

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self._model = joblib.load(model_dir / "lr_classifier.joblib")
        self._le = joblib.load(model_dir / "label_encoder.joblib")
        self._extractor: DinoFeatureExtractor | None = None

        logger.info(
            "ClassifierPredictor loaded from %s, classes: %s",
            model_dir,
            list(self._model.classes_),
        )

    def predict_from_embedding(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """Classify a pre-computed DINOv3 embedding.

        Args:
            embedding: 1-D float32 array of shape ``(768,)``.

        Returns:
            Tuple of (class_name, probability, all_class_probs) where
            all_class_probs maps each class name to its predicted probability.
        """
        emb = embedding.reshape(1, -1)
        probs = self._model.predict_proba(emb)[0]
        class_names: list[str] = list(self._model.classes_)
        best_idx = int(np.argmax(probs))
        return (
            class_names[best_idx],
            float(probs[best_idx]),
            {cls: float(p) for cls, p in zip(class_names, probs)},
        )

    def predict(
        self,
        image_path: Path,
    ) -> tuple[str, float, dict[str, float]]:
        """Extract a DINOv3 embedding and classify an image.

        Requires ``torch`` and ``transformers`` (installed via
        ``pip install -e ".[ml]"``).

        Args:
            image_path: Path to a JPEG or PNG image file.

        Returns:
            Tuple of (class_name, probability, all_class_probs).
        """
        from PIL import Image

        from src.features.dino_embeddings import DinoFeatureExtractor

        if self._extractor is None:
            # Note: uses default DinoConfig() — dino_* fields from ClassifierConfig
            # are not forwarded here. Pass a DinoConfig explicitly if needed.
            self._extractor = DinoFeatureExtractor()

        img = Image.open(image_path).convert("RGB")
        embedding = self._extractor.extract_cls(img)
        return self.predict_from_embedding(embedding)

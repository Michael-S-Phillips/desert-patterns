"""Single-image prediction using persisted UMAP + HDBSCAN models.

Loads fitted models from a model directory (produced by
``scripts/run_clustering.py --save-models``) and classifies new images
into the discovered pattern clusters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from src.data.preprocessing import standardize_image

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of classifying a single image."""

    cluster_id: int
    cluster_probability: float
    umap_2d: np.ndarray  # (2,)
    umap_3d: np.ndarray  # (3,)
    umap_high_dim: np.ndarray  # (D,)
    features_raw: np.ndarray  # (768,)


class PatternPredictor:
    """Classify new images into discovered pattern clusters.

    Args:
        model_dir: Directory containing ``umap_high_dim.joblib``,
            ``umap_2d.joblib``, ``umap_3d.joblib``, ``hdbscan.joblib``,
            and ``pipeline_config.json``.
    """

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        config_path = model_dir / "pipeline_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        self._umap_high = joblib.load(model_dir / "umap_high_dim.joblib")
        self._umap_2d = joblib.load(model_dir / "umap_2d.joblib")
        self._umap_3d = joblib.load(model_dir / "umap_3d.joblib")
        self._hdbscan = joblib.load(model_dir / "hdbscan.joblib")

        # Load curation directions if present
        curation_path = model_dir / "curation.joblib"
        if curation_path.exists():
            self._curation_directions = joblib.load(curation_path)
            logger.info(
                "Loaded %d curation directions from %s",
                self._curation_directions.shape[0],
                curation_path,
            )
        else:
            self._curation_directions = None

        self._dino: Any = None

        logger.info("Loaded models from %s (ablation=%s)", model_dir, self.config.get("ablation_name"))

    def _ensure_dino(self) -> None:
        """Lazily load the DINOv3 feature extractor."""
        if self._dino is not None:
            return
        from src.features.dino_embeddings import DinoFeatureExtractor
        self._dino = DinoFeatureExtractor()
        logger.info("DINOv3 extractor loaded for inference")

    def predict(self, image_path: Path) -> PredictionResult:
        """Classify a single image.

        Reads the image, standardizes to 518x518, extracts DINOv3 [CLS]
        embedding, then projects through the fitted pipeline.

        Args:
            image_path: Path to the image file.

        Returns:
            PredictionResult with cluster assignment and embeddings.
        """
        self._ensure_dino()

        # Read and standardize
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        std = standardize_image(bgr, target_size=518)

        # Convert BGR→RGB→PIL for DINOv3
        rgb = cv2.cvtColor(std, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        features = self._dino.extract_cls(pil_img)
        return self.predict_from_features(features)

    def predict_from_features(self, features: np.ndarray) -> PredictionResult:
        """Classify using pre-extracted features.

        If curation directions were saved with the model, the scene-content
        directions are projected out before UMAP transform.

        Args:
            features: 1-D feature vector (e.g. 768-d DINOv3 [CLS]).

        Returns:
            PredictionResult with cluster assignment and embeddings.
        """
        import hdbscan

        features_2d = features.reshape(1, -1)

        # Apply curation if directions were saved with the model
        if self._curation_directions is not None:
            from src.features.feature_curation import project_out_directions
            features_2d = project_out_directions(features_2d, self._curation_directions)

        # Project through UMAP high-dim
        high_dim = self._umap_high.transform(features_2d).astype(np.float32)

        # HDBSCAN approximate prediction
        labels, probs = hdbscan.approximate_predict(self._hdbscan, high_dim)

        # Project through 2D and 3D UMAP
        umap_2d = self._umap_2d.transform(features_2d).astype(np.float32)
        umap_3d = self._umap_3d.transform(features_2d).astype(np.float32)

        return PredictionResult(
            cluster_id=int(labels[0]),
            cluster_probability=float(probs[0]),
            umap_2d=umap_2d[0],
            umap_3d=umap_3d[0],
            umap_high_dim=high_dim[0],
            features_raw=features,
        )

    def find_nearest_neighbors(
        self,
        prediction: PredictionResult,
        assignments_df: pd.DataFrame,
        n: int = 5,
    ) -> list[str]:
        """Find the nearest existing patches in 2D UMAP space.

        Args:
            prediction: Result from :meth:`predict` or :meth:`predict_from_features`.
            assignments_df: DataFrame with ``image_id``, ``umap_x``, ``umap_y``.
            n: Number of neighbors to return.

        Returns:
            List of image IDs for the *n* closest patches.
        """
        coords = assignments_df[["umap_x", "umap_y"]].values
        query = prediction.umap_2d.reshape(1, 2)
        distances = np.linalg.norm(coords - query, axis=1)
        indices = np.argsort(distances)[:n]
        return assignments_df.iloc[indices]["image_id"].tolist()

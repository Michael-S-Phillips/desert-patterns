"""HDF5-backed feature storage for the desert patterns pipeline.

Manages computed features (DINOv3 embeddings, texture descriptors, fused vectors)
with incremental storage, validation, and provenance tracking.

Schema::

    /image_ids             (N,)    variable-length UTF-8 strings
    /dino_cls              (N, 768) float32
    /texture               (N, 90)  float32
    /texture_feature_names (90,)    variable-length UTF-8 strings
    /dino_pca              (N, D1)  float32
    /texture_pca           (N, D2)  float32
    /fused                 (N, D3)  float32

Each dataset carries ``attrs`` for provenance (creation time, config hash, etc.).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# HDF5 string dtype for variable-length UTF-8
_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


class FeatureStore:
    """HDF5 feature store with incremental save/load and validation.

    Args:
        store_path: Path to the HDF5 file.  Created on first write if absent.
    """

    def __init__(self, store_path: str | Path):
        self.store_path = Path(store_path)

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_image_ids(self, image_ids: list[str]) -> None:
        """Save image IDs to the store.

        Args:
            image_ids: List of image identifier strings.
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.store_path, "a") as f:
            if "image_ids" in f:
                del f["image_ids"]
            f.create_dataset("image_ids", data=image_ids, dtype=_STR_DTYPE)
            f["image_ids"].attrs["saved_at"] = _now_iso()

    def save_dino(self, embeddings: np.ndarray) -> None:
        """Save DINOv3 [CLS] embeddings.

        Args:
            embeddings: Array of shape ``(N, 768)`` float32.
        """
        self._save_array("dino_cls", embeddings.astype(np.float32))

    def save_texture(
        self,
        features: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """Save texture feature vectors.

        Args:
            features: Array of shape ``(N, 90)`` float32.
            feature_names: Optional list of feature names.
        """
        self._save_array("texture", features.astype(np.float32))
        if feature_names is not None:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(self.store_path, "a") as f:
                if "texture_feature_names" in f:
                    del f["texture_feature_names"]
                f.create_dataset(
                    "texture_feature_names", data=feature_names, dtype=_STR_DTYPE
                )

    def save_fused(
        self,
        fused: np.ndarray,
        dino_pca: np.ndarray | None = None,
        texture_pca: np.ndarray | None = None,
    ) -> None:
        """Save fused feature vectors (and optional intermediate PCA arrays).

        Args:
            fused: Fused feature array of shape ``(N, D)``.
            dino_pca: Optional DINOv3 PCA-reduced array.
            texture_pca: Optional texture PCA-reduced array.
        """
        self._save_array("fused", fused.astype(np.float32))
        if dino_pca is not None:
            self._save_array("dino_pca", dino_pca.astype(np.float32))
        if texture_pca is not None:
            self._save_array("texture_pca", texture_pca.astype(np.float32))

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_image_ids(self) -> list[str]:
        """Load image IDs from the store.

        Returns:
            List of image identifier strings.
        """
        with h5py.File(self.store_path, "r") as f:
            raw = f["image_ids"][:]
            return [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw]

    def load_dino(self) -> np.ndarray:
        """Load DINOv3 [CLS] embeddings.

        Returns:
            Array of shape ``(N, 768)`` float32.
        """
        return self._load_array("dino_cls")

    def load_texture(self) -> np.ndarray:
        """Load texture feature vectors.

        Returns:
            Array of shape ``(N, 90)`` float32.
        """
        return self._load_array("texture")

    def load_texture_feature_names(self) -> list[str]:
        """Load texture feature names.

        Returns:
            List of feature name strings.
        """
        with h5py.File(self.store_path, "r") as f:
            raw = f["texture_feature_names"][:]
            return [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw]

    def load_fused(self) -> np.ndarray:
        """Load fused feature vectors.

        Returns:
            Array of shape ``(N, D)`` float32.
        """
        return self._load_array("fused")

    def load_dino_pca(self) -> np.ndarray:
        """Load DINOv3 PCA-reduced features.

        Returns:
            Array of shape ``(N, D1)`` float32.
        """
        return self._load_array("dino_pca")

    def load_texture_pca(self) -> np.ndarray:
        """Load texture PCA-reduced features.

        Returns:
            Array of shape ``(N, D2)`` float32.
        """
        return self._load_array("texture_pca")

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def has_dino(self) -> bool:
        """Check whether DINOv3 embeddings exist in the store."""
        return self._has_dataset("dino_cls")

    def has_texture(self) -> bool:
        """Check whether texture features exist in the store."""
        return self._has_dataset("texture")

    def has_fused(self) -> bool:
        """Check whether fused features exist in the store."""
        return self._has_dataset("fused")

    def get_n_images(self) -> int:
        """Return the number of images stored, or 0 if empty."""
        if not self.store_path.exists():
            return 0
        with h5py.File(self.store_path, "r") as f:
            if "image_ids" in f:
                return f["image_ids"].shape[0]
        return 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the feature store for common issues.

        Checks for NaN/Inf values, shape mismatches between datasets,
        and consistency with image_ids.

        Returns:
            List of warning/error strings.  Empty list means all clean.
        """
        issues: list[str] = []

        if not self.store_path.exists():
            issues.append("Feature store file does not exist")
            return issues

        with h5py.File(self.store_path, "r") as f:
            if "image_ids" not in f:
                issues.append("Missing /image_ids dataset")
                return issues

            n_images = f["image_ids"].shape[0]

            for name in ["dino_cls", "texture", "fused", "dino_pca", "texture_pca"]:
                if name not in f:
                    continue
                data = f[name][:]

                # Shape check
                if data.shape[0] != n_images:
                    issues.append(
                        f"/{name} has {data.shape[0]} rows but "
                        f"/image_ids has {n_images}"
                    )

                # NaN check
                nan_count = int(np.isnan(data).sum())
                if nan_count > 0:
                    issues.append(f"/{name} contains {nan_count} NaN values")

                # Inf check
                inf_count = int(np.isinf(data).sum())
                if inf_count > 0:
                    issues.append(f"/{name} contains {inf_count} Inf values")

        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_array(self, name: str, data: np.ndarray) -> None:
        """Save a numpy array as an HDF5 dataset, replacing if exists."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.store_path, "a") as f:
            if name in f:
                del f[name]
            f.create_dataset(name, data=data)
            f[name].attrs["saved_at"] = _now_iso()
            f[name].attrs["shape"] = list(data.shape)

    def _load_array(self, name: str) -> np.ndarray:
        """Load a numpy array from an HDF5 dataset."""
        with h5py.File(self.store_path, "r") as f:
            return f[name][:].astype(np.float32)

    def _has_dataset(self, name: str) -> bool:
        """Check if a dataset exists in the store."""
        if not self.store_path.exists():
            return False
        with h5py.File(self.store_path, "r") as f:
            return name in f


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()

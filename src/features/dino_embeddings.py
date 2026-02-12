"""DINOv3 ViT-B/16 feature extraction for desert pattern images.

Extracts 768-dimensional [CLS] embeddings (and optional patch tokens) using
Meta's DINOv3 model via HuggingFace ``transformers``.  The model is lazily
loaded on first use so this module imports cleanly without torch installed.

Requires ``pip install -e ".[ml]"`` for torch + transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DinoConfig:
    """Configuration for DINOv3 feature extraction."""

    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    input_size: int = 518
    batch_size: int = 32
    extract_patch_tokens: bool = False
    device: str = "auto"


def load_dino_config(config_dict: dict) -> DinoConfig:
    """Create a DinoConfig from a config dictionary.

    Args:
        config_dict: Dictionary with dino configuration values.

    Returns:
        Populated DinoConfig.
    """
    return DinoConfig(
        model_name=config_dict.get("model_name", DinoConfig.model_name),
        input_size=config_dict.get("input_size", DinoConfig.input_size),
        batch_size=config_dict.get("batch_size", DinoConfig.batch_size),
        extract_patch_tokens=config_dict.get(
            "extract_patch_tokens", DinoConfig.extract_patch_tokens
        ),
        device=config_dict.get("device", DinoConfig.device),
    )


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class DinoFeatureExtractor:
    """Extract DINOv3 ViT-B/16 embeddings from images.

    Lazily loads the model and processor on first use (same pattern as
    ``SAMGroundMasker``).  Requires ``torch`` and ``transformers`` packages
    (installed via ``pip install -e ".[ml]"``).

    Args:
        config: DINOv3 extraction configuration.
    """

    def __init__(self, config: DinoConfig | None = None):
        self.config = config or DinoConfig()
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None

    def _resolve_device(self) -> str:
        """Resolve the device string, preferring MPS over CPU (never CUDA)."""
        import torch

        if self.config.device != "auto":
            return self.config.device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Load DINOv3 model and processor lazily."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._device = self._resolve_device()
        logger.info(
            "Loading DINOv3 model %s on device=%s",
            self.config.model_name,
            self._device,
        )

        self._processor = AutoImageProcessor.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(self.config.model_name)
        self._model.eval().to(self._device)
        logger.info("DINOv3 model loaded successfully")

    @property
    def model(self) -> Any:
        """Lazily loaded model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self) -> Any:
        """Lazily loaded image processor."""
        if self._processor is None:
            self._load_model()
        return self._processor

    @property
    def device(self) -> str:
        """Resolved device string."""
        if self._device is None:
            self._load_model()
        return self._device  # type: ignore[return-value]

    def extract_cls(self, image: Image.Image) -> np.ndarray:
        """Extract the [CLS] token embedding from a single image.

        Args:
            image: PIL Image (any mode — processor handles conversion).

        Returns:
            1-D numpy array of shape ``(768,)`` with float32 values.
        """
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.pooler_output[0].cpu().numpy()
        return cls_embedding.astype(np.float32)

    def extract_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """Extract patch-level token embeddings from a single image.

        Skips the first 5 tokens (1 CLS + 4 register tokens) to return
        only the spatial patch tokens.

        Args:
            image: PIL Image.

        Returns:
            2-D numpy array of shape ``(n_patches, 768)`` with float32 values.
        """
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Skip 1 CLS + 4 register tokens
        patch_tokens = outputs.last_hidden_state[0, 5:, :].cpu().numpy()
        return patch_tokens.astype(np.float32)

    def extract_batch(
        self,
        image_paths: list[Path] | list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Extract [CLS] embeddings for a batch of images.

        Args:
            image_paths: List of paths to image files.
            batch_size: Override the config batch size.

        Returns:
            2-D numpy array of shape ``(N, 768)`` with float32 values.
        """
        import torch

        bs = batch_size or self.config.batch_size
        n = len(image_paths)
        embeddings: list[np.ndarray] = []
        log_interval = max(1, 10)

        for batch_idx in range(0, n, bs):
            batch_end = min(batch_idx + bs, n)
            batch_num = batch_idx // bs
            if batch_num % log_interval == 0:
                logger.info(
                    "DINOv3 batch %d/%d (images %d–%d of %d)",
                    batch_num + 1,
                    (n + bs - 1) // bs,
                    batch_idx + 1,
                    batch_end,
                    n,
                )

            images = []
            for p in image_paths[batch_idx:batch_end]:
                img = Image.open(p).convert("RGB")
                images.append(img)

            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            cls_batch = outputs.pooler_output.cpu().numpy().astype(np.float32)
            embeddings.append(cls_batch)

        return np.concatenate(embeddings, axis=0)

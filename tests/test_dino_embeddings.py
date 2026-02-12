"""Tests for DINOv3 feature extraction.

Uses mocked model/processor to avoid downloading the actual model during tests.
Tests that need torch are skipped when it is not installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.features.dino_embeddings import (
    DinoConfig,
    DinoFeatureExtractor,
    load_dino_config,
)

torch = pytest.importorskip("torch", reason="torch required for DINO tests")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestDinoConfig:
    """Tests for DinoConfig and load_dino_config."""

    def test_defaults(self):
        cfg = DinoConfig()
        assert cfg.model_name == "facebook/dinov3-vitb16-pretrain-lvd1689m"
        assert cfg.input_size == 518
        assert cfg.batch_size == 32
        assert cfg.extract_patch_tokens is False
        assert cfg.device == "auto"

    def test_load_from_dict(self):
        d = {"model_name": "custom/model", "batch_size": 16, "device": "cpu"}
        cfg = load_dino_config(d)
        assert cfg.model_name == "custom/model"
        assert cfg.batch_size == 16
        assert cfg.device == "cpu"
        # Defaults preserved for missing keys
        assert cfg.input_size == 518

    def test_load_empty_dict(self):
        cfg = load_dino_config({})
        assert cfg.model_name == DinoConfig.model_name


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_outputs(batch_size: int = 1, embed_dim: int = 768, seq_len: int = 1030):
    """Create a mock model output with pooler_output and last_hidden_state."""
    outputs = MagicMock()
    outputs.pooler_output = torch.randn(batch_size, embed_dim)
    outputs.last_hidden_state = torch.randn(batch_size, seq_len, embed_dim)
    return outputs


def _make_mock_processor():
    """Create a mock processor that returns fake tensors."""
    processor = MagicMock()
    processor.side_effect = lambda images, return_tensors: {
        "pixel_values": torch.randn(
            len(images) if isinstance(images, list) else 1, 3, 518, 518
        )
    }
    return processor


def _make_mock_model(batch_size: int = 1):
    """Create a mock model that returns fake outputs."""
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model

    def forward(**kwargs):
        bs = kwargs["pixel_values"].shape[0]
        return _make_mock_outputs(batch_size=bs)

    model.__call__ = forward
    model.side_effect = forward
    return model


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------


class TestDinoFeatureExtractor:
    """Tests for DinoFeatureExtractor using mocked model."""

    @pytest.fixture
    def extractor(self):
        """Create an extractor with mocked internals."""
        ext = DinoFeatureExtractor(DinoConfig(device="cpu"))
        ext._device = "cpu"
        ext._processor = _make_mock_processor()
        ext._model = _make_mock_model()
        return ext

    def test_extract_cls_shape(self, extractor):
        """CLS embedding should be (768,) float32."""
        img = Image.fromarray(np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8))
        result = extractor.extract_cls(img)
        assert result.shape == (768,)
        assert result.dtype == np.float32

    def test_extract_patch_tokens_skips_5_tokens(self, extractor):
        """Patch tokens should skip 1 CLS + 4 register tokens."""
        img = Image.fromarray(np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8))
        result = extractor.extract_patch_tokens(img)
        # last_hidden_state has 1030 tokens → 1030 - 5 = 1025 patch tokens
        assert result.shape == (1025, 768)
        assert result.dtype == np.float32

    def test_extract_batch_shape(self, tmp_path, extractor):
        """Batch extraction should return (N, 768)."""
        paths = []
        for i in range(5):
            p = tmp_path / f"img_{i}.png"
            img = Image.fromarray(
                np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
            )
            img.save(p)
            paths.append(p)

        result = extractor.extract_batch(paths, batch_size=2)
        assert result.shape == (5, 768)
        assert result.dtype == np.float32

    def test_lazy_loading_not_called_on_init(self):
        """Model should not be loaded on __init__."""
        ext = DinoFeatureExtractor(DinoConfig(device="cpu"))
        assert ext._model is None
        assert ext._processor is None

    def test_device_resolution_cpu(self):
        """Device 'cpu' should resolve directly."""
        ext = DinoFeatureExtractor(DinoConfig(device="cpu"))
        ext._device = "cpu"
        ext._processor = _make_mock_processor()
        ext._model = _make_mock_model()
        assert ext.device == "cpu"

    def test_device_resolution_auto(self):
        """Device 'auto' should resolve to mps or cpu."""
        ext = DinoFeatureExtractor(DinoConfig(device="auto"))
        device = ext._resolve_device()
        assert device in ("cpu", "mps")

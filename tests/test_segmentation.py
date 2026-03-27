"""Tests for src/segmentation/segment.py."""
from __future__ import annotations

import numpy as np
import pytest


def test_segmentation_config_defaults():
    from src.segmentation.segment import SegmentationConfig
    cfg = SegmentationConfig()
    assert cfg.n_foreground_prompts == 5
    assert cfg.n_background_prompts == 5
    assert cfg.threshold_method == "otsu"
    assert cfg.attention_percentile == pytest.approx(0.70)
    assert cfg.min_mask_area_fraction == pytest.approx(0.005)
    assert cfg.iou_dedup_threshold == pytest.approx(0.5)
    assert cfg.output_dir == "outputs/segmentations"


def test_load_segmentation_config_overrides():
    from src.segmentation.segment import load_segmentation_config
    cfg_dict = {
        "segmentation": {
            "n_foreground_prompts": 3,
            "threshold_method": "percentile",
            "attention_percentile": 0.80,
        }
    }
    cfg = load_segmentation_config(cfg_dict)
    assert cfg.n_foreground_prompts == 3
    assert cfg.threshold_method == "percentile"
    assert cfg.attention_percentile == pytest.approx(0.80)
    assert cfg.n_background_prompts == 5  # default unchanged


def test_load_segmentation_config_missing_section():
    from src.segmentation.segment import SegmentationConfig, load_segmentation_config
    cfg = load_segmentation_config({})
    assert cfg.threshold_method == SegmentationConfig.threshold_method
    assert cfg.n_foreground_prompts == SegmentationConfig.n_foreground_prompts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock


def _make_segmenter(threshold_method: str = "otsu", attention_percentile: float = 0.70):
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    seg_config = SegmentationConfig(
        threshold_method=threshold_method,
        attention_percentile=attention_percentile,
    )
    return PatternSegmenter(MagicMock(), seg_config, DinoConfig())


# ---------------------------------------------------------------------------
# Attention map tests
# ---------------------------------------------------------------------------


def test_compute_attention_output_shape():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(42)
    patch_tokens = rng.standard_normal((196, 768)).astype(np.float32)
    cls_coef = rng.standard_normal(768).astype(np.float32)
    attn = segmenter._compute_attention(patch_tokens, cls_coef, image_size=(320, 240))
    assert attn.shape == (240, 320)  # (height, width)


def test_compute_attention_normalized():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(0)
    patch_tokens = rng.standard_normal((196, 768)).astype(np.float32)
    cls_coef = rng.standard_normal(768).astype(np.float32)
    attn = segmenter._compute_attention(patch_tokens, cls_coef, image_size=(128, 128))
    assert attn.min() >= 0.0
    assert attn.max() <= 1.0
    assert attn.dtype == np.float32


def test_compute_attention_dynamic_grid():
    """Works for non-196 patch counts (10x10 = 100 patches)."""
    segmenter = _make_segmenter()
    rng = np.random.default_rng(1)
    patch_tokens = rng.standard_normal((100, 768)).astype(np.float32)
    cls_coef = rng.standard_normal(768).astype(np.float32)
    attn = segmenter._compute_attention(patch_tokens, cls_coef, image_size=(200, 150))
    assert attn.shape == (150, 200)


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------


def test_threshold_percentile_values_and_dtype():
    segmenter = _make_segmenter(threshold_method="percentile", attention_percentile=0.5)
    attn = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    binary = segmenter._threshold(attn)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})


def test_threshold_otsu_bimodal():
    segmenter = _make_segmenter(threshold_method="otsu")
    attn = np.zeros((100, 100), dtype=np.float32)
    attn[:50, :] = 0.9  # top half bright
    binary = segmenter._threshold(attn)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
    # Top half should be foreground (255)
    assert binary[:50, :].mean() > 200
    # Bottom half should be background (0)
    assert binary[50:, :].mean() < 55

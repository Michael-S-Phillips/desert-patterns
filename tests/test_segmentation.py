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


# ---------------------------------------------------------------------------
# Patch centroid tests
# ---------------------------------------------------------------------------


def test_patch_centroids_shape():
    segmenter = _make_segmenter()
    attn = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    fg, bg = segmenter._patch_centroids(attn, n_patches=196)
    assert fg.shape == (5, 2)  # n_foreground_prompts=5 by default
    assert bg.shape == (5, 2)
    assert fg.dtype == np.float32


def test_patch_centroids_coords_in_bounds():
    segmenter = _make_segmenter()
    attn = np.random.default_rng(1).random((200, 300)).astype(np.float32)
    fg, bg = segmenter._patch_centroids(attn, n_patches=196)
    assert fg[:, 0].max() <= 300  # x within width
    assert fg[:, 1].max() <= 200  # y within height
    assert fg.min() >= 0


def test_patch_centroids_top_left_corner():
    """Lowest-activation region appears in background prompts."""
    segmenter = _make_segmenter()
    # Make entire attention = 1 except top-left block which is 0
    attn = np.ones((140, 140), dtype=np.float32)
    attn[:10, :10] = 0.0  # top-left patch region lowest
    _, bg = segmenter._patch_centroids(attn, n_patches=196)
    # Center of top-left patch (140/14=10px per patch) → centroid ≈ (5, 5)
    assert bg[0, 0] == pytest.approx(5.0, abs=5.0)
    assert bg[0, 1] == pytest.approx(5.0, abs=5.0)


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------


def test_iou_identical():
    segmenter = _make_segmenter()
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    assert segmenter._iou(mask, mask) == pytest.approx(1.0)


def test_iou_disjoint():
    segmenter = _make_segmenter()
    a = np.zeros((50, 50), dtype=bool)
    b = np.zeros((50, 50), dtype=bool)
    a[0:5, 0:5] = True
    b[10:15, 10:15] = True
    assert segmenter._iou(a, b) == pytest.approx(0.0)


def test_iou_partial_overlap():
    segmenter = _make_segmenter()
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:4] = True  # 16 px
    b[2:6, 2:6] = True  # 16 px, overlap = 2x2 = 4 px → union = 28
    assert segmenter._iou(a, b) == pytest.approx(4 / 28)


def test_iou_empty_masks():
    segmenter = _make_segmenter()
    empty = np.zeros((50, 50), dtype=bool)
    assert segmenter._iou(empty, empty) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


def test_deduplicate_removes_near_duplicate():
    segmenter = _make_segmenter()
    base = np.zeros((100, 100), dtype=bool)
    base[10:50, 10:50] = True  # 40x40 = 1600 px

    near_dup = np.zeros((100, 100), dtype=bool)
    near_dup[12:52, 12:52] = True  # heavily overlaps base

    distinct = np.zeros((100, 100), dtype=bool)
    distinct[60:80, 60:80] = True  # non-overlapping

    kept_masks, kept_scores = segmenter._deduplicate_masks(
        [base, near_dup, distinct], [0.9, 0.8, 0.7]
    )
    assert len(kept_masks) == 2
    assert kept_scores[0] == pytest.approx(0.9)  # base kept (highest score)


def test_deduplicate_all_distinct():
    segmenter = _make_segmenter()
    a = np.zeros((100, 100), dtype=bool)
    b = np.zeros((100, 100), dtype=bool)
    a[0:10, 0:10] = True
    b[50:60, 50:60] = True
    kept_masks, kept_scores = segmenter._deduplicate_masks([a, b], [0.8, 0.9])
    assert len(kept_masks) == 2


def test_deduplicate_empty_input():
    segmenter = _make_segmenter()
    kept_masks, kept_scores = segmenter._deduplicate_masks([], [])
    assert kept_masks == []
    assert kept_scores == []

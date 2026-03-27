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

"""Tests for src/segmentation/clasp.py."""
from __future__ import annotations

import numpy as np
import pytest


def test_clasp_config_defaults():
    from src.segmentation.clasp import ClaspConfig
    cfg = ClaspConfig()
    assert cfg.output_dir == "outputs/segmentations_clasp"
    assert cfg.bandwidth == pytest.approx(0.5)
    assert cfg.min_clusters == 2
    assert cfg.max_clusters == 15
    assert cfg.dense_crf is True
    assert cfg.crf_iterations == 20
    assert cfg.crf_gt_prob == pytest.approx(0.8)
    assert cfg.crf_gaussian_sxy == 4
    assert cfg.crf_gaussian_compat == 4
    assert cfg.crf_bilateral_sxy == 80
    assert cfg.crf_bilateral_srgb == 13
    assert cfg.crf_bilateral_compat == 10


def test_load_clasp_config_overrides():
    from src.segmentation.clasp import load_clasp_config
    cfg = load_clasp_config({"clasp": {"bandwidth": 0.3, "min_clusters": 3, "dense_crf": False}})
    assert cfg.bandwidth == pytest.approx(0.3)
    assert cfg.min_clusters == 3
    assert cfg.dense_crf is False
    assert cfg.max_clusters == 15   # default unchanged


def test_load_clasp_config_missing_section():
    from src.segmentation.clasp import ClaspConfig, load_clasp_config
    cfg = load_clasp_config({})
    assert cfg.bandwidth == ClaspConfig.bandwidth
    assert cfg.crf_iterations == ClaspConfig.crf_iterations

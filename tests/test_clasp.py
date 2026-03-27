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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from src.features.dino_embeddings import DinoConfig


def _make_segmenter(dense_crf: bool = False):
    from src.segmentation.clasp import ClaspConfig, ClaspSegmenter
    cfg = ClaspConfig(dense_crf=dense_crf)
    return ClaspSegmenter(cfg, DinoConfig())


# ---------------------------------------------------------------------------
# Affinity matrix tests
# ---------------------------------------------------------------------------


def test_build_affinity_shape():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((100, 768)).astype(np.float32)
    A = segmenter._build_affinity(tokens)
    assert A.shape == (100, 100)


def test_build_affinity_diagonal():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(1)
    tokens = rng.standard_normal((50, 768)).astype(np.float32)
    A = segmenter._build_affinity(tokens)
    np.testing.assert_allclose(np.diag(A), 1.0, atol=1e-5)


def test_build_affinity_range():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(2)
    tokens = rng.standard_normal((80, 768)).astype(np.float32)
    A = segmenter._build_affinity(tokens)
    assert A.min() >= 0.0
    assert A.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Eigengap tests
# ---------------------------------------------------------------------------


def test_eigengap_k_clear_drop():
    """Sharp drop between λ_3 and λ_4 → K_opt = 4."""
    segmenter = _make_segmenter()
    # gaps: [2, 1, 6, 0.1, 0.1, 0.1]  → biggest gap at 1-based index 3 → K_opt=4
    eigenvalues = np.array([10.0, 8.0, 7.0, 1.0, 0.9, 0.8, 0.7], dtype=np.float32)
    k = segmenter._eigengap_k(eigenvalues)
    assert k == 4


def test_eigengap_k_monotone():
    """Monotone decay still returns a valid int."""
    segmenter = _make_segmenter()
    eigenvalues = np.linspace(10.0, 1.0, 20).astype(np.float32)
    k = segmenter._eigengap_k(eigenvalues)
    assert isinstance(k, int)
    assert 1 <= k <= len(eigenvalues)

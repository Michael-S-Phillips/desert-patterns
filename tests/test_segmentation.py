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


def test_segmentation_config_new_field_defaults():
    from src.segmentation.segment import SegmentationConfig
    cfg = SegmentationConfig()
    assert cfg.attention_mode == "classifier"
    assert cfg.prompt_strategy == "topk"
    assert cfg.sam_version == "sam1"
    assert cfg.sam2_checkpoint == ""
    assert cfg.sam2_model_cfg == "sam2_hiera_b+.yaml"


def test_load_segmentation_config_new_fields():
    from src.segmentation.segment import load_segmentation_config
    cfg = load_segmentation_config({
        "segmentation": {
            "attention_mode": "self_attention",
            "prompt_strategy": "fps",
            "sam_version": "sam2",
            "sam2_checkpoint": "/tmp/sam2.pt",
            "sam2_model_cfg": "sam2_hiera_l.yaml",
        }
    })
    assert cfg.attention_mode == "self_attention"
    assert cfg.prompt_strategy == "fps"
    assert cfg.sam_version == "sam2"
    assert cfg.sam2_checkpoint == "/tmp/sam2.pt"
    assert cfg.sam2_model_cfg == "sam2_hiera_l.yaml"


def test_load_segmentation_config_new_field_defaults():
    """load_segmentation_config({}) returns correct defaults for all 5 new fields."""
    from src.segmentation.segment import SegmentationConfig, load_segmentation_config
    cfg = load_segmentation_config({})
    assert cfg.attention_mode == "classifier"
    assert cfg.prompt_strategy == "topk"
    assert cfg.sam_version == "sam1"
    assert cfg.sam2_checkpoint == ""
    assert cfg.sam2_model_cfg == "sam2_hiera_b+.yaml"


def test_segmenter_no_classifier_self_attention_ok():
    """Constructing with classifier_model=None is fine in self_attention mode."""
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    cfg = SegmentationConfig(attention_mode="self_attention")
    PatternSegmenter(None, cfg, DinoConfig())  # should not raise


def test_segmenter_no_classifier_raises_on_segment():
    """segment() raises ValueError when classifier mode but no classifier."""
    from pathlib import Path
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    cfg = SegmentationConfig(attention_mode="classifier")
    segmenter = PatternSegmenter(None, cfg, DinoConfig())
    with pytest.raises(ValueError, match="classifier_model"):
        segmenter.segment(Path("dummy.jpg"), "mudcrack")


# ---------------------------------------------------------------------------
# extract_attention_maps tests
# ---------------------------------------------------------------------------


def test_extract_attention_maps_shape():
    """Output shape is (n_heads, n_patches) with n_heads=12 for ViT-B."""
    pytest.importorskip("torch")
    from PIL import Image as PILImage
    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    extractor = DinoFeatureExtractor(DinoConfig())
    img = PILImage.fromarray(np.zeros((518, 518, 3), dtype=np.uint8))
    attn = extractor.extract_attention_maps(img)
    assert attn.ndim == 2
    n_heads, n_patches = attn.shape
    assert n_heads == 12        # ViT-B has 12 heads
    assert n_patches > 0        # read dynamically; empirically 1369 (37x37) at 518px


def test_extract_attention_maps_sums_to_one():
    """Each head's attention sums to ~1.0 after per-head re-normalization."""
    pytest.importorskip("torch")
    from PIL import Image as PILImage
    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    extractor = DinoFeatureExtractor(DinoConfig())
    img = PILImage.fromarray(
        np.random.default_rng(0).integers(0, 255, (518, 518, 3), dtype=np.uint8)
    )
    attn = extractor.extract_attention_maps(img)
    row_sums = attn.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_extract_all_layer_attentions_shape():
    """extract_all_layer_attentions returns (n_layers, n_heads, n_patches)."""
    pytest.importorskip("torch")
    from PIL import Image as PILImage
    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    extractor = DinoFeatureExtractor(DinoConfig())
    img = PILImage.fromarray(np.zeros((518, 518, 3), dtype=np.uint8))
    result = extractor.extract_all_layer_attentions(img)
    assert result.ndim == 3
    n_layers, n_heads, n_patches = result.shape
    assert n_layers == 12
    assert n_heads == 12
    assert n_patches > 0


def test_extract_all_layer_attentions_sums_to_one():
    """Each [layer, head] row of extract_all_layer_attentions sums to 1.0 or 0.0.

    Early-layer heads sometimes concentrate all attention on CLS/register tokens,
    leaving the patch slice all-zero. Those rows sum to 0; the rest sum to ~1.
    """
    pytest.importorskip("torch")
    from PIL import Image as PILImage
    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    extractor = DinoFeatureExtractor(DinoConfig())
    img = PILImage.fromarray(np.zeros((518, 518, 3), dtype=np.uint8))
    result = extractor.extract_all_layer_attentions(img)
    row_sums = result.sum(axis=-1)  # (n_layers, n_heads)
    # Each row must sum to either ~1.0 (normalized patch attention) or ~0.0 (no patch attention)
    assert np.all((np.abs(row_sums - 1.0) < 1e-4) | (row_sums < 1e-4))


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


# ---------------------------------------------------------------------------
# PatternSegmenter init + SAM (torch-optional)
# ---------------------------------------------------------------------------


def test_pattern_segmenter_lazy_init():
    """PatternSegmenter constructs without loading torch, DINOv3, or SAM."""
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig

    segmenter = PatternSegmenter(MagicMock(), SegmentationConfig(), DinoConfig())
    assert segmenter._extractor is None
    assert segmenter._sam_predictor is None


def test_run_sam_returns_lists():
    """_run_sam returns (list, list) even when all masks are below min area."""
    pytest.importorskip("torch")

    from unittest.mock import patch
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig

    segmenter = PatternSegmenter(MagicMock(), SegmentationConfig(), DinoConfig())

    # Mock SAM predictor to return tiny masks (below min_mask_area_fraction)
    mock_predictor = MagicMock()
    tiny_mask = np.zeros((100, 100), dtype=bool)
    mock_predictor.predict.return_value = (
        np.array([tiny_mask, tiny_mask, tiny_mask]),
        np.array([0.9, 0.8, 0.7]),
        None,
    )
    segmenter._sam_predictor = mock_predictor

    attn = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    masks, scores = segmenter._run_sam(image_np, attn, n_patches=196)

    assert isinstance(masks, list)
    assert isinstance(scores, list)


# ---------------------------------------------------------------------------
# Head selection + _compute_self_attention tests
# ---------------------------------------------------------------------------


def test_select_attention_head_max_entropy():
    """Head with uniform distribution (max entropy) is selected."""
    segmenter = _make_segmenter()
    n_heads, n_patches = 3, 16
    attn_maps = np.zeros((n_heads, n_patches), dtype=np.float32)
    attn_maps[0, 0] = 1.0                       # head 0: all mass on patch 0 (min entropy)
    attn_maps[1, :] = 1.0 / n_patches           # head 1: uniform (max entropy)
    attn_maps[2, :4] = 0.25                     # head 2: semi-uniform (mid entropy)
    head = segmenter._select_attention_head(attn_maps)
    assert head == 1


def test_select_attention_head_valid_index():
    """Always returns a valid head index for any input."""
    segmenter = _make_segmenter()
    rng = np.random.default_rng(42)
    attn_maps = rng.random((12, 1369)).astype(np.float32)
    attn_maps /= attn_maps.sum(axis=-1, keepdims=True)
    head = segmenter._select_attention_head(attn_maps)
    assert 0 <= head < 12


def test_compute_self_attention_shape():
    """Output attention map matches image (H, W)."""
    segmenter = _make_segmenter()
    n_patches = 16
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = (
        np.full((2, n_patches), 1.0 / n_patches, dtype=np.float32)
    )
    segmenter._extractor = mock_extractor

    from PIL import Image as PILImage
    dummy = PILImage.fromarray(np.zeros((56, 56, 3), dtype=np.uint8))
    attn_out, n_patches_out = segmenter._compute_self_attention(dummy, (56, 56))
    assert attn_out.shape == (56, 56)
    assert n_patches_out == n_patches


def test_compute_self_attention_range():
    """Output values are in [0, 1]."""
    segmenter = _make_segmenter()
    rng = np.random.default_rng(7)
    n_patches = 25
    attn = rng.random((3, n_patches)).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = attn
    segmenter._extractor = mock_extractor

    from PIL import Image as PILImage
    dummy = PILImage.fromarray(np.zeros((100, 120, 3), dtype=np.uint8))
    attn_out, _ = segmenter._compute_self_attention(dummy, (120, 100))
    assert attn_out.min() >= 0.0
    assert attn_out.max() <= 1.0 + 1e-6
    assert attn_out.dtype == np.float32


def test_compute_self_attention_returns_n_patches():
    """Returned n_patches equals the patch dimension of the attention maps."""
    segmenter = _make_segmenter()
    n_patches = 36
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = (
        np.ones((4, n_patches), dtype=np.float32) / n_patches
    )
    segmenter._extractor = mock_extractor

    from PIL import Image as PILImage
    dummy = PILImage.fromarray(np.zeros((84, 84, 3), dtype=np.uint8))
    _, returned_n = segmenter._compute_self_attention(dummy, (84, 84))
    assert returned_n == n_patches


# ---------------------------------------------------------------------------
# _farthest_point_sample tests
# ---------------------------------------------------------------------------


def test_fps_returns_correct_count():
    segmenter = _make_segmenter()
    candidates = np.arange(25, dtype=int)  # 25 patch indices
    selected = segmenter._farthest_point_sample(candidates, grid_size=5, n_points=4)
    assert len(selected) == 4


def test_fps_no_duplicates():
    segmenter = _make_segmenter()
    candidates = np.arange(20, dtype=int)
    selected = segmenter._farthest_point_sample(candidates, grid_size=5, n_points=5)
    assert len(selected) == len(set(selected.tolist()))


def test_fps_fewer_candidates_than_n_points():
    """Returns all candidates when len(candidates) <= n_points."""
    segmenter = _make_segmenter()
    candidates = np.array([0, 1, 2], dtype=int)
    selected = segmenter._farthest_point_sample(candidates, grid_size=5, n_points=6)
    assert set(selected.tolist()) == {0, 1, 2}


def test_fps_more_spread_than_topk():
    """FPS selects points more spatially spread than top-K from a clustered set."""
    segmenter = _make_segmenter()
    grid_size = 10
    # 6 candidates in top-left, 4 candidates in bottom-right
    topleft = [0, 1, 10, 11, 20, 21]    # rows 0-2, cols 0-1
    botright = [88, 89, 98, 99]          # rows 8-9, cols 8-9
    candidates = np.array(topleft + botright, dtype=int)

    fps_sel = segmenter._farthest_point_sample(candidates, grid_size, n_points=3)
    topk_sel = candidates[:3]  # first 3 = all top-left cluster

    def mean_pairwise_dist(idxs: np.ndarray) -> float:
        rows = idxs // grid_size
        cols = idxs % grid_size
        coords = np.stack([rows, cols], axis=1).astype(float)
        dists = [
            np.linalg.norm(coords[i] - coords[j])
            for i in range(len(coords))
            for j in range(i + 1, len(coords))
        ]
        return float(np.mean(dists))

    assert mean_pairwise_dist(fps_sel) > mean_pairwise_dist(topk_sel)


# ---------------------------------------------------------------------------
# _patch_centroids FPS branch tests
# ---------------------------------------------------------------------------


def _make_segmenter_fps():
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    cfg = SegmentationConfig(prompt_strategy="fps")
    return PatternSegmenter(MagicMock(), cfg, DinoConfig())


def test_patch_centroids_fps_correct_count():
    """FPS branch still returns n_foreground_prompts and n_background_prompts points."""
    segmenter = _make_segmenter_fps()
    attn = np.random.default_rng(5).random((100, 100)).astype(np.float32)
    fg, bg = segmenter._patch_centroids(attn, n_patches=196)
    assert fg.shape == (5, 2)
    assert bg.shape == (5, 2)


def test_patch_centroids_fps_in_bounds():
    """FPS-selected centroids are within the image bounds."""
    segmenter = _make_segmenter_fps()
    attn = np.random.default_rng(6).random((200, 300)).astype(np.float32)
    fg, bg = segmenter._patch_centroids(attn, n_patches=196)
    assert fg[:, 0].max() <= 300  # x ≤ width
    assert fg[:, 1].max() <= 200  # y ≤ height
    assert fg.min() >= 0


# ---------------------------------------------------------------------------
# SAM 2 predictor test
# ---------------------------------------------------------------------------


def test_get_sam2_predictor_type():
    """_get_sam_predictor returns SAM2ImagePredictor when sam_version='sam2'."""
    pytest.importorskip("sam2")
    from pathlib import Path
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig

    checkpoint = "/Volumes/Fangorn/Software/sam2/sam2_hiera_base_plus.pt"
    if not Path(checkpoint).exists():
        pytest.skip(f"SAM 2 checkpoint not found at {checkpoint}")

    cfg = SegmentationConfig(
        sam_version="sam2",
        sam2_checkpoint=checkpoint,
        sam2_model_cfg="sam2_hiera_b+.yaml",
    )
    segmenter = PatternSegmenter(MagicMock(), cfg, DinoConfig())
    predictor = segmenter._get_sam_predictor()
    assert isinstance(predictor, SAM2ImagePredictor)


# ---------------------------------------------------------------------------
# segment() self_attention path (mocked)
# ---------------------------------------------------------------------------


def test_segment_self_attention_path():
    """segment() works with attention_mode='self_attention' and mocked extractors."""
    pytest.importorskip("torch")
    from pathlib import Path
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig

    cfg = SegmentationConfig(attention_mode="self_attention")
    segmenter = PatternSegmenter(None, cfg, DinoConfig())

    # Mock attention extraction
    h, w = 100, 100
    n_patches = 16
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = (
        np.full((2, n_patches), 1.0 / n_patches, dtype=np.float32)
    )
    segmenter._extractor = mock_extractor

    # Mock SAM predictor to return one valid mask
    mock_predictor = MagicMock()
    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 20:60] = True
    mock_predictor.predict.return_value = (
        np.array([mask, mask, mask]),
        np.array([0.9, 0.8, 0.7]),
        None,
    )
    segmenter._sam_predictor = mock_predictor

    import tempfile
    from PIL import Image as PILImage
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
    PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(tmp_path)

    try:
        result = segmenter.segment(tmp_path, "mudcrack")
        assert result.attention_map.shape == (h, w)
        assert isinstance(result.instance_masks, list)
    finally:
        tmp_path.unlink(missing_ok=True)

# Desert Pattern Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch-segment 367 labeled desert pattern images using DINOv3 patch attention maps as SAM prompts, producing per-image instance mask overlays, binary attention masks, metadata JSON, and per-class gallery figures.

**Architecture:** `src/segmentation/segment.py` holds the core `PatternSegmenter` class (attention computation, SAM prompting, IoU deduplication). `scripts/generate_segmentations.py` is a standalone batch script that loads the existing classifier model and embeddings, runs `PatternSegmenter` per image, saves outputs, and generates gallery figures.

**Tech Stack:** DINOv3 patch tokens (via `DinoFeatureExtractor.extract_patch_tokens`), SAM ViT-B (via `segment_anything`), `scipy.ndimage.gaussian_filter`, `cv2` (Otsu threshold + contour drawing), `PIL`, `numpy`, `matplotlib`, `joblib`.

**Spec:** `docs/superpowers/specs/2026-03-27-segmentation-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/segmentation/__init__.py` | Create | Package marker |
| `src/segmentation/segment.py` | Create | `SegmentationConfig`, `SegmentationResult`, `load_segmentation_config`, `PatternSegmenter` |
| `configs/classifier_config.yaml` | Modify | Add `segmentation:` block |
| `tests/test_segmentation.py` | Create | Unit tests (no SAM/torch required for most) |
| `scripts/generate_segmentations.py` | Create | Batch processing script + gallery figures |

---

## Task 1: Package setup, config, and dataclasses

**Files:**
- Create: `src/segmentation/__init__.py`
- Create: `src/segmentation/segment.py` (config/dataclasses/loader only)
- Modify: `configs/classifier_config.yaml`
- Create: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing config tests**

```python
# tests/test_segmentation.py
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
source .venv/bin/activate
pytest tests/test_segmentation.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `src.segmentation.segment` doesn't exist yet.

- [ ] **Step 3: Create package marker**

```python
# src/segmentation/__init__.py
"""Desert pattern segmentation using DINOv3 attention maps and SAM prompts."""
```

- [ ] **Step 4: Create segment.py with config/dataclasses/loader**

```python
# src/segmentation/segment.py
"""Pattern segmentation: DINOv3 patch attention maps → SAM instance masks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SegmentationConfig:
    """Configuration for DINOv3-attention + SAM segmentation."""

    sam_checkpoint: str = "/Volumes/Rohan/Software/sam/sam_vit_b_01ec64.pth"
    output_dir: str = "outputs/segmentations"
    n_foreground_prompts: int = 5
    n_background_prompts: int = 5
    threshold_method: str = "otsu"           # "otsu" | "percentile"
    attention_percentile: float = 0.70
    min_mask_area_fraction: float = 0.005
    iou_dedup_threshold: float = 0.5


def load_segmentation_config(config_dict: dict) -> SegmentationConfig:
    """Load SegmentationConfig from the full parsed YAML config dict.

    Reads the ``segmentation:`` sub-dict internally, same pattern as
    ``load_classifier_config()``.

    Args:
        config_dict: Full parsed YAML dictionary.

    Returns:
        Populated SegmentationConfig.
    """
    seg = config_dict.get("segmentation", {})
    return SegmentationConfig(
        sam_checkpoint=seg.get("sam_checkpoint", SegmentationConfig.sam_checkpoint),
        output_dir=seg.get("output_dir", SegmentationConfig.output_dir),
        n_foreground_prompts=seg.get("n_foreground_prompts", SegmentationConfig.n_foreground_prompts),
        n_background_prompts=seg.get("n_background_prompts", SegmentationConfig.n_background_prompts),
        threshold_method=seg.get("threshold_method", SegmentationConfig.threshold_method),
        attention_percentile=seg.get("attention_percentile", SegmentationConfig.attention_percentile),
        min_mask_area_fraction=seg.get("min_mask_area_fraction", SegmentationConfig.min_mask_area_fraction),
        iou_dedup_threshold=seg.get("iou_dedup_threshold", SegmentationConfig.iou_dedup_threshold),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """Output of segmenting a single image."""

    attention_map: np.ndarray        # float32, shape (H, W), values in [0, 1]
    binary_mask: np.ndarray          # uint8, shape (H, W), values 0 or 255
    instance_masks: list[np.ndarray] # each bool array shape (H, W)
    iou_scores: list[float]          # one per instance mask (SAM predicted IoU)
    image_path: Path
    class_name: str
```

- [ ] **Step 5: Add segmentation block to classifier_config.yaml**

Open `configs/classifier_config.yaml` and append at the end:

```yaml

segmentation:
  sam_checkpoint: /Volumes/Rohan/Software/sam/sam_vit_b_01ec64.pth
  output_dir: outputs/segmentations
  n_foreground_prompts: 5
  n_background_prompts: 5
  threshold_method: otsu          # otsu | percentile
  attention_percentile: 0.70      # used only when threshold_method=percentile
  min_mask_area_fraction: 0.005
  iou_dedup_threshold: 0.5
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_segmentation.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/segmentation/__init__.py src/segmentation/segment.py \
        configs/classifier_config.yaml tests/test_segmentation.py
git commit -m "feat: add segmentation package skeleton with config and dataclasses"
```

---

## Task 2: Attention map computation and thresholding

**Files:**
- Modify: `src/segmentation/segment.py` (add `PatternSegmenter` with `_compute_attention`, `_threshold`)
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing attention + threshold tests**

Add to `tests/test_segmentation.py`:

```python
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
    """Works for non-196 patch counts (10×10 = 100 patches)."""
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_segmentation.py::test_compute_attention_output_shape -v
```

Expected: `AttributeError: 'PatternSegmenter' object has no attribute '_compute_attention'` (or similar — `PatternSegmenter` doesn't exist yet).

- [ ] **Step 3: Add PatternSegmenter with _compute_attention and _threshold to segment.py**

Add after the `SegmentationResult` dataclass:

```python
# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class PatternSegmenter:
    """Segment desert pattern images using DINOv3 attention maps as SAM prompts.

    DINOv3 and SAM are both lazy-loaded on first use.

    Args:
        classifier_model: Fitted ``sklearn.linear_model.LogisticRegression``.
        seg_config: Segmentation configuration.
        dino_config: DINOv3 configuration (from ``load_dino_config()``).
    """

    def __init__(
        self,
        classifier_model: Any,
        seg_config: SegmentationConfig,
        dino_config: Any,
    ) -> None:
        self._model = classifier_model
        self._seg_config = seg_config
        self._dino_config = dino_config
        self._extractor: Any = None
        self._sam_predictor: Any = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            from src.features.dino_embeddings import DinoFeatureExtractor
            self._extractor = DinoFeatureExtractor(self._dino_config)
        return self._extractor

    def _get_sam_predictor(self) -> Any:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=self._seg_config.sam_checkpoint)
        sam.to(device)
        self._sam_predictor = SamPredictor(sam)
        logger.info("SAM loaded on device=%s", device)
        return self._sam_predictor

    # ------------------------------------------------------------------
    # Attention map
    # ------------------------------------------------------------------

    def _compute_attention(
        self,
        patch_tokens: np.ndarray,
        cls_coef: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Compute spatial attention map from patch tokens × LR coefficient.

        Args:
            patch_tokens: shape (n_patches, 768)
            cls_coef: shape (768,) — LR weight vector for the image's class
            image_size: (width, height) in PIL convention

        Returns:
            float32 attention map of shape (height, width), values in [0, 1]
        """
        from scipy.ndimage import gaussian_filter

        n_patches = patch_tokens.shape[0]
        grid_size = int(round(np.sqrt(n_patches)))

        activations = (patch_tokens @ cls_coef)[: grid_size * grid_size]
        spatial = activations.reshape(grid_size, grid_size).astype(np.float32)

        # Normalize grid to [0, 1]
        vmin, vmax = spatial.min(), spatial.max()
        spatial_norm = (
            (spatial - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(spatial)
        )

        # Upsample to native image size: PIL.Image.resize takes (width, height)
        width, height = image_size
        heat_pil = Image.fromarray((spatial_norm * 255).astype(np.uint8), mode="L")
        heat_up = (
            np.asarray(heat_pil.resize((width, height), Image.BILINEAR), dtype=np.float32)
            / 255.0
        )

        # Gaussian smoothing then re-normalize
        heat_smooth = gaussian_filter(heat_up, sigma=2).astype(np.float32)
        smin, smax = heat_smooth.min(), heat_smooth.max()
        return (
            (heat_smooth - smin) / (smax - smin) if smax > smin else heat_smooth
        )

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def _threshold(self, attention_map: np.ndarray) -> np.ndarray:
        """Convert soft attention map to binary mask (0 / 255 uint8).

        Uses Otsu or fixed percentile depending on ``seg_config.threshold_method``.
        """
        uint8_map = (attention_map * 255).astype(np.uint8)

        if self._seg_config.threshold_method == "otsu":
            _, binary = cv2.threshold(
                uint8_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # "percentile"
            thresh_val = float(
                np.quantile(attention_map, self._seg_config.attention_percentile)
            )
            binary = ((attention_map >= thresh_val) * 255).astype(np.uint8)

        return binary
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_segmentation.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/segment.py tests/test_segmentation.py
git commit -m "feat: add PatternSegmenter with attention map and thresholding"
```

---

## Task 3: Patch centroids, IoU, and mask deduplication

**Files:**
- Modify: `src/segmentation/segment.py`
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_segmentation.py`:

```python
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
    """Patch 0 (top-left grid cell) should appear in bg when attention is 0 there."""
    segmenter = _make_segmenter()
    # Make entire attention = 1 except top-left block which is 0
    attn = np.ones((140, 140), dtype=np.float32)
    attn[:10, :10] = 0.0  # top-left patch region lowest
    _, bg = segmenter._patch_centroids(attn, n_patches=196)
    # First bg point should be near (5, 5) — center of top-left patch
    # Each patch is 140/14 = 10px wide; center of patch 0 = (5, 5)
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
    b[2:6, 2:6] = True  # 16 px, overlap = 2×2 = 4 px → union = 28
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
    base[10:50, 10:50] = True  # 40×40 = 1600 px

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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_segmentation.py::test_patch_centroids_shape -v
```

Expected: `AttributeError` — `_patch_centroids` not yet defined.

- [ ] **Step 3: Add _patch_centroids, _iou, _deduplicate_masks to PatternSegmenter**

Add these methods inside the `PatternSegmenter` class in `src/segmentation/segment.py`, after `_threshold`:

```python
    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _patch_centroids(
        self,
        attention_map: np.ndarray,
        n_patches: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map top/bottom attention patches to pixel coordinates for SAM prompts.

        Args:
            attention_map: float32, shape (H, W) — native-resolution attention map
            n_patches: total number of patches (derives grid_size dynamically)

        Returns:
            (fg_points, bg_points) each shape (k, 2) as [[x, y], ...] float32
        """
        grid_size = int(round(np.sqrt(n_patches)))
        height, width = attention_map.shape

        # Score each grid cell by its mean attention in the upsampled map
        patch_activations = np.array(
            [
                attention_map[
                    int(row / grid_size * height) : int((row + 1) / grid_size * height),
                    int(col / grid_size * width) : int((col + 1) / grid_size * width),
                ].mean()
                for row in range(grid_size)
                for col in range(grid_size)
            ],
            dtype=np.float32,
        )

        n_fg = min(self._seg_config.n_foreground_prompts, n_patches)
        n_bg = min(self._seg_config.n_background_prompts, n_patches)

        sorted_idx = np.argsort(patch_activations)
        fg_indices = sorted_idx[-n_fg:][::-1]  # highest activation first
        bg_indices = sorted_idx[:n_bg]          # lowest activation first

        def to_pixel(patch_idx: int) -> list[float]:
            row, col = divmod(int(patch_idx), grid_size)
            cx = (col + 0.5) / grid_size * width
            cy = (row + 0.5) / grid_size * height
            return [cx, cy]

        fg_points = np.array([to_pixel(i) for i in fg_indices], dtype=np.float32)
        bg_points = np.array([to_pixel(i) for i in bg_indices], dtype=np.float32)
        return fg_points, bg_points

    # ------------------------------------------------------------------
    # IoU and deduplication
    # ------------------------------------------------------------------

    def _iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Intersection-over-Union between two boolean masks."""
        intersection = int((mask_a & mask_b).sum())
        union = int((mask_a | mask_b).sum())
        return float(intersection) / float(union) if union > 0 else 0.0

    def _deduplicate_masks(
        self,
        masks: list[np.ndarray],
        scores: list[float],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Remove masks that overlap too much with a higher-scoring mask.

        Processes masks in descending score order; discards any mask whose IoU
        with an already-accepted mask exceeds ``iou_dedup_threshold``.
        """
        if not masks:
            return [], []

        order = np.argsort(scores)[::-1]
        kept_masks: list[np.ndarray] = []
        kept_scores: list[float] = []

        for idx in order:
            mask = masks[idx]
            if all(
                self._iou(mask, kept) < self._seg_config.iou_dedup_threshold
                for kept in kept_masks
            ):
                kept_masks.append(mask)
                kept_scores.append(scores[idx])

        return kept_masks, kept_scores
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_segmentation.py -v
```

Expected: all 19 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/segment.py tests/test_segmentation.py
git commit -m "feat: add patch centroid mapping, IoU, and mask deduplication"
```

---

## Task 4: SAM integration and segment()

**Files:**
- Modify: `src/segmentation/segment.py` (add `_run_sam`, `segment`)
- Modify: `tests/test_segmentation.py`

- [ ] **Step 1: Write tests**

Add to `tests/test_segmentation.py`:

```python
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
    """_run_sam returns (list, list) even if SAM produces no valid masks."""
    pytest.importorskip("torch")

    import numpy as np
    from unittest.mock import MagicMock, patch
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig

    segmenter = PatternSegmenter(MagicMock(), SegmentationConfig(), DinoConfig())

    # Mock SAM predictor to return empty/tiny masks (below min_mask_area_fraction)
    mock_predictor = MagicMock()
    tiny_mask = np.zeros((100, 100), dtype=bool)  # 0 px — below threshold
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_segmentation.py::test_pattern_segmenter_lazy_init -v
```

Expected: PASS (the `__init__` is already implemented). Then:

```bash
pytest tests/test_segmentation.py::test_run_sam_returns_lists -v
```

Expected: `AttributeError: 'PatternSegmenter' object has no attribute '_run_sam'`

- [ ] **Step 3: Add _run_sam and segment() to PatternSegmenter**

Add after `_deduplicate_masks` in `src/segmentation/segment.py`:

```python
    # ------------------------------------------------------------------
    # SAM
    # ------------------------------------------------------------------

    def _run_sam(
        self,
        image_np: np.ndarray,
        attention_map: np.ndarray,
        n_patches: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run SAM with attention-derived prompts; return deduped masks + IoU scores.

        For each foreground prompt point, runs ``SamPredictor.predict()``
        with that point plus all background points.  Selects the candidate
        mask with the highest predicted IoU, filters by minimum area, then
        deduplicates across prompts.

        Args:
            image_np: uint8 RGB array of shape (H, W, 3) at native resolution.
            attention_map: float32, shape (H, W) — already at native resolution.
            n_patches: number of patch tokens (used to derive grid_size).

        Returns:
            (instance_masks, iou_scores) — parallel lists after deduplication.
        """
        predictor = self._get_sam_predictor()
        predictor.set_image(image_np)

        fg_points, bg_points = self._patch_centroids(attention_map, n_patches)
        h, w = image_np.shape[:2]
        min_area = self._seg_config.min_mask_area_fraction * h * w

        all_masks: list[np.ndarray] = []
        all_scores: list[float] = []

        for fg_pt in fg_points:
            # Combine this fg point with all bg points
            point_coords = np.vstack([fg_pt[np.newaxis, :], bg_points])  # (1+n_bg, 2)
            point_labels = np.array([1] + [0] * len(bg_points), dtype=np.int32)

            masks, iou_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            # Select the mask with the highest predicted IoU
            best_idx = int(np.argmax(iou_scores))
            mask = masks[best_idx].astype(bool)

            if mask.sum() >= min_area:
                all_masks.append(mask)
                all_scores.append(float(iou_scores[best_idx]))

        if not all_masks:
            return [], []

        return self._deduplicate_masks(all_masks, all_scores)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment(self, image_path: Path, class_name: str) -> SegmentationResult:
        """Run full segmentation pipeline for a single image.

        Loads the image at native resolution, extracts DINOv3 patch tokens,
        computes the attention map, thresholds it, and runs SAM with the
        attention-derived prompt points.

        Args:
            image_path: Path to the image file (JPEG or PNG).
            class_name: The image's true class label (e.g. "mudcrack").
                        Must be present in ``classifier_model.classes_``.

        Returns:
            SegmentationResult with attention map, binary mask, and instance masks.
        """
        class_idx = list(self._model.classes_).index(class_name)
        cls_coef = self._model.coef_[class_idx]  # (768,)

        pil_img = Image.open(image_path).convert("RGB")
        image_size = pil_img.size  # (width, height) in PIL convention
        image_np = np.asarray(pil_img)  # (H, W, 3) uint8 for SAM

        patch_tokens = self._get_extractor().extract_patch_tokens(pil_img)
        n_patches = patch_tokens.shape[0]

        attention_map = self._compute_attention(patch_tokens, cls_coef, image_size)
        binary_mask = self._threshold(attention_map)
        instance_masks, iou_scores = self._run_sam(image_np, attention_map, n_patches)

        return SegmentationResult(
            attention_map=attention_map,
            binary_mask=binary_mask,
            instance_masks=instance_masks,
            iou_scores=iou_scores,
            image_path=image_path,
            class_name=class_name,
        )
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_segmentation.py -v
```

Expected: all 21 tests PASS (the `test_run_sam_returns_lists` test will be skipped if torch is not installed).

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/segment.py tests/test_segmentation.py
git commit -m "feat: add _run_sam and segment() completing PatternSegmenter"
```

---

## Task 5: Batch script — generate_segmentations.py

**Files:**
- Create: `scripts/generate_segmentations.py`

This task has no new automated tests (visual outputs verified manually). After implementing, run on a few images to verify.

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python
"""Batch segmentation of desert pattern images using DINOv3 attention + SAM.

For each labeled image in checked-images/, computes a DINOv3 patch attention
map, thresholds it to a binary mask, and prompts SAM to generate instance masks.
Saves per-image overlays, binary masks, attention maps, and metadata JSON.
Also generates per-class gallery figures of the top-12 most-confident examples.

Usage:
    python scripts/generate_segmentations.py [--force] [--verbose]
    python scripts/generate_segmentations.py --image IMG_8346.jpeg IMG_8350.jpeg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _colorize_attention(attention_map: np.ndarray) -> np.ndarray:
    """Render float32 [0,1] attention map as RGB uint8 using the 'hot' colormap."""
    from matplotlib import colormaps
    rgba = (colormaps["viridis"](attention_map) * 255).astype(np.uint8)
    return rgba[:, :, :3]


def make_overlay(image_np: np.ndarray, result) -> np.ndarray:
    """Composite original image with attention heatmap and instance mask overlays.

    Blends:
    - Attention heatmap at 30% alpha (underneath masks)
    - Instance masks at 50% alpha (Wong palette)
    - Instance boundary outlines (2px, same color)

    Args:
        image_np: uint8 RGB array, shape (H, W, 3)
        result: SegmentationResult

    Returns:
        uint8 RGB composite, same shape as image_np.
    """
    from src.visualization.style import WONG_PALETTE

    out = image_np.astype(np.float32) / 255.0

    # Attention heatmap at 30% alpha
    heat_rgb = _colorize_attention(result.attention_map).astype(np.float32) / 255.0
    out = out * 0.7 + heat_rgb * 0.3

    # Instance masks at 50% alpha
    for i, mask in enumerate(result.instance_masks):
        r, g, b = _hex_to_rgb(WONG_PALETTE[i % len(WONG_PALETTE)])
        color = np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
        out[mask] = out[mask] * 0.5 + color * 0.5

    out_uint8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)

    # Boundary outlines — convert RGB→BGR for OpenCV, draw, convert back
    out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
    for i, mask in enumerate(result.instance_masks):
        r, g, b = _hex_to_rgb(WONG_PALETTE[i % len(WONG_PALETTE)])
        bgr_color = (b, g, r)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out_bgr, contours, -1, bgr_color, 2)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def save_segmentation_outputs(result, class_dir: Path) -> None:
    """Write attention PNG, binary mask PNG, overlay PNG, and masks JSON.

    Args:
        result: SegmentationResult
        class_dir: Directory for this image's class (e.g. outputs/segmentations/mudcrack/)
    """
    stem = result.image_path.stem
    image_np = np.asarray(Image.open(result.image_path).convert("RGB"))

    # Soft attention heatmap blended with original (50/50)
    heat_rgb = _colorize_attention(result.attention_map)
    attention_overlay = (image_np * 0.5 + heat_rgb * 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(attention_overlay).save(class_dir / f"{stem}_attention.png")

    # Binary mask (0/255)
    Image.fromarray(result.binary_mask).save(class_dir / f"{stem}_mask.png")

    # Full composite overlay
    overlay = make_overlay(image_np, result)
    Image.fromarray(overlay).save(class_dir / f"{stem}_overlay.png")

    # Instance metadata JSON
    masks_data = []
    for mask, iou_score in zip(result.instance_masks, result.iou_scores):
        if mask.any():
            ys, xs = np.where(mask)
            masks_data.append({
                "area_px": int(mask.sum()),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
                "iou_score": float(iou_score),
            })

    with open(class_dir / f"{stem}_masks.json", "w") as f:
        json.dump(masks_data, f, indent=2)

    logger.info("Saved %s — %d instance(s)", stem, len(masks_data))


# ---------------------------------------------------------------------------
# Gallery figure
# ---------------------------------------------------------------------------


def make_gallery_figure(
    class_name: str,
    seg_output_dir: Path,
    model,
    X: np.ndarray,
    y: list[str],
    image_list: list,
    figures_dir: Path,
    style,
) -> None:
    """3×4 gallery of top-12 confidence images showing overlay thumbnails."""
    import matplotlib.pyplot as plt
    from src.visualization.style import save_figure

    class_idx = list(model.classes_).index(class_name)
    proba = model.predict_proba(X)[:, class_idx]

    cls_mask = np.array([lbl == class_name for lbl in y])
    cls_indices = np.where(cls_mask)[0]
    cls_probs = proba[cls_indices]

    n_show = min(12, len(cls_indices))
    top_local = np.argsort(cls_probs)[::-1][:n_show]
    top_global = cls_indices[top_local]

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.6))
    class_dir = seg_output_dir / class_name

    for ax_idx, gi in enumerate(top_global):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        ax.axis("off")
        img_path, _ = image_list[gi]
        overlay_path = class_dir / f"{img_path.stem}_overlay.png"
        if overlay_path.exists():
            thumb = Image.open(overlay_path).convert("RGB").resize((224, 224), Image.LANCZOS)
            ax.imshow(np.asarray(thumb))
            ax.set_title(f"{cls_probs[top_local[ax_idx]]:.3f}", fontsize=7)
        else:
            ax.set_facecolor("#cccccc")

    # Hide unused grid cells
    for ax_idx in range(len(top_global), nrows * ncols):
        axes[ax_idx // ncols, ax_idx % ncols].set_visible(False)

    fig.suptitle(
        f"{class_name} — top-{n_show} by confidence (instance overlay)",
        fontsize=style.font_size_base,
    )
    fig.tight_layout()
    out = figures_dir / f"segmentation_gallery_{class_name}"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Saved gallery for %s", class_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-segment desert pattern images with DINOv3 attention + SAM."
    )
    parser.add_argument("--config", default="configs/classifier_config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if _overlay.png already exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--image", nargs="+", default=None,
        help="Process only these image files (filename or stem). Gallery skipped.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    from src.classification.train import load_classifier_config, scan_labeled_images
    from src.features.dino_embeddings import load_dino_config
    from src.segmentation.segment import PatternSegmenter, load_segmentation_config
    from src.visualization.style import FigureStyle, setup_matplotlib_style

    clf_config = load_classifier_config(config_dict)
    seg_config = load_segmentation_config(config_dict)
    dino_config = load_dino_config(config_dict.get("dino", {}))

    setup_matplotlib_style()
    style = FigureStyle()

    # Load classifier model and cached data
    model_dir = Path(clf_config.model_dir)
    model = joblib.load(model_dir / "lr_classifier.joblib")
    X = np.load(clf_config.embedding_cache)
    y = list(np.load(clf_config.label_cache, allow_pickle=True))

    # Scan images and validate alignment with label cache
    image_list = scan_labeled_images(Path(clf_config.image_dir), clf_config.label_map)
    if [lbl for _, lbl in image_list] != y:
        raise ValueError(
            "Image list and label cache are misaligned — "
            "re-run scripts/train_classifier.py to regenerate the cache."
        )

    # Filter to --image subset if specified
    if args.image:
        targets = set(args.image)
        image_list_to_process = [
            (p, lbl) for p, lbl in image_list
            if p.name in targets or p.stem in targets
        ]
        if not image_list_to_process:
            raise ValueError(
                f"No images matched --image {args.image!r}. "
                "Check filenames against the checked-images/ directory."
            )
        run_gallery = False
        logger.info("Processing %d specific image(s)", len(image_list_to_process))
    else:
        image_list_to_process = image_list
        run_gallery = True
        logger.info("Processing all %d images", len(image_list))

    # Create output directories
    seg_output_dir = Path(seg_config.output_dir)
    figures_dir = Path(clf_config.figures_dir)
    for class_name in model.classes_:
        (seg_output_dir / class_name).mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialise segmenter (DINOv3 and SAM load lazily on first image)
    segmenter = PatternSegmenter(model, seg_config, dino_config)

    # Process images
    for img_path, class_name in image_list_to_process:
        class_dir = seg_output_dir / class_name
        overlay_path = class_dir / f"{img_path.stem}_overlay.png"
        if overlay_path.exists() and not args.force:
            logger.debug("Skipping %s (overlay exists; use --force to re-run)", img_path.stem)
            continue

        logger.info("Segmenting %s (%s)", img_path.stem, class_name)
        try:
            result = segmenter.segment(img_path, class_name)
            save_segmentation_outputs(result, class_dir)
        except Exception as exc:
            logger.error("Failed to segment %s: %s", img_path.stem, exc, exc_info=True)

    # Gallery figures
    if run_gallery:
        logger.info("Generating per-class gallery figures")
        for class_name in model.classes_:
            make_gallery_figure(
                class_name, seg_output_dir, model, X, y, image_list, figures_dir, style
            )

    logger.info("Done — segmentation outputs at %s", seg_output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite to confirm nothing broken**

```bash
pytest tests/test_segmentation.py tests/test_classifier_train.py tests/test_classifier_predict.py -v
```

Expected: all tests PASS (21 segmentation + existing classifier tests).

- [ ] **Step 3: Smoke-test the script on one image per class (skips SAM checkpoint check)**

```bash
source .venv/bin/activate
python scripts/generate_segmentations.py \
  --image "IMG_8346.jpeg" \
  --verbose
```

This will load DINOv3 and SAM. Expected: one `_overlay.png`, `_attention.png`, `_mask.png`, `_masks.json` in `outputs/segmentations/big_pool/`.

If the SAM checkpoint is missing, you'll see: `FileNotFoundError: /Volumes/Rohan/Software/sam/sam_vit_b_01ec64.pth`. Verify the path exists first:

```bash
ls /Volumes/Rohan/Software/sam/sam_vit_b_01ec64.pth
```

- [ ] **Step 4: Inspect the outputs visually**

Open `outputs/segmentations/big_pool/IMG_8346_overlay.png` and verify:
- Attention heatmap visible underneath
- At least one colored instance mask visible
- Instance boundaries drawn as outlines

- [ ] **Step 5: Run on all images (takes ~5–15 min depending on hardware)**

```bash
python scripts/generate_segmentations.py --verbose
```

Expected at completion: `Done — segmentation outputs at outputs/segmentations` plus gallery figure saves logged.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_segmentations.py
git commit -m "feat: add generate_segmentations.py batch script with gallery figures"
```

---

## Final verification

- [ ] **Run the full test suite one last time**

```bash
pytest tests/test_segmentation.py -v
```

Expected: 21 tests PASS (SAM/torch-dependent tests skip if torch not installed).

- [ ] **Check outputs directory structure**

```bash
ls outputs/segmentations/mudcrack/ | head -10
ls outputs/segmentations/big_pool/ | head -10
ls outputs/segmentations/jbio/ | head -10
ls outputs/figures/segmentation_gallery_*.png
```

Expected: `_overlay.png`, `_attention.png`, `_mask.png`, `_masks.json` files for each image, plus 3 gallery PNGs.

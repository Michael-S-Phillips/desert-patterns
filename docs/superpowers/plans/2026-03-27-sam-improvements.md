# SAM Segmentation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-head DINOv3 self-attention, farthest-point prompt sampling, and SAM 2 support to the existing SAM segmentation pipeline, all gated by config flags with no breaking changes.

**Architecture:** Extend `PatternSegmenter` and `DinoFeatureExtractor` in-place using the same patterns already in the codebase (dataclass config, lazy loading, `pytest.importorskip` for optional deps). All four combinations of attention mode × prompt strategy work independently; SAM version is orthogonal.

**Tech Stack:** Python 3.10, numpy, PIL, scipy.ndimage, scikit-learn, transformers (DINOv3 attention), segment-anything (SAM 1), sam2 (SAM 2).

---

## File Map

- Modify: `src/features/dino_embeddings.py` — add `extract_attention_maps()`
- Modify: `src/segmentation/segment.py` — 5 new config fields, `load_segmentation_config` update, optional `classifier_model`, `_select_attention_head()`, `_compute_self_attention()`, `_farthest_point_sample()`, updated `_patch_centroids()`, updated `_get_sam_predictor()`, updated `segment()`
- Modify: `configs/classifier_config.yaml` — 5 new fields under `segmentation:`
- Modify: `pyproject.toml` — add `sam2` to `[ml]` extras
- Modify: `tests/test_segmentation.py` — new tests throughout

---

## Task 1: Config + optional `classifier_model`

**Files:**
- Modify: `src/segmentation/segment.py:24-60, 96-106`
- Modify: `configs/classifier_config.yaml`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_segmentation.py` after `test_load_segmentation_config_missing_section`:

```python
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


def test_segmenter_no_classifier_self_attention_ok():
    """Constructing with classifier_model=None is fine in self_attention mode."""
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    cfg = SegmentationConfig(attention_mode="self_attention")
    PatternSegmenter(None, cfg, DinoConfig())  # should not raise


def test_load_segmentation_config_new_field_defaults():
    """load_segmentation_config({}) returns correct defaults for all 5 new fields."""
    from src.segmentation.segment import SegmentationConfig, load_segmentation_config
    cfg = load_segmentation_config({})
    assert cfg.attention_mode == "classifier"
    assert cfg.prompt_strategy == "topk"
    assert cfg.sam_version == "sam1"
    assert cfg.sam2_checkpoint == ""
    assert cfg.sam2_model_cfg == "sam2_hiera_b+.yaml"


def test_segmenter_no_classifier_raises_on_segment():
    """segment() raises ValueError when classifier mode but no classifier."""
    from pathlib import Path
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    cfg = SegmentationConfig(attention_mode="classifier")
    segmenter = PatternSegmenter(None, cfg, DinoConfig())
    with pytest.raises(ValueError, match="classifier_model"):
        segmenter.segment(Path("dummy.jpg"), "mudcrack")
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Volumes/Fangorn/desert_patterns
source .venv/bin/activate
pytest tests/test_segmentation.py::test_segmentation_config_new_field_defaults tests/test_segmentation.py::test_load_segmentation_config_new_fields tests/test_segmentation.py::test_load_segmentation_config_new_field_defaults tests/test_segmentation.py::test_segmenter_no_classifier_self_attention_ok tests/test_segmentation.py::test_segmenter_no_classifier_raises_on_segment -v
```

Expected: FAIL (AttributeError or AssertionError — fields don't exist yet)

- [ ] **Step 3: Add 5 new fields to `SegmentationConfig` in `segment.py`**

In `src/segmentation/segment.py`, after line 35 (`iou_dedup_threshold: float = 0.5`), add:

```python
    attention_mode: str = "classifier"       # "classifier" | "self_attention"
    prompt_strategy: str = "topk"            # "topk" | "fps"
    sam_version: str = "sam1"               # "sam1" | "sam2"
    sam2_checkpoint: str = ""
    sam2_model_cfg: str = "sam2_hiera_b+.yaml"
```

- [ ] **Step 4: Update `load_segmentation_config` in `segment.py`**

In `load_segmentation_config`, after the `iou_dedup_threshold` line (line 59), add before the closing `)`:

```python
        attention_mode=seg.get("attention_mode", SegmentationConfig.attention_mode),
        prompt_strategy=seg.get("prompt_strategy", SegmentationConfig.prompt_strategy),
        sam_version=seg.get("sam_version", SegmentationConfig.sam_version),
        sam2_checkpoint=seg.get("sam2_checkpoint", SegmentationConfig.sam2_checkpoint),
        sam2_model_cfg=seg.get("sam2_model_cfg", SegmentationConfig.sam2_model_cfg),
```

- [ ] **Step 5: Make `classifier_model` accept `None` and add guard in `segment()`**

In `PatternSegmenter.__init__` (line 98), change type annotation:
```python
        classifier_model: Any | None,
```

In `segment()`, replace the first two lines (lines 368-369):
```python
        if self._seg_config.attention_mode == "classifier":
            if self._model is None:
                raise ValueError(
                    "classifier_model is required when attention_mode='classifier'. "
                    "Pass a fitted LogisticRegression or set attention_mode='self_attention'."
                )
            class_idx = list(self._model.classes_).index(class_name)
            cls_coef = self._model.coef_[class_idx]  # (768,)
```

(Leave the rest of `segment()` unchanged for now — Task 5 completes it.)

- [ ] **Step 6: Update `configs/classifier_config.yaml`**

Under the `segmentation:` section (after `iou_dedup_threshold: 0.5`), add:

```yaml
  attention_mode: classifier          # classifier | self_attention
  prompt_strategy: topk               # topk | fps
  sam_version: sam1                   # sam1 | sam2
  sam2_checkpoint: ""
  sam2_model_cfg: sam2_hiera_b+.yaml
```

- [ ] **Step 7: Run tests to confirm pass**

```bash
pytest tests/test_segmentation.py::test_segmentation_config_new_field_defaults tests/test_segmentation.py::test_load_segmentation_config_new_fields tests/test_segmentation.py::test_load_segmentation_config_new_field_defaults tests/test_segmentation.py::test_segmenter_no_classifier_self_attention_ok tests/test_segmentation.py::test_segmenter_no_classifier_raises_on_segment -v
```

Expected: 5 passed

- [ ] **Step 8: Run full test suite to confirm no regressions**

```bash
pytest tests/test_segmentation.py -v
```

Expected: all existing tests still pass

- [ ] **Step 9: Commit**

```bash
git add src/segmentation/segment.py configs/classifier_config.yaml tests/test_segmentation.py
git commit -m "feat: add attention_mode, prompt_strategy, sam_version config fields; optional classifier_model"
```

---

## Task 2: `extract_attention_maps` in `DinoFeatureExtractor`

**Files:**
- Modify: `src/features/dino_embeddings.py:148-170`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_segmentation.py` after the deduplication tests:

```python
# ---------------------------------------------------------------------------
# extract_attention_maps tests
# ---------------------------------------------------------------------------


def test_extract_attention_maps_shape():
    """Output shape is (n_heads, n_patches) — (12, 1369) at 518px for this model."""
    torch = pytest.importorskip("torch")
    from PIL import Image
    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    extractor = DinoFeatureExtractor(DinoConfig())
    img = Image.fromarray(
        np.zeros((518, 518, 3), dtype=np.uint8)
    )
    attn = extractor.extract_attention_maps(img)
    assert attn.ndim == 2
    n_heads, n_patches = attn.shape
    assert n_heads == 12        # ViT-B has 12 heads
    assert n_patches > 0        # read dynamically; empirically 1369 (37x37) at 518px


def test_extract_attention_maps_sums_to_one():
    """Each head's attention sums to ~1.0 after per-head re-normalization."""
    pytest.importorskip("torch")
    from PIL import Image
    from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

    extractor = DinoFeatureExtractor(DinoConfig())
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (518, 518, 3), dtype=np.uint8)
    )
    attn = extractor.extract_attention_maps(img)
    row_sums = attn.sum(axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_segmentation.py::test_extract_attention_maps_shape tests/test_segmentation.py::test_extract_attention_maps_sums_to_one -v
```

Expected: FAIL with `AttributeError: 'DinoFeatureExtractor' object has no attribute 'extract_attention_maps'`

- [ ] **Step 3: Implement `extract_attention_maps` in `dino_embeddings.py`**

Add after `extract_patch_tokens` (after line 170):

```python
    def extract_attention_maps(self, image: Image.Image) -> np.ndarray:
        """Extract per-head CLS→patch self-attention from the last transformer block.

        Runs a forward pass with ``output_attentions=True``, slices the last
        layer's attention tensor to CLS→patch weights (skipping CLS and the
        4 register tokens in the key axis), then re-normalizes per head so
        each row sums to 1.0.

        Args:
            image: PIL Image.

        Returns:
            float32 array of shape ``(n_heads, n_patches)``, values sum to
            1.0 per head. ``n_patches`` is always ``return_value.shape[1]``.
        """
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (1, n_heads, n_tokens, n_tokens) per layer
        # Last layer, CLS query (row 0), patch keys (columns 5+: skip CLS + 4 registers)
        attn = outputs.attentions[-1][0, :, 0, 5:].cpu().numpy().astype(np.float32)

        # Re-normalize: the patch-only slice no longer sums to 1 (CLS+register cols excluded)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-10)
        return attn
```

- [ ] **Step 4: Run to confirm pass**

```bash
pytest tests/test_segmentation.py::test_extract_attention_maps_shape tests/test_segmentation.py::test_extract_attention_maps_sums_to_one -v
```

Expected: 2 passed (note: loads DINOv3 model, takes ~30s on first run)

- [ ] **Step 5: Commit**

```bash
git add src/features/dino_embeddings.py tests/test_segmentation.py
git commit -m "feat: add extract_attention_maps() to DinoFeatureExtractor with per-head normalization"
```

---

## Task 3: `_compute_self_attention` and head selection

**Files:**
- Modify: `src/segmentation/segment.py` (after `_compute_attention`, around line 177)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_segmentation.py` after the attention map tests section:

```python
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
    from unittest.mock import MagicMock
    segmenter = _make_segmenter()
    n_patches = 16
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = (
        np.full((2, n_patches), 1.0 / n_patches, dtype=np.float32)
    )
    segmenter._extractor = mock_extractor

    from PIL import Image
    dummy = Image.fromarray(np.zeros((56, 56, 3), dtype=np.uint8))
    attn_out, n_patches_out = segmenter._compute_self_attention(dummy, (56, 56))
    assert attn_out.shape == (56, 56)
    assert n_patches_out == n_patches


def test_compute_self_attention_range():
    """Output values are in [0, 1]."""
    from unittest.mock import MagicMock
    segmenter = _make_segmenter()
    rng = np.random.default_rng(7)
    n_patches = 25
    attn = rng.random((3, n_patches)).astype(np.float32)
    attn /= attn.sum(axis=-1, keepdims=True)
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = attn
    segmenter._extractor = mock_extractor

    from PIL import Image
    dummy = Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8))
    attn_out, _ = segmenter._compute_self_attention(dummy, (120, 100))
    assert attn_out.min() >= 0.0
    assert attn_out.max() <= 1.0 + 1e-6
    assert attn_out.dtype == np.float32


def test_compute_self_attention_returns_n_patches():
    """Returned n_patches equals the patch dimension of the attention maps."""
    from unittest.mock import MagicMock
    segmenter = _make_segmenter()
    n_patches = 36
    mock_extractor = MagicMock()
    mock_extractor.extract_attention_maps.return_value = (
        np.ones((4, n_patches), dtype=np.float32) / n_patches
    )
    segmenter._extractor = mock_extractor

    from PIL import Image
    dummy = Image.fromarray(np.zeros((84, 84, 3), dtype=np.uint8))
    _, returned_n = segmenter._compute_self_attention(dummy, (84, 84))
    assert returned_n == n_patches
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_segmentation.py::test_select_attention_head_max_entropy tests/test_segmentation.py::test_select_attention_head_valid_index tests/test_segmentation.py::test_compute_self_attention_shape tests/test_segmentation.py::test_compute_self_attention_range tests/test_segmentation.py::test_compute_self_attention_returns_n_patches -v
```

Expected: FAIL with AttributeError (methods don't exist yet)

- [ ] **Step 3: Add `_select_attention_head` and `_compute_self_attention` to `PatternSegmenter`**

Add after the `_compute_attention` method (after line 177 in `segment.py`), before `# Thresholding`:

```python
    # ------------------------------------------------------------------
    # Self-attention map (alternative to classifier-based attention)
    # ------------------------------------------------------------------

    def _select_attention_head(self, attn_maps: np.ndarray) -> int:
        """Return the index of the head with the highest spatial entropy.

        Higher entropy = attention is spread over more patches = more
        spatially informative for segmentation prompts.

        Args:
            attn_maps: float32, shape (n_heads, n_patches), each row sums to 1.

        Returns:
            Index of the highest-entropy head.
        """
        eps = 1e-10
        entropy = -(attn_maps * np.log(attn_maps + eps)).sum(axis=-1)  # (n_heads,)
        return int(np.argmax(entropy))

    def _compute_self_attention(
        self,
        pil_img: Image.Image,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, int]:
        """Compute attention map from the max-entropy DINOv3 self-attention head.

        Extracts per-head CLS→patch attention, selects the head with the
        highest spatial entropy, then upsamples and smooths to native resolution
        (same post-processing as ``_compute_attention``).

        Args:
            pil_img: PIL Image at native resolution.
            image_size: (width, height) in PIL convention.

        Returns:
            (attention_map, n_patches) where attention_map is float32 (H, W)
            in [0, 1] and n_patches is read from the attention tensor shape.
        """
        from scipy.ndimage import gaussian_filter

        attn_maps = self._get_extractor().extract_attention_maps(pil_img)
        n_patches = attn_maps.shape[1]

        best_head = self._select_attention_head(attn_maps)
        selected = attn_maps[best_head]  # (n_patches,)

        grid_size = int(round(np.sqrt(n_patches)))
        width, height = image_size

        spatial = selected.reshape(grid_size, grid_size).astype(np.float32)
        heat_pil = Image.fromarray((spatial * 255).astype(np.uint8), mode="L")
        heat_up = (
            np.asarray(heat_pil.resize((width, height), Image.BILINEAR), dtype=np.float32)
            / 255.0
        )

        heat_smooth = gaussian_filter(heat_up, sigma=2).astype(np.float32)
        smin, smax = heat_smooth.min(), heat_smooth.max()
        attn_out = (
            (heat_smooth - smin) / (smax - smin) if smax > smin else heat_smooth
        )
        return attn_out, n_patches
```

- [ ] **Step 4: Run to confirm pass**

```bash
pytest tests/test_segmentation.py::test_select_attention_head_max_entropy tests/test_segmentation.py::test_select_attention_head_valid_index tests/test_segmentation.py::test_compute_self_attention_shape tests/test_segmentation.py::test_compute_self_attention_range tests/test_segmentation.py::test_compute_self_attention_returns_n_patches -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/segment.py tests/test_segmentation.py
git commit -m "feat: add _select_attention_head and _compute_self_attention to PatternSegmenter"
```

---

## Task 4: `_farthest_point_sample` and FPS prompt strategy

**Files:**
- Modify: `src/segmentation/segment.py` (`_patch_centroids` and new `_farthest_point_sample`)
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_segmentation.py` after the patch centroid tests:

```python
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
    from unittest.mock import MagicMock
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_segmentation.py::test_fps_returns_correct_count tests/test_segmentation.py::test_fps_no_duplicates tests/test_segmentation.py::test_fps_fewer_candidates_than_n_points tests/test_segmentation.py::test_fps_more_spread_than_topk tests/test_segmentation.py::test_patch_centroids_fps_correct_count tests/test_segmentation.py::test_patch_centroids_fps_in_bounds -v
```

Expected: FAIL (methods not implemented yet)

- [ ] **Step 3: Add `_farthest_point_sample` to `PatternSegmenter`**

Add just before `_patch_centroids` (around line 206):

```python
    def _farthest_point_sample(
        self,
        candidates: np.ndarray,
        grid_size: int,
        n_points: int,
    ) -> np.ndarray:
        """Select spatially diverse patch indices via farthest-point sampling.

        Seeds from the first candidate (caller should pre-sort descending by
        attention so the highest-attention patch is the seed).

        Args:
            candidates: 1-D int array of flat patch indices.
            grid_size: side length of the patch grid (e.g. 14 for 196 patches).
            n_points: number of points to select.

        Returns:
            Selected flat patch indices, shape (min(n_points, len(candidates)),).
        """
        if len(candidates) <= n_points:
            return candidates

        rows = candidates // grid_size
        cols = candidates % grid_size
        coords = np.stack([rows, cols], axis=1).astype(np.float32)  # (N, 2)

        selected_local = [0]  # seed: first candidate (highest attention)
        min_dists = np.full(len(candidates), np.inf)

        for _ in range(n_points - 1):
            last = coords[selected_local[-1]]
            dists = np.linalg.norm(coords - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected_local] = -np.inf  # exclude already selected
            selected_local.append(int(np.argmax(min_dists)))

        return candidates[np.array(selected_local)]
```

- [ ] **Step 4: Update `_patch_centroids` to support FPS branch**

Replace the body of `_patch_centroids` starting from the `n_fg = ...` line (around line 236) with:

```python
        n_fg = min(self._seg_config.n_foreground_prompts, n_patches)
        n_bg = min(self._seg_config.n_background_prompts, n_patches)

        sorted_idx = np.argsort(patch_activations)

        if self._seg_config.prompt_strategy == "fps":
            median_val = float(np.median(patch_activations))
            # Foreground: at or above median, sorted descending (seed = highest attn patch)
            fg_mask = patch_activations >= median_val
            fg_candidates = np.where(fg_mask)[0]
            fg_candidates = fg_candidates[np.argsort(patch_activations[fg_candidates])[::-1]]
            # Background: below median, sorted ascending (seed = lowest attn patch —
            # most clearly background, analogous to seeding fg from highest attn patch)
            bg_candidates = np.where(~fg_mask)[0]
            bg_candidates = bg_candidates[np.argsort(patch_activations[bg_candidates])]
            fg_indices = self._farthest_point_sample(fg_candidates, grid_size, n_fg)
            bg_indices = self._farthest_point_sample(bg_candidates, grid_size, n_bg)
        else:  # "topk"
            fg_indices = sorted_idx[-n_fg:][::-1]  # highest activation first
            bg_indices = sorted_idx[:n_bg]          # lowest activation first
```

- [ ] **Step 5: Run to confirm pass**

```bash
pytest tests/test_segmentation.py::test_fps_returns_correct_count tests/test_segmentation.py::test_fps_no_duplicates tests/test_segmentation.py::test_fps_fewer_candidates_than_n_points tests/test_segmentation.py::test_fps_more_spread_than_topk tests/test_segmentation.py::test_patch_centroids_fps_correct_count tests/test_segmentation.py::test_patch_centroids_fps_in_bounds -v
```

Expected: 6 passed

- [ ] **Step 6: Run full segmentation test suite**

```bash
pytest tests/test_segmentation.py -v
```

Expected: all tests pass (both topk and fps centroid tests)

- [ ] **Step 7: Commit**

```bash
git add src/segmentation/segment.py tests/test_segmentation.py
git commit -m "feat: add _farthest_point_sample and fps prompt strategy to PatternSegmenter"
```

---

## Task 5: SAM 2 + `pyproject.toml` + wire `segment()` self-attention path

**Files:**
- Modify: `src/segmentation/segment.py` (`_get_sam_predictor`, `segment()`)
- Modify: `pyproject.toml`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_segmentation.py`:

```python
# ---------------------------------------------------------------------------
# SAM 2 predictor test
# ---------------------------------------------------------------------------


def test_get_sam2_predictor_type():
    """_get_sam_predictor returns SAM2ImagePredictor when sam_version='sam2'."""
    sam2 = pytest.importorskip("sam2")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from src.segmentation.segment import PatternSegmenter, SegmentationConfig
    from src.features.dino_embeddings import DinoConfig
    from unittest.mock import MagicMock

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
    from unittest.mock import MagicMock, patch
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

    # Write a tiny image for segment() to open
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
```

Also add `from pathlib import Path` at the top of the new test if not already present (check — it's imported in the existing test file body).

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_segmentation.py::test_get_sam2_predictor_type tests/test_segmentation.py::test_segment_self_attention_path -v
```

Expected: `test_get_sam2_predictor_type` skips if sam2 not installed; `test_segment_self_attention_path` FAILS (segment() ignores attention_mode currently)

- [ ] **Step 3: Update `pyproject.toml` — add sam2 to `[ml]` extras**

In `pyproject.toml`, in the `ml = [` block, add after the pydensecrf line:

```toml
    "sam2 @ git+https://github.com/facebookresearch/sam2.git",
```

Install it:

```bash
pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"
```

If SAM 2 checkpoint not yet downloaded, download `sam2_hiera_base_plus.pt`:

```bash
mkdir -p /Volumes/Fangorn/Software/sam2
# Download from Meta's SAM 2 release:
# https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# Save to /Volumes/Fangorn/Software/sam2/sam2_hiera_base_plus.pt
```

- [ ] **Step 4: Update `_get_sam_predictor` to branch on `sam_version`**

Replace the body of `_get_sam_predictor` (lines 118-128 in `segment.py`) with:

```python
    def _get_sam_predictor(self) -> Any:
        if self._sam_predictor is None:
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cpu"

            if self._seg_config.sam_version == "sam2":
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam2_model = build_sam2(
                    self._seg_config.sam2_model_cfg,
                    self._seg_config.sam2_checkpoint,
                    device=device,
                )
                self._sam_predictor = SAM2ImagePredictor(sam2_model)
                logger.info("SAM 2 loaded on device=%s", device)
            else:  # "sam1"
                from segment_anything import SamPredictor, sam_model_registry

                sam = sam_model_registry["vit_b"](
                    checkpoint=self._seg_config.sam_checkpoint
                )
                sam.to(device)
                self._sam_predictor = SamPredictor(sam)
                logger.info("SAM loaded on device=%s", device)

        return self._sam_predictor
```

- [ ] **Step 5: Update `segment()` to branch on `attention_mode`**

Replace the full body of `segment()` with:

```python
    def segment(self, image_path: Path, class_name: str) -> SegmentationResult:
        """Run full segmentation pipeline for a single image.

        Supports two attention modes (set via ``seg_config.attention_mode``):
        - ``"classifier"``: uses LR coefficient × patch tokens (requires classifier_model)
        - ``"self_attention"``: uses max-entropy DINOv3 self-attention head (unsupervised)

        Args:
            image_path: Path to the image file (JPEG or PNG).
            class_name: The image's true class label (e.g. ``"mudcrack"``).
                        Only required when ``attention_mode="classifier"``.

        Returns:
            SegmentationResult with attention map, binary mask, and instance masks.
        """
        pil_img = Image.open(image_path).convert("RGB")
        image_size = pil_img.size          # (width, height) PIL convention
        image_np = np.asarray(pil_img)     # (H, W, 3) uint8 for SAM (read-only view is fine)

        if self._seg_config.attention_mode == "classifier":
            if self._model is None:
                raise ValueError(
                    "classifier_model is required when attention_mode='classifier'. "
                    "Pass a fitted LogisticRegression or set attention_mode='self_attention'."
                )
            class_idx = list(self._model.classes_).index(class_name)
            cls_coef = self._model.coef_[class_idx]  # (768,)
            patch_tokens = self._get_extractor().extract_patch_tokens(pil_img)
            n_patches = patch_tokens.shape[0]
            attention_map = self._compute_attention(patch_tokens, cls_coef, image_size)
        else:  # "self_attention"
            attention_map, n_patches = self._compute_self_attention(pil_img, image_size)

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

- [ ] **Step 6: Run all new tests**

```bash
pytest tests/test_segmentation.py::test_get_sam2_predictor_type tests/test_segmentation.py::test_segment_self_attention_path -v
```

Expected: `test_segment_self_attention_path` passes; `test_get_sam2_predictor_type` passes if SAM 2 installed + checkpoint exists, skips otherwise

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/test_segmentation.py -v
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add src/segmentation/segment.py pyproject.toml tests/test_segmentation.py
git commit -m "feat: add SAM 2 support and wire self_attention path in segment()"
```

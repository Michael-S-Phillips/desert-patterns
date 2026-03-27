# SAM Segmentation Improvements Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the SAM-based segmentation pipeline with three orthogonal improvements: DINOv3 per-head self-attention as an alternative attention source, farthest-point sampling for spatially diverse prompts, and SAM 2 support — all gated by config flags with no breaking changes to existing behavior.

**Architecture:** Extend `PatternSegmenter` and `DinoFeatureExtractor` in-place. Five new config fields control the new modes. All combinations of attention mode, prompt strategy, and SAM version are independently usable.

**Tech Stack:** Python, PyTorch/transformers (DINOv3 attention extraction), SAM 2 (`sam2` from facebookresearch/sam2), scikit-learn (existing), numpy, PIL.

---

## Files

- Modify: `src/features/dino_embeddings.py` — add `extract_attention_maps()`
- Modify: `src/segmentation/segment.py` — new config fields, updated `load_segmentation_config()`, `_compute_self_attention()`, `_farthest_point_sample()`, updated `_patch_centroids()`, updated `_get_sam_predictor()`, optional `classifier_model`
- Modify: `configs/classifier_config.yaml` — add 5 new fields under `segmentation:`
- Modify: `pyproject.toml` — add `sam2` to `[ml]` extras
- Modify: `tests/test_segmentation.py` — new tests for all additions

---

## Config

Five new fields added to `SegmentationConfig` and `configs/classifier_config.yaml`:

```yaml
segmentation:
  # existing fields unchanged
  attention_mode: classifier        # "classifier" | "self_attention"
  prompt_strategy: topk             # "topk" | "fps"
  sam_version: sam1                 # "sam1" | "sam2"
  sam2_checkpoint: ""               # path to sam2_hiera_base_plus.pt
  sam2_model_cfg: sam2_hiera_b+.yaml
```

Defaults preserve current behavior (`classifier`, `topk`, `sam1`). `sam2_checkpoint` and `sam2_model_cfg` are only used when `sam_version: sam2`.

### `load_segmentation_config` update

Add these five lines to the existing field-by-field construction inside `load_segmentation_config()`:

```python
attention_mode=seg.get("attention_mode", SegmentationConfig.attention_mode),
prompt_strategy=seg.get("prompt_strategy", SegmentationConfig.prompt_strategy),
sam_version=seg.get("sam_version", SegmentationConfig.sam_version),
sam2_checkpoint=seg.get("sam2_checkpoint", SegmentationConfig.sam2_checkpoint),
sam2_model_cfg=seg.get("sam2_model_cfg", SegmentationConfig.sam2_model_cfg),
```

---

## `DinoFeatureExtractor` changes

### `extract_attention_maps(image: Image.Image) -> np.ndarray`

Runs a forward pass with `output_attentions=True`. Extracts the last transformer block's attention tensor (`outputs.attentions[-1]`), shape `(1, n_heads, n_tokens, n_tokens)`. Slices CLS→patch attention: `attn[0, :, 0, 5:]`, skipping the CLS token itself (index 0 in the query axis) and 4 register tokens (indices 1–4 in the key axis), yielding shape `(n_heads, n_patches)`.

**Normalization:** The raw slice does not sum to 1.0 per head (because the CLS and register columns are excluded). Re-normalize before returning:

```python
attn_patch = outputs.attentions[-1][0, :, 0, 5:].cpu().numpy().astype(np.float32)
attn_patch = attn_patch / (attn_patch.sum(axis=-1, keepdims=True) + 1e-10)
return attn_patch
```

Returns `(n_heads, n_patches)` float32 numpy array, values sum to 1.0 per head. `n_patches` is always `return_value.shape[1]` and must be read dynamically. At 518px input the model produces 1369 patches (37×37) — the `vitb16` in the model name is a family identifier, not literal patch size; the model uses 14px patches (518/14=37 exactly), as confirmed by the existing `extract_patch_tokens` returning shape `(1369, 768)`.

---

## `PatternSegmenter` changes

### `SegmentationConfig` — new fields

```python
attention_mode: str = "classifier"       # "classifier" | "self_attention"
prompt_strategy: str = "topk"            # "topk" | "fps"
sam_version: str = "sam1"               # "sam1" | "sam2"
sam2_checkpoint: str = ""
sam2_model_cfg: str = "sam2_hiera_b+.yaml"
```

### Constructor — `classifier_model` becomes optional, order preserved

The existing parameter order is kept to avoid breaking call sites. `classifier_model` moves to a keyword-only optional:

```python
def __init__(
    self,
    classifier_model: Any | None,
    seg_config: SegmentationConfig,
    dino_config: Any,
) -> None:
```

`classifier_model` accepts `None`; `segment()` raises `ValueError` with a clear message if `attention_mode == "classifier"` and `classifier_model is None`.

Existing tests that construct `PatternSegmenter(mock_clf, SegmentationConfig(), DinoConfig())` require no changes.

### `_compute_self_attention(pil_img, image_size) -> tuple[np.ndarray, int]`

Returns `(attention_map, n_patches)` — `n_patches` is derived from the tensor rather than passed in, eliminating the circular dependency in `segment()`.

1. Calls `self._get_extractor().extract_attention_maps(pil_img)` → `(n_heads, n_patches)`; read `n_patches = attn_maps.shape[1]`
2. Selects head with highest spatial entropy: `H(h) = −Σ p·log(p + ε)` where `p = attn_maps[h]` (already normalized)
3. Reshapes selected head to `(grid_size, grid_size)` where `grid_size = round(sqrt(n_patches))`
4. Upsamples to native image resolution `(W, H)` via `PIL.Image.resize(BILINEAR)`
5. Applies Gaussian smoothing (sigma=2, same as classifier path)
6. Normalizes to `[0, 1]`
7. Returns `(float32 (H, W) attention map, n_patches: int)`

### `_farthest_point_sample(candidates: np.ndarray, grid_size: int, n_points: int) -> np.ndarray`

Selects `n_points` spatially diverse patch indices from a candidate set using farthest-point sampling in patch-grid coordinates.

- `candidates`: 1-D int array of flat patch indices (subset of `[0, n_patches)`)
- `grid_size`: side length of the patch grid (e.g. 37 for 1369 patches)

Algorithm:
1. Convert flat indices to `(row, col)` grid coordinates: `row = idx // grid_size`, `col = idx % grid_size`
2. If `len(candidates) <= n_points`, return `candidates` unchanged
3. Seed selection with the first element of `candidates` (caller pre-sorts by descending attention so highest-attention patch goes first)
4. Iteratively: compute minimum Euclidean distance from each remaining candidate to any already-selected point; pick the candidate that maximizes this distance
5. Return selected flat indices, shape `(n_points,)`

### `_patch_centroids` — updated

Branches on `self._seg_config.prompt_strategy`:

- `"topk"`: existing behavior unchanged — top-N and bottom-N patches by attention value
- `"fps"`: split candidates at the **median** of `patch_activations` (the per-cell averaged attention used internally); foreground = patches at or above median; background = patches below median. Sort each set by descending attention (so FPS seeds from the highest-attention patch). Call `_farthest_point_sample` on each set to select `n_foreground_prompts` / `n_background_prompts` diverse points. Convert selected indices to pixel centroids as before.

### `_get_sam_predictor` — updated

Branches on `self._seg_config.sam_version`:

- `"sam1"`: existing `SamPredictor` from `segment_anything` (unchanged)
- `"sam2"`: imports `sam2.build_sam.build_sam2` and `sam2.sam2_image_predictor.SAM2ImagePredictor`; builds from `sam2_checkpoint` and `sam2_model_cfg`; device resolution same as SAM 1 (MPS → CPU)

The `predict()` call interface is identical between SAM 1 and SAM 2 (`set_image`, `predict` with `point_coords`, `point_labels`, `multimask_output`), so `_run_sam()` requires no changes.

### `segment()` — updated

```python
if self._seg_config.attention_mode == "classifier":
    if self._classifier_model is None:
        raise ValueError("classifier_model required when attention_mode='classifier'")
    patch_tokens = self._get_extractor().extract_patch_tokens(pil_img)
    n_patches = patch_tokens.shape[0]
    attention_map = self._compute_attention(patch_tokens, cls_coef, image_size)
else:  # self_attention
    attention_map, n_patches = self._compute_self_attention(pil_img, image_size)

binary_mask = self._threshold(attention_map)
instance_masks, iou_scores = self._run_sam(image_np, attention_map, n_patches)
```

---

## `pyproject.toml`

Add to `[ml]` extras:
```toml
"sam2 @ git+https://github.com/facebookresearch/sam2.git",
```

**Checkpoint:** User must download `sam2_hiera_base_plus.pt` from Meta's SAM 2 release and set `sam2_checkpoint` in config. Recommended path: `/Volumes/Fangorn/Software/sam2/sam2_hiera_base_plus.pt`.

---

## Data Flow Summary

| `attention_mode` | `prompt_strategy` | Attention source | Prompt selection |
|---|---|---|---|
| `classifier` | `topk` | LR coef · patch tokens | top-N / bottom-N |
| `classifier` | `fps` | LR coef · patch tokens | FPS from median split |
| `self_attention` | `topk` | Max-entropy DINOv3 head | top-N / bottom-N |
| `self_attention` | `fps` | Max-entropy DINOv3 head | FPS from median split |

`sam_version` is orthogonal to both dimensions.

---

## Testing

All tests added to `tests/test_segmentation.py`.

### `extract_attention_maps`
- `test_extract_attention_maps_shape` — `pytest.importorskip("torch")`; output shape `(12, 1369)` at 518px — confirmed empirically by existing `extract_patch_tokens` tests; read `n_patches` from the returned tensor shape, never hardcode
- `test_extract_attention_maps_sums_to_one` — each head sums to ≈ 1.0 after re-normalization

### Head selection
- `test_select_max_entropy_head_clear_winner` — synthetic `(3, 16)` attention where head 1 has uniform distribution (max entropy); assert index 1 returned
- `test_select_max_entropy_head_valid_index` — random input; returned index in `[0, n_heads)`

### `_compute_self_attention`
- `test_compute_self_attention_shape` — output `attention_map` shape matches `(H, W)`; mock `extract_attention_maps` to avoid torch
- `test_compute_self_attention_range` — values in `[0, 1]`
- `test_compute_self_attention_returns_n_patches` — returned `n_patches` equals `mock_attn.shape[1]`

### `_farthest_point_sample`
- `test_fps_returns_correct_count` — returns exactly `n_points` indices
- `test_fps_no_duplicates` — all returned indices are unique
- `test_fps_more_spread_than_topk` — mean pairwise distance of FPS selection > mean pairwise distance of top-N selection on synthetic clustered attention
- `test_fps_fewer_candidates_than_n_points` — returns all candidates unchanged

### Config
- `test_load_segmentation_config_new_fields` — all 5 new fields read correctly from dict
- `test_load_segmentation_config_new_field_defaults` — missing section returns `"classifier"`, `"topk"`, `"sam1"`, `""`, `"sam2_hiera_b+.yaml"`

### `_get_sam_predictor` with SAM 2
- `test_get_sam2_predictor` — `pytest.importorskip("sam2")`; `sam_version="sam2"` with valid checkpoint returns `SAM2ImagePredictor`

### Constructor
- `test_segmenter_no_classifier_self_attention_ok` — constructs with `classifier_model=None` when `attention_mode="self_attention"` without error
- `test_segmenter_no_classifier_raises_on_segment` — `attention_mode="classifier"` with `classifier_model=None` raises `ValueError` on `segment()`

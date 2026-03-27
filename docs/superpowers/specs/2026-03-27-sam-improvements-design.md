# SAM Segmentation Improvements Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the SAM-based segmentation pipeline with three orthogonal improvements: DINOv3 per-head self-attention as an alternative attention source, farthest-point sampling for spatially diverse prompts, and SAM 2 support — all gated by config flags with no breaking changes to existing behavior.

**Architecture:** Extend `PatternSegmenter` and `DinoFeatureExtractor` in-place. Four new config fields control the new modes. All combinations of attention mode, prompt strategy, and SAM version are independently usable.

**Tech Stack:** Python, PyTorch/transformers (DINOv3 attention extraction), SAM 2 (`sam2` from facebookresearch/sam2), scikit-learn (existing), numpy, PIL.

---

## Files

- Modify: `src/features/dino_embeddings.py` — add `extract_attention_maps()`
- Modify: `src/segmentation/segment.py` — new config fields, `_compute_self_attention()`, `_farthest_point_sample()`, updated `_patch_centroids()`, updated `_get_sam_predictor()`, optional `classifier_model`
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

---

## `DinoFeatureExtractor` changes

### `extract_attention_maps(image: Image.Image) -> np.ndarray`

Runs a forward pass with `output_attentions=True`. Extracts the last transformer block's attention tensor (`outputs.attentions[-1]`), shape `(1, n_heads, n_tokens, n_tokens)`. Slices CLS→patch attention: `attn[0, :, 0, 5:]`, skipping the CLS token itself (index 0) and 4 register tokens (indices 1–4) to get patch attention only.

Returns `(n_heads, n_patches)` float32 numpy array. For DINOv3 ViT-B/16 at 518px input: `(12, 1369)`.

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

### Constructor — `classifier_model` becomes optional

```python
def __init__(
    self,
    seg_config: SegmentationConfig,
    dino_config: Any,
    classifier_model: Any | None = None,
) -> None:
```

`segment()` raises `ValueError` if `attention_mode == "classifier"` and `classifier_model is None`.

### `_compute_self_attention(pil_img, image_size, n_patches) -> np.ndarray`

1. Calls `self._get_extractor().extract_attention_maps(pil_img)` → `(n_heads, n_patches)`
2. Selects head with highest spatial entropy: `H(h) = −Σ p·log(p+ε)` where `p = softmax(attn[h])` over patches
3. Reshapes selected head to `(grid_size, grid_size)` where `grid_size = round(sqrt(n_patches))`
4. Upsamples to native image resolution via `PIL.Image.resize(BILINEAR)`
5. Applies Gaussian smoothing (sigma=2, same as classifier path)
6. Normalizes to `[0, 1]`
7. Returns float32 `(H, W)` attention map

### `_farthest_point_sample(patch_indices, grid_size, n_points) -> np.ndarray`

Selects `n_points` spatially diverse patches from a candidate set using farthest-point sampling in patch-grid coordinates.

1. Convert flat patch indices to `(row, col)` grid coordinates
2. Seed with the candidate having the highest attention value
3. Iteratively: compute minimum Euclidean distance from each remaining candidate to any already-selected point; pick the candidate that maximizes this distance
4. Return selected flat indices, shape `(n_points,)`

If `len(patch_indices) <= n_points`, returns all candidates (no sampling needed).

### `_patch_centroids` — updated

Branches on `self._seg_config.prompt_strategy`:

- `"topk"`: existing behavior — top-N and bottom-N patches by attention value
- `"fps"`: foreground candidates = patches above threshold; background = patches below threshold; FPS applied to each set separately to select `n_foreground_prompts` / `n_background_prompts` diverse points

In both cases, pixel centroids are computed from patch grid position as before.

### `_get_sam_predictor` — updated

Branches on `self._seg_config.sam_version`:

- `"sam1"`: existing `SamPredictor` from `segment_anything` (unchanged)
- `"sam2"`: imports `sam2.build_sam.build_sam2` and `sam2.sam2_image_predictor.SAM2ImagePredictor`; builds from `sam2_checkpoint` and `sam2_model_cfg`; device resolution same as SAM 1 (MPS → CPU)

The `predict()` call interface is identical between SAM 1 and SAM 2 (`set_image`, `predict` with `point_coords`, `point_labels`, `multimask_output`), so `_run_sam()` requires no changes.

### `segment()` — updated

```
if attention_mode == "classifier":
    patch_tokens = extract_patch_tokens(pil_img)
    attention_map = _compute_attention(patch_tokens, cls_coef, image_size)
else:  # self_attention
    attention_map = _compute_self_attention(pil_img, image_size, n_patches)

binary_mask = _threshold(attention_map)
instance_masks, iou_scores = _run_sam(image_np, attention_map, n_patches)
```

When `attention_mode == "self_attention"`, `n_patches` is derived from `attention_map` shape (`H * W / (patch_size^2)` is not available directly — instead use `grid_size^2` stored during `_compute_self_attention`). Simplest: `n_patches` is stored as an instance variable during attention computation, or returned alongside the attention map.

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
| `classifier` | `fps` | LR coef · patch tokens | FPS diverse |
| `self_attention` | `topk` | Max-entropy DINOv3 head | top-N / bottom-N |
| `self_attention` | `fps` | Max-entropy DINOv3 head | FPS diverse |

`sam_version` is orthogonal to both dimensions.

---

## Testing

All tests added to `tests/test_segmentation.py`.

### `extract_attention_maps`
- `test_extract_attention_maps_shape` — `pytest.importorskip("torch")`; shape `(12, 1369)` for ViT-B at 518px
- `test_extract_attention_maps_sums_to_one` — each head's attention sums to approximately 1.0 (softmax rows)

### Head selection
- `test_select_max_entropy_head_clear_winner` — synthetic `(3, 16)` attention where head 1 has uniform distribution (max entropy); assert index 1 returned
- `test_select_max_entropy_head_valid_index` — random input; returned index in `[0, n_heads)`

### `_compute_self_attention`
- `test_compute_self_attention_shape` — output shape matches `(H, W)` of input image; mocked `extract_attention_maps`
- `test_compute_self_attention_range` — values in `[0, 1]`

### `_farthest_point_sample`
- `test_fps_returns_correct_count` — returns exactly `n_points` indices
- `test_fps_no_duplicates` — all returned indices are unique
- `test_fps_more_spread_than_topk` — mean pairwise distance of FPS selection > mean pairwise distance of top-N selection on synthetic clustered attention

### Config
- `test_load_segmentation_config_new_fields` — all 5 new fields read correctly from dict
- `test_load_segmentation_config_new_field_defaults` — missing section returns `"classifier"`, `"topk"`, `"sam1"`, `""`, `"sam2_hiera_b+.yaml"`

### `_get_sam_predictor` with SAM 2
- `test_get_sam2_predictor` — `pytest.importorskip("sam2")`; `sam_version="sam2"` with valid checkpoint returns `SAM2ImagePredictor`

### Constructor
- `test_segmenter_no_classifier_self_attention_ok` — constructs without classifier when `attention_mode="self_attention"`
- `test_segmenter_no_classifier_raises_on_segment` — `attention_mode="classifier"` with `classifier_model=None` raises `ValueError` on `segment()`

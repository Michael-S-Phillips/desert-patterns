# Multi-Scale Segmentation & Per-Layer Attention Design

## Goal

Two independent improvements to the segmentation pipeline:

1. **A3 — Per-layer attention selection:** instead of always using the last DINOv3 transformer block's attention, select the `(layer, head)` pair with the highest spatial entropy across all 12 layers × 12 heads.
2. **B — Multi-scale analysis:** a new script that groups images by site and altitude, runs the existing segmenters independently at each scale, and produces side-by-side comparison figures and a per-site JSON summary.

Both are gated by config flags with no breaking changes to existing code or tests.

---

## Architecture

### A3 — Per-layer attention selection

**Motivation:** The last transformer block's attention tends to be the most semantically abstract. Intermediate layers often retain finer spatial structure that is more useful for segmentation prompts. Selecting the best `(layer, head)` pair by entropy gives the most spatially discriminative map without requiring any supervision.

**Changes:**

- **`src/features/dino_embeddings.py`** — add `extract_all_layer_attentions(image)`:
  - Runs a forward pass with `output_attentions=True` (model already loaded with `attn_implementation="eager"`)
  - Returns shape `(n_layers, n_heads, n_patches)` = `(12, 12, 1369)` at 518px input, float32, each `[layer, head]` row sums to 1.0 (per-head normalized, CLS→patch slice, register tokens excluded, same normalization as `extract_attention_maps()`)

- **`src/segmentation/segment.py`** — add `_select_best_layer_and_head(all_layer_attns)`:
  - Input: `(n_layers, n_heads, n_patches)`
  - Computes entropy for all `n_layers × n_heads` combinations: `-(a * log(a + eps)).sum(axis=-1)` reshaped to `(n_layers, n_heads)`
  - Returns `(layer_idx, head_idx)` of the pair with highest entropy (both indices within bounds)

- **`src/segmentation/segment.py`** — add `attention_layer: str = "last"` to `SegmentationConfig`:
  - `"last"` → existing path (`extract_attention_maps()` → `_select_attention_head()`), no change
  - `"best"` → new path (`extract_all_layer_attentions()` → `_select_best_layer_and_head()` → index `all_layer_attns[layer_idx, head_idx]` → `(n_patches,)` vector → existing upsample/smooth/threshold path)

- **`src/segmentation/segment.py`** — update `_compute_self_attention()` to branch on `attention_layer`:
  ```python
  if self._seg_config.attention_layer == "best":
      all_attns = self._get_extractor().extract_all_layer_attentions(pil_img)
      n_patches = all_attns.shape[2]
      layer_idx, head_idx = self._select_best_layer_and_head(all_attns)
      selected = all_attns[layer_idx, head_idx]   # (n_patches,)
  else:  # "last"
      attn_maps = self._get_extractor().extract_attention_maps(pil_img)
      n_patches = attn_maps.shape[1]
      best_head = self._select_attention_head(attn_maps)
      selected = attn_maps[best_head]              # (n_patches,)
  # Both branches continue with the same post-processing block:
  #   reshape selected (n_patches,) → (grid_size, grid_size)
  #   → upsample to (width, height) via PIL BILINEAR
  #   → gaussian_filter(sigma=2)
  #   → min-max normalize to [0, 1]
  # This block must be present once, after the if/else, operating on `selected` and `n_patches`.
  ```

- **`src/segmentation/segment.py`** — update `load_segmentation_config()`:
  Add after the `sam2_model_cfg` line:
  ```python
  attention_layer=seg.get("attention_layer", SegmentationConfig.attention_layer),
  ```

- **`configs/classifier_config.yaml`** — add `attention_layer: last` under `segmentation:`

No changes to `extract_attention_maps()` or `_select_attention_head()` — all existing tests remain valid.

---

### B — Multi-scale analysis script

**Motivation:** The dataset has full altitude stacks at Big Pool and JBIO sites (1m, 5m, 20m, 60m, 120m, 240m drone images). Running segmentation independently at each altitude and comparing results lets us understand how pattern structure changes across scales — both for research insight and for diagnosing classifier overfitting.

**New file: `scripts/generate_multiscale_segmentations.py`**

Pipeline:
1. Load config and image catalog (`data/metadata/image_catalog.csv`)
2. Group images by `(site_name, altitude_m)` using the catalog's `site_name` and `altitude_m` columns. The `site_name` column contains values such as `big_pool`, `biofilm_pool`, `mudcrack`. The `--site` flag accepts these values; `jbio` is an alias for `biofilm_pool`.
3. For each site, iterate images sorted by ascending `altitude_m`
4. For the comparison figure, one representative image is selected per distinct `altitude_m` value (the first image by `image_id` when multiple images share the same altitude). This uses `altitude_m` directly rather than `altitude_group` to preserve the distinction between e.g. 120m and 240m images (both carry `altitude_group = "high"`).
5. Run the configured segmenter (`PatternSegmenter` in `self_attention` mode, or `ClaspSegmenter`) on each image
6. Save per-image outputs to `outputs/segmentations_multiscale/{site_name}/{altitude_m}m/`
7. Produce per-site comparison figure (see layout below)
8. Write per-site JSON summary: `{"{altitude_m}m": {n_images, mean_n_instances, mean_mask_area_fraction, mean_attention_entropy}}`; for `ClaspSegmenter`, `mean_attention_entropy` is omitted (no attention map). For `PatternSegmenter`, `mean_attention_entropy` is the mean over images of the Shannon entropy computed from the patch-level attention vector (`selected`, shape `(n_patches,)`, sums to 1.0) before upsampling — not from the final `(H, W)` attention map.

**CLI:**
```
python scripts/generate_multiscale_segmentations.py \
  --config configs/classifier_config.yaml \
  --segmenter [sam|clasp]                    # default: sam
  --site [big_pool|biofilm_pool|jbio|all]    # default: all; jbio = biofilm_pool
  --force
  --verbose
```

**Comparison figure layout:**

- **SAM segmenter:** columns = (thumbnail, attention map, mask overlay); rows = one per distinct `altitude_m`, sorted ascending
- **CLASP segmenter:** columns = (thumbnail, label map overlay); rows = one per distinct `altitude_m`, sorted ascending. The attention map column is omitted — CLASP does not produce an attention map.

Uses `FigureStyle` and colourblind-safe palette consistent with the rest of the codebase. Saved as 300 DPI PNG + SVG.

**Outputs:**
- `outputs/segmentations_multiscale/{site_name}/{altitude_m}m/{stem}_overlay.png`
- `outputs/segmentations_multiscale/{site_name}/{altitude_m}m/{stem}_attn.png` (SAM only)
- `outputs/segmentations_multiscale/{site_name}_comparison.png` + `.svg`
- `outputs/segmentations_multiscale/{site_name}_summary.json`

---

## Config additions

```yaml
segmentation:
  attention_layer: last        # NEW: last | best
```

`SegmentationConfig` dataclass:
```python
attention_layer: str = "last"  # "last" | "best"
```

`load_segmentation_config()` addition (after `sam2_model_cfg` line):
```python
attention_layer=seg.get("attention_layer", SegmentationConfig.attention_layer),
```

---

## Data flow

```
A3 path (attention_mode=self_attention, attention_layer=best):
  pil_img
    → DinoFeatureExtractor.extract_all_layer_attentions()   # (12, 12, n_patches)
    → PatternSegmenter._select_best_layer_and_head()        # (layer_idx, head_idx)
    → all_layer_attns[layer_idx, head_idx]                  # (n_patches,)  ← indexing step
    → [existing reshape → upsample → gaussian_filter → normalize]
    → attention_map  # float32 (H, W)

A3 path (attention_layer=last):
  [existing path unchanged]

B path:
  image_catalog.csv
    → group by (site_name, altitude_m)
    → for each image: run PatternSegmenter or ClaspSegmenter
    → collect SegmentationResult / ClaspResult
    → select one representative per altitude_m (first by image_id)
    → generate comparison figure (layout depends on segmenter type)
    → write JSON summary
```

---

## Testing

**`tests/test_segmentation.py`** — new tests for A3:

- `test_extract_all_layer_attentions_shape` — shape is `(12, 12, n_patches)`, `n_patches > 0` (torch-gated)
- `test_extract_all_layer_attentions_sums_to_one` — each `[layer, head]` row sums to ~1.0 (torch-gated)
- `test_select_best_layer_and_head_picks_max_entropy` — construct a `(3, 2, 16)` tensor where `[1, 0]` is uniform (max entropy) and all others are peaked; assert returned indices are `(1, 0)` and both are in bounds
- `test_compute_self_attention_best_layer` — `_compute_self_attention()` with `attention_layer="best"` and mocked `extract_all_layer_attentions()` (returns `(2, 3, 16)`) produces output `attn_map` of shape `(H, W)` and asserts `n_patches == 16` (read from `all_attns.shape[2]`)

**`tests/test_multiscale.py`** — new test file for B:

- `test_group_by_site_altitude` — given a small mock DataFrame with `site_name` and `altitude_m` columns, the grouping function returns the correct `{site_name: {altitude_m: [image_rows]}}` structure
- `test_representative_image_selection` — when multiple images share the same `altitude_m`, the first by `image_id` is selected as representative
- `test_multiscale_summary_keys` — given mocked segmentation results, the JSON summary has the expected top-level altitude keys and per-altitude sub-keys

No torch-dependent tests for B (mocked segmenters throughout).

---

## File map

| File | Action |
|------|--------|
| `src/features/dino_embeddings.py` | Add `extract_all_layer_attentions()` |
| `src/segmentation/segment.py` | Add `attention_layer` config field, `_select_best_layer_and_head()`, update `_compute_self_attention()`, update `load_segmentation_config()` |
| `configs/classifier_config.yaml` | Add `attention_layer: last` |
| `scripts/generate_multiscale_segmentations.py` | Create new script |
| `tests/test_segmentation.py` | Add A3 tests |
| `tests/test_multiscale.py` | Create new test file for B |

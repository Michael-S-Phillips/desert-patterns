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
  - Returns shape `(n_layers, n_heads, n_patches)` = `(12, 12, 1369)` at 518px input, float32, each row sums to 1.0 (per-head normalized, CLS→patch slice, register tokens excluded, same normalization as `extract_attention_maps()`)

- **`src/segmentation/segment.py`** — add `_select_best_layer_and_head(all_layer_attns)`:
  - Input: `(n_layers, n_heads, n_patches)`
  - Computes entropy for all `n_layers × n_heads` combinations
  - Returns `(layer_idx, head_idx)` of the pair with highest entropy

- **`src/segmentation/segment.py`** — add `attention_layer: str = "last"` to `SegmentationConfig`:
  - `"last"` → existing path (`extract_attention_maps()` → `_select_attention_head()`), no change
  - `"best"` → new path (`extract_all_layer_attentions()` → `_select_best_layer_and_head()`)

- **`src/segmentation/segment.py`** — update `_compute_self_attention()` to branch on `attention_layer`

- **`configs/classifier_config.yaml`** — add `attention_layer: last` under `segmentation:`

- **`load_segmentation_config()`** — add `attention_layer` field

No changes to `extract_attention_maps()` or `_select_attention_head()` — all existing tests remain valid.

---

### B — Multi-scale analysis script

**Motivation:** The dataset has full altitude stacks at Big Pool and JBIO sites (1m, 5m, 20m, 60m, 120m, 240m drone images). Running segmentation independently at each altitude and comparing results lets us understand how pattern structure changes across scales — both for research insight and for diagnosing classifier overfitting.

**New file: `scripts/generate_multiscale_segmentations.py`**

Pipeline:
1. Load config and image catalog (`data/metadata/image_catalog.csv`)
2. Group images by `(site_label, altitude_group)` using the catalog's `label` and `altitude_group` columns (low/mid/high/ground)
3. For each site, iterate altitude groups in order (ground → low → mid → high)
4. Run the configured segmenter (`PatternSegmenter` in `self_attention` mode, or `ClaspSegmenter`) on each image
5. Save per-image outputs to `outputs/segmentations_multiscale/{site}/{altitude}/`
6. Produce per-site comparison figure: grid with one row per altitude group, columns = (thumbnail, attention map, mask overlay)
7. Write per-site JSON summary: `{altitude_group: {n_images, mean_n_instances, mean_mask_area_fraction, mean_attention_entropy}}`

**CLI:**
```
python scripts/generate_multiscale_segmentations.py \
  --config configs/classifier_config.yaml \
  --segmenter [sam|clasp]    # default: sam
  --site [big_pool|jbio|all] # default: all
  --force
  --verbose
```

**Outputs:**
- `outputs/segmentations_multiscale/{site}/{altitude_group}/{stem}_overlay.png`
- `outputs/segmentations_multiscale/{site}/{altitude_group}/{stem}_attn.png`
- `outputs/segmentations_multiscale/{site}_comparison.png` — grid figure (300 DPI PNG + SVG)
- `outputs/segmentations_multiscale/{site}_summary.json`

**Comparison figure layout:** rows = altitude groups sorted low→high altitude number; columns = (thumbnail, attention map, segmentation overlay). One representative image per altitude group (highest-quality or first available). Uses `FigureStyle` and colourblind-safe palette consistent with the rest of the codebase.

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

---

## Data flow

```
A3 path (attention_mode=self_attention, attention_layer=best):
  pil_img
    → DinoFeatureExtractor.extract_all_layer_attentions()  # (12, 12, n_patches)
    → PatternSegmenter._select_best_layer_and_head()       # (layer_idx, head_idx)
    → selected attention vector                             # (n_patches,)
    → [existing upsample + smooth + threshold + SAM]

A3 path (attention_layer=last):
  [existing path unchanged]

B path:
  image_catalog.csv
    → group by (site, altitude_group)
    → for each image: run PatternSegmenter or ClaspSegmenter
    → collect SegmentationResult / ClaspResult
    → generate comparison figure + JSON summary
```

---

## Testing

**`tests/test_segmentation.py`** — new tests for A3:

- `test_extract_all_layer_attentions_shape` — shape is `(12, 12, n_patches)`, `n_patches > 0` (torch-gated)
- `test_extract_all_layer_attentions_sums_to_one` — each `[layer, head]` row sums to ~1.0 (torch-gated)
- `test_select_best_layer_and_head_valid_indices` — returned `(layer_idx, head_idx)` in bounds for random input
- `test_select_best_layer_and_head_picks_uniform` — when one `(layer, head)` is uniform and rest are peaked, it picks the uniform one
- `test_compute_self_attention_best_layer` — `_compute_self_attention()` with `attention_layer="best"` and mocked `extract_all_layer_attentions()` produces correct output shape

**`tests/test_multiscale.py`** — new test file for B:

- `test_group_by_site_altitude` — grouping logic produces correct structure from a small mock catalog DataFrame
- `test_multiscale_output_structure` — given mocked segmenters, output dirs are created and JSON summary has expected keys

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

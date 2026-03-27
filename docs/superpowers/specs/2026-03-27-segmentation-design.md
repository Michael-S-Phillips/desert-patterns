# Desert Pattern Segmentation Design

**Date:** 2026-03-27
**Status:** Approved
**Scope:** Batch segmentation of 367 labeled desert pattern images using DINOv3 patch attention maps as SAM prompts. Produces per-image instance mask overlays, per-instance metadata JSON, and per-class summary gallery figures.

---

## Problem Statement

The patch projection overlays (from `scripts/generate_classifier_figures.py`) show which spatial regions of each image drive the logistic regression classifier's decision. This spec extends that signal into two concrete segmentation outputs:

- **Saliency/attention mask (soft + binary):** which pixels are most responsible for the classification
- **Instance outlines:** pixel-precise boundaries of individual pattern objects (mudcrack cells, pools, biofilm patches), obtained by prompting SAM with the attention-derived foreground/background points

---

## Architecture

### New module

`src/segmentation/segment.py` — `SegmentationConfig`, `SegmentationResult`, `PatternSegmenter`. Parallel to `src/classification/`. Reusable for future inference-time segmentation.

### Script

`scripts/generate_segmentations.py` — batch processes all 367 labeled images (or a user-specified subset), saves per-image outputs and summary gallery figures.

---

## Per-Image Pipeline

For each image:

1. **Attention map** — Pass the PIL image to `DinoFeatureExtractor.extract_patch_tokens()` (which internally runs the HuggingFace processor; do **not** resize the PIL image before passing it). The processor resizes to `input_size` (518px) before tokenizing, and the ViT-B/16 patch size is 16px. The empirically-observed output is `n_patches=196` (14×14 grid), confirmed at runtime in this codebase. Derive `grid_size = int(round(np.sqrt(n_patches)))` dynamically — do not hardcode 14. Output shape is `(n_patches, 768)`. Project onto `model.coef_[class_idx]` where `class_idx = list(model.classes_).index(class_name)`: `activations = patch_tokens @ cls_coef` → `(n_patches,)`. Reshape to `(grid_size, grid_size)`, normalize to `[0, 1]`, upsample bilinearly to the **file's native resolution** (PIL `.size` is `(width, height)`; for `PIL.Image.resize` pass `(width, height)`; for numpy indexing use `(height, width)`) — this is the size SAM will receive. Apply Gaussian smoothing (`sigma=2`). Save as a soft heatmap overlay PNG.

2. **Binary attention mask** — Threshold the smoothed attention map using the method specified by `threshold_method`:
   - `otsu` (default): apply OpenCV Otsu thresholding (`cv2.threshold` with `cv2.THRESH_OTSU`) to the uint8-scaled map
   - `percentile`: threshold at `attention_percentile` (e.g., 0.70 → top 30% of activation values)
   Save as a binary PNG (0/255).

3. **SAM instance masks** — Load the image from disk at native resolution as a uint8 RGB numpy array and pass it to `SamPredictor.set_image()`. (The attention map was already upsampled to this same native resolution in step 1, so the coordinate spaces match.) Map the top-`n_foreground_prompts` highest-activation patch centroids to pixel coordinates in the native-resolution image → SAM foreground prompt points (label=1). Map the bottom-`n_background_prompts` patches → background points (label=0). For each foreground prompt point, call `SamPredictor.predict(point_coords, point_labels)` independently (each call returns `(masks, iou_scores, logits)` with 3 candidate masks at different scales; select the mask with the highest predicted IoU score). Deduplicate overlapping masks by IoU: discard any mask whose IoU with an already-accepted mask exceeds `iou_dedup_threshold`. Discard masks with area < `min_mask_area_fraction` × image area.

4. **Overlay PNG** — Composite: original image + per-instance colored masks at 50% alpha + instance boundary outlines + attention heatmap blended at 30% alpha underneath the masks. Save as `{image_stem}_overlay.png`.

5. **Metadata JSON** — Save `{image_stem}_masks.json`: list of dicts, one per accepted instance mask: `{area_px, bbox_xyxy, centroid_xy, iou_score}`.

### Patch centroid → pixel coordinate mapping

Patch index `i` maps to grid position `(row, col) = divmod(i, grid_size)` where `grid_size = int(round(np.sqrt(n_patches)))`. The centroid pixel coordinate in the native-resolution image (used for SAM prompts) is:

```
cx = (col + 0.5) / grid_size * image_width
cy = (row + 0.5) / grid_size * image_height
```

---

## Outputs

```
outputs/segmentations/
├── mudcrack/
│   ├── {stem}_overlay.png      # original + instance masks + attention heatmap
│   ├── {stem}_attention.png    # soft heatmap overlay only
│   ├── {stem}_mask.png         # binary attention mask (0/255)
│   └── {stem}_masks.json       # [{area_px, bbox_xyxy, centroid_xy, iou_score}, ...]
├── big_pool/
└── jbio/

outputs/figures/
├── segmentation_gallery_mudcrack.png/.svg
├── segmentation_gallery_big_pool.png/.svg
└── segmentation_gallery_jbio.png/.svg
```

Summary gallery per class: 12 images in a 3×4 grid, selected as the top-12 highest-confidence examples for that class using `model.predict_proba(X)` on the cached embeddings `X` (same ranking as `fig_class_gallery` in `generate_classifier_figures.py`). Each thumbnail is the `_overlay.png` resized to 224×224. Uses `src.visualization.style.save_figure()` and project font/DPI conventions (300 DPI, Arial/DejaVu fallback, PNG + SVG).

---

## Script

`scripts/generate_segmentations.py`

**CLI flags:**
- `--config` (default: `configs/classifier_config.yaml`)
- `--force` — re-run even if `_overlay.png` already exists
- `--verbose`
- `--image` — one or more image filenames (stems or full names); if set, only those images are processed and gallery figures are skipped

**Steps:**
1. Load config YAML → `SegmentationConfig` (from `segmentation:` sub-dict via `load_segmentation_config()`) and `DinoConfig` (from `dino:` sub-dict via `load_dino_config()`). Both are passed to `PatternSegmenter.__init__`.
2. Load `lr_classifier.joblib` + `label_encoder.joblib` from `outputs/models/classifier/`
3. Load cached embeddings + labels from `outputs/features/classifier_embeddings.npy` / `classifier_labels.npy`
4. Scan labeled images via `scan_labeled_images()` (same sorted rglob order as training). Validate alignment with label cache: if `[lbl for _, lbl in image_list] != list(y)`, raise `ValueError("Image list and label cache are misaligned — re-run train_classifier.py to regenerate the cache")`. Use the image list to look up each image's `class_name` and the corresponding `model.coef_[class_idx]`.
5. If `--image` provided, filter image list to matching stems/filenames; raise a descriptive error if no match found
6. For each image: run pipeline (skip if overlay exists and not `--force`)
7. If `--image` not set: generate per-class gallery figures

---

## Configuration

New `segmentation:` block added to `configs/classifier_config.yaml`:

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

`load_segmentation_config(config_dict: dict) -> SegmentationConfig` reads the `segmentation:` sub-dict and populates the dataclass. Follows the same pattern as `load_classifier_config()`.

---

## `src/segmentation/segment.py`

### `SegmentationConfig` (dataclass)

```python
@dataclass
class SegmentationConfig:
    sam_checkpoint: str = "/Volumes/Rohan/Software/sam/sam_vit_b_01ec64.pth"
    output_dir: str = "outputs/segmentations"
    n_foreground_prompts: int = 5
    n_background_prompts: int = 5
    threshold_method: str = "otsu"        # "otsu" | "percentile"
    attention_percentile: float = 0.70
    min_mask_area_fraction: float = 0.005
    iou_dedup_threshold: float = 0.5
```

### `SegmentationResult` (dataclass)

```python
@dataclass
class SegmentationResult:
    attention_map: np.ndarray        # float32, shape (H, W), values in [0, 1]
    binary_mask: np.ndarray          # uint8, shape (H, W), values 0 or 255
    instance_masks: list[np.ndarray] # each bool array shape (H, W)
    iou_scores: list[float]    # one per instance mask
    image_path: Path
    class_name: str
```

### `PatternSegmenter`

- `__init__(self, classifier_model, seg_config: SegmentationConfig, dino_config: DinoConfig)` — lazy-loads DINOv3 extractor and SAM predictor on first use
- `segment(self, image_path: Path, class_name: str) -> SegmentationResult` — runs the full per-image pipeline
- `_compute_attention(self, patch_tokens: np.ndarray, cls_coef: np.ndarray, image_size: tuple[int, int]) -> np.ndarray` — steps 1–2; `image_size` is `(width, height)` in PIL convention, matching `PIL.Image.size`
- `_threshold(self, attention_map: np.ndarray) -> np.ndarray` — Otsu or percentile thresholding
- `_run_sam(self, image: np.ndarray, attention_map: np.ndarray) -> tuple[list[np.ndarray], list[float]]` — prompt generation + SAM predict + deduplication
- `_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float` — IoU between two boolean masks

---

## Testing

`tests/test_segmentation.py` — all tests use synthetic numpy arrays; SAM/torch-dependent tests use `pytest.importorskip("torch")`.

- `SegmentationConfig` default values and `load_segmentation_config()` round-trip
- `_compute_attention()`: given synthetic `(196, 768)` patch tokens and `(768,)` coef, verify output shape matches image size, values in `[0, 1]`
- `_threshold()` with `method="percentile"`: verify binary mask has only 0/255 values and correct foreground fraction
- `_threshold()` with `method="otsu"`: verify binary mask has only 0/255 values on a synthetic bimodal map
- `_iou()`: verify correct IoU for known overlapping/non-overlapping boolean arrays
- IoU deduplication logic: given a list of synthetic masks with known overlaps, verify correct masks are kept/discarded
- Patch centroid → pixel coordinate mapping: verify correct pixel coords for corner and center patches in a known image size

No automated tests for overlay PNGs or gallery figures — verified manually.

---

## Dependencies

No new dependencies. All present:
- `segment_anything` (SAM, via `[ml]` extras)
- `torch` (via `[ml]` extras)
- `transformers` (DINOv3, via `[ml]` extras)
- `opencv-python` (Otsu thresholding, already in core deps)
- `PIL`, `numpy`, `matplotlib`, `joblib`

# CLASP Segmentation Design

> **For agentic workers:** See `docs/superpowers/plans/` for the implementation plan.

**Goal:** Implement CLASP (Clustering via Adaptive Spectral Processing) as a second, fully-unsupervised segmentation method to compare against the existing DINOv3-attention + SAM pipeline.

**Reference:** Curie & da Costa, "CLASP: Adaptive Spectral Clustering for Unsupervised Per-Image Segmentation," arXiv:2509.25016v2, Oct 2025.

---

## Algorithm

CLASP segments a single image by:

1. Extracting per-patch feature vectors from DINOv3 (reusing `DinoFeatureExtractor.extract_patch_tokens()`).
2. Building an n×n cosine affinity matrix A where A_ij = (f_i · f_j) / (‖f_i‖ ‖f_j‖).
3. Eigendecomposing A = QΛQ^T (symmetric, so `scipy.linalg.eigh`; eigenvalues sorted descending).
4. Finding K_opt via the **eigengap elbow heuristic**: compute gaps δ_i = λ_i − λ_{i+1}, fit a line from (1, δ_1) to (n−1, δ_{n−1}), find i* = argmax perpendicular distance, K_opt = i* + 1.
5. Sweeping K ∈ [⌊K_opt(1−β)⌋, ⌈K_opt(1+β)⌉] (clamped to [min_clusters, max_clusters]): for each K run K-means on the first K eigenvectors and compute the silhouette score; select K with the highest score.
6. Upsampling the patch-level label map to native image resolution via nearest-patch assignment.
7. Optionally refining boundaries with DenseCRF (paper parameters: 20 iterations, Gaussian kernel σ_xy=4 / compat=4, bilateral kernel σ_xy=80 / σ_rgb=13 / compat=10).

For 518×518 DINOv3 input (37×37 = 1369 patches), the affinity matrix is 1369×1369 — tractable for eigendecomposition on CPU.

---

## Architecture

### New file: `src/segmentation/clasp.py`

**`ClaspConfig`** (dataclass)

| Field | Default | Notes |
|---|---|---|
| `output_dir` | `"outputs/segmentations_clasp"` | |
| `bandwidth` | `0.5` | β for K search range |
| `min_clusters` | `2` | hard floor |
| `max_clusters` | `15` | hard ceiling |
| `dense_crf` | `True` | apply DenseCRF refinement |
| `crf_iterations` | `20` | |
| `crf_gaussian_sxy` | `4` | spatial σ for Gaussian kernel |
| `crf_gaussian_compat` | `4` | |
| `crf_bilateral_sxy` | `80` | spatial σ for bilateral kernel |
| `crf_bilateral_srgb` | `13` | color σ |
| `crf_bilateral_compat` | `10` | |

**`load_clasp_config(config_dict: dict) -> ClaspConfig`** — reads `clasp:` sub-dict from the full YAML dict; same pattern as `load_segmentation_config`.

**`ClaspResult`** (dataclass)

| Field | Type | Description |
|---|---|---|
| `label_map` | `np.ndarray` | int32, shape (H, W), values 0…K−1 |
| `k_chosen` | `int` | final K after silhouette search |
| `segment_areas` | `list[int]` | pixel count per segment, sorted descending |
| `image_path` | `Path` | |

**`ClaspSegmenter`**

| Method | Description |
|---|---|
| `__init__(seg_config, dino_config)` | No classifier needed. Sets `_extractor = None`. |
| `_get_extractor()` | Lazy-loads `DinoFeatureExtractor`. |
| `_build_affinity(patch_tokens)` | L2-normalise rows, compute dot-product matrix → A (n×n float32). |
| `_eigengap_k(eigenvalues)` | Elbow heuristic on descending eigenvalue sequence; returns K_opt (int). |
| `_search_best_k(eigenvectors, k_opt)` | Silhouette sweep; returns best K (int) and patch label array (n,). |
| `_make_label_map(labels, image_size, n_patches)` | Nearest-patch upsample to native `image_size` (width, height); returns int32 (H, W). |
| `_apply_dense_crf(image_np, label_map, k)` | DenseCRF refinement; returns refined int32 (H, W). No-op when `dense_crf=False`. Guarded by `importorskip`-style lazy import. |
| `segment(image_path)` | Full pipeline; returns `ClaspResult`. No `class_name` argument — fully unsupervised. |

### Modified: `configs/classifier_config.yaml`

Add `clasp:` block with all `ClaspConfig` fields and their defaults.

### Modified: `pyproject.toml`

Add `pydensecrf` to the `[ml]` optional extras group.

### New file: `scripts/generate_clasp_segmentations.py`

Same CLI pattern as `generate_segmentations.py` (`--config`, `--force`, `--verbose`, `--image`).

- Scans the same labeled image directories as the existing script.
- Outputs to `outputs/segmentations_clasp/{class_name}/` to mirror the SAM-based structure.
- Saves per image:
  - `{stem}_clasp_overlay.png` — original image with segment colors blended at 50% alpha (Wong palette)
  - `{stem}_clasp_labelmap.png` — raw label map saved as indexed-color PNG
  - `{stem}_clasp_segments.json` — `{"k": K, "segment_areas": [...], "image_path": "..."}`
- No gallery figure (no classifier probabilities to rank by).

---

## Dependencies

`pydensecrf` is added to `[ml]` optional extras. It is lazy-imported inside `_apply_dense_crf` with a clear `ImportError` message directing users to `pip install ".[ml]"`. When `dense_crf=False` the import is never triggered.

---

## Configuration (full `clasp:` block)

```yaml
clasp:
  output_dir: outputs/segmentations_clasp
  bandwidth: 0.5
  min_clusters: 2
  max_clusters: 15
  dense_crf: true
  crf_iterations: 20
  crf_gaussian_sxy: 4
  crf_gaussian_compat: 4
  crf_bilateral_sxy: 80
  crf_bilateral_srgb: 13
  crf_bilateral_compat: 10
```

---

## Testing (`tests/test_clasp.py`)

| Test | What it verifies |
|---|---|
| `test_clasp_config_defaults` | All default field values |
| `test_load_clasp_config_overrides` | Partial override from dict |
| `test_load_clasp_config_missing_section` | Empty dict → defaults |
| `test_build_affinity_shape` | Output is (n, n) |
| `test_build_affinity_diagonal` | Diagonal values ≈ 1.0 |
| `test_build_affinity_range` | All values in [−1, 1] |
| `test_eigengap_k_clear_drop` | Synthetic eigenvalues with a sharp drop at index 3 → K_opt = 4 |
| `test_eigengap_k_monotone` | Monotone decay → returns valid int |
| `test_search_best_k_in_range` | Returned K is within clamped search range |
| `test_search_best_k_respects_bounds` | K_opt near min/max → K clamped correctly |
| `test_make_label_map_shape` | Output (H, W) matches image_size |
| `test_make_label_map_label_range` | All values in [0, K−1] |
| `test_apply_dense_crf_skipped` | `dense_crf=False` → input label map returned unchanged |
| `test_apply_dense_crf_with_mock` | `pydensecrf` importorskip; mock predictor → output shape matches |
| `test_clasp_segmenter_lazy_init` | `_extractor is None` at construction |

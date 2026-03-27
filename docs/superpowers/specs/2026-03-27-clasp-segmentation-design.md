# CLASP Segmentation Design

> **For agentic workers:** See `docs/superpowers/plans/` for the implementation plan.

**Goal:** Implement CLASP (Clustering via Adaptive Spectral Processing) as a second, fully-unsupervised segmentation method to compare against the existing DINOv3-attention + SAM pipeline.

**Reference:** Curie & da Costa, "CLASP: Adaptive Spectral Clustering for Unsupervised Per-Image Segmentation," arXiv:2509.25016v2, Oct 2025.

---

## Algorithm

CLASP segments a single image by:

1. Extracting per-patch feature vectors from DINOv3 (reusing `DinoFeatureExtractor.extract_patch_tokens()`).
2. Building an n×n cosine affinity matrix A where A_ij = (f_i · f_j) / (‖f_i‖ ‖f_j‖). Clip A to [0, 1] (`A = np.maximum(A, 0)`) before eigendecomposition — negative cosine similarities indicate opposing features and should not contribute affinity; standard spectral clustering requires a non-negative affinity matrix.
3. Eigendecomposing A = QΛQ^T (symmetric, so `scipy.linalg.eigh`; eigenvalues sorted descending).
4. Finding K_opt via the **eigengap elbow heuristic**: compute gaps δ_i = λ_i − λ_{i+1} for i = 1…n−1 (1-based, so gap at index i is the drop between eigenvalue i and i+1). Fit a line from point (1, δ_1) to point (n−1, δ_{n−1}). Find i* = argmax perpendicular distance from this line. K_opt = i* + 1. Example: a sharp drop between λ_3 and λ_4 gives i*=3, K_opt=4.
5. Sweeping K ∈ [⌊K_opt(1−β)⌋, ⌈K_opt(1+β)⌉] (clamped to [min_clusters, max_clusters]): for each K, take the first K eigenvectors (columns of Q corresponding to the K largest eigenvalues), row-normalize to unit L2 norm (`sklearn.preprocessing.normalize`), run K-means on the normalized rows, and compute the silhouette score using Euclidean distance in that normalized eigenvector space; select K with the highest score.
6. Upsampling the patch-level label map to native image resolution via nearest-patch assignment.
7. Optionally refining boundaries with DenseCRF. Parameters from §4.3 of the paper: 20 iterations, `gt_prob=0.8`, uncertain labeling disabled, Gaussian kernel σ_xy=4 / compat=4, bilateral kernel σ_xy=80 / σ_rgb=13 / compat=10. The `_apply_dense_crf` method operates on the native-resolution label map (post-upsample), not the 518×518 DINOv3 input.

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
| `crf_gt_prob` | `0.8` | ground-truth probability for unary potentials |
| `crf_gaussian_sxy` | `4` | spatial σ for Gaussian kernel |
| `crf_gaussian_compat` | `4` | |
| `crf_bilateral_sxy` | `80` | spatial σ for bilateral kernel |
| `crf_bilateral_srgb` | `13` | color σ |
| `crf_bilateral_compat` | `10` | |

**`load_clasp_config(config_dict: dict) -> ClaspConfig`** — reads `clasp:` sub-dict from the full YAML dict; same pattern as `load_segmentation_config`.

**`ClaspResult`** (dataclass)

| Field | Type | Description |
|---|---|---|
| `label_map` | `np.ndarray` | int32, shape (H, W) at native image resolution, values 0…K−1. Always native-res: `_make_label_map` upsamples from patch-grid to native, and `_apply_dense_crf` also operates at native resolution. |
| `k_chosen` | `int` | final K after silhouette search |
| `segment_areas` | `list[int]` | pixel count per segment, sorted descending |
| `image_path` | `Path` | |

**`ClaspSegmenter`**

| Method | Description |
|---|---|
| `__init__(self, seg_config: ClaspConfig, dino_config: DinoConfig)` | No classifier needed. Sets `_extractor = None`. |
| `_get_extractor()` | Lazy-loads `DinoFeatureExtractor(self._dino_config)`. |
| `_build_affinity(patch_tokens)` | L2-normalise rows, compute dot-product matrix → A (n×n float32); clip to [0, 1] with `np.maximum(A, 0)`. |
| `_eigengap_k(eigenvalues)` | Elbow heuristic on descending eigenvalue sequence; i* is 1-based index of the largest gap; returns K_opt = i* + 1 (int). |
| `_search_best_k(eigenvectors, k_opt)` | Silhouette sweep over clamped K range; returns best K (int) and patch label array (n,). |
| `_make_label_map(labels, image_size, n_patches)` | Nearest-patch upsample to native `image_size` (width, height); returns int32 (H, W) at native resolution. |
| `_apply_dense_crf(image_np, label_map, k)` | DenseCRF refinement at native resolution using `crf_gt_prob`, kernels, and iterations from config. Uncertain labeling disabled. Returns refined int32 (H, W). No-op (returns `label_map` unchanged) when `dense_crf=False`. `pydensecrf` lazy-imported with clear `ImportError` message. |
| `segment(image_path)` | Full pipeline; returns `ClaspResult`. No `class_name` argument — fully unsupervised. |

### Modified: `configs/classifier_config.yaml`

Add `clasp:` block with all `ClaspConfig` fields and their defaults.

### Modified: `pyproject.toml`

Add `pydensecrf` to the `[ml]` optional extras group.

### New file: `scripts/generate_clasp_segmentations.py`

Same CLI pattern as `generate_segmentations.py` (`--config`, `--force`, `--verbose`, `--image`).

- Loads `dino_config` via `load_dino_config(config_dict.get("dino", {}))` from the YAML — same as the SAM script, ensuring the correct DINOv3 model name is used.
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
  crf_gt_prob: 0.8
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
| `test_clasp_config_defaults` | All default field values including `crf_gt_prob=0.8` |
| `test_load_clasp_config_overrides` | Partial override from dict |
| `test_load_clasp_config_missing_section` | Empty dict → defaults |
| `test_build_affinity_shape` | Output is (n, n) |
| `test_build_affinity_diagonal` | Diagonal values ≈ 1.0 |
| `test_build_affinity_range` | All values in [0, 1] after clipping |
| `test_eigengap_k_clear_drop` | Synthetic eigenvalues with sharp drop between index 3 and 4 → K_opt = 4 |
| `test_eigengap_k_monotone` | Monotone decay → returns valid int in [1, n] |
| `test_search_best_k_in_range` | Returned K is within clamped search range |
| `test_search_best_k_respects_bounds` | K_opt near min/max → K clamped correctly |
| `test_make_label_map_shape` | Output (H, W) matches native image_size, not patch-grid size |
| `test_make_label_map_label_range` | All values in [0, K−1] |
| `test_apply_dense_crf_skipped` | `dense_crf=False` → input label map returned unchanged, no import attempted |
| `test_apply_dense_crf_with_mock` | `pytest.importorskip("pydensecrf")`; mock CRF2D → output shape matches input |
| `test_clasp_segmenter_lazy_init` | `_extractor is None` at construction |

# Classifier Analysis Figures Design

**Date:** 2026-03-25
**Status:** Approved
**Scope:** Nine publication-quality figures visualizing class structure and DINOv3 embedding interpretability for the supervised pattern classifier. Personal analysis use.

---

## Problem Statement

The trained logistic regression classifier achieves near-perfect CV accuracy (macro-F1 0.997). This spec covers a set of figures to understand *what* the classifier learned: which embedding dimensions discriminate the classes, what visual patterns correspond to those dimensions, and how the classes are distributed in embedding space.

---

## Output

All figures saved to `outputs/figures/classifier_analysis/` (created if absent).
Format: 300 DPI PNG + SVG. Font: Arial (DejaVu Sans fallback). Palette: Wong (discrete), viridis/cividis (continuous) per project conventions.

---

## Script

Single self-contained script: `scripts/generate_classifier_figures.py`

No new modules. Loads:
- `outputs/features/classifier_embeddings.npy` — shape (367, 768), float32
- `outputs/features/classifier_labels.npy` — shape (367,), str
- `outputs/models/classifier/lr_classifier.joblib` — sklearn LogisticRegression
- Image paths re-scanned via `scan_labeled_images()` from `src.classification.train`

CLI flags: `--config` (default `configs/classifier_config.yaml`), `--verbose`

---

## Figures

### 1. Per-class image galleries — `class_gallery_{name}.png/.svg` (3 files)

For each class (mudcrack, big_pool, jbio):
- Compute `proba = model.predict_proba(X)` — columns ordered by `model.classes_`; select the column for class `c` as `proba[:, list(model.classes_).index(c)]`
- Select top-12 highest-confidence images among those belonging to class `c` (or all if fewer than 12)
- Thumbnail grid: 3 rows × 4 cols, thumbnails resized to 224×224 (INTER_AREA)
- Title: `"{class_name} — top-12 by confidence"`, subtitle shows class count
- No axes; tight layout

### 2. LR coefficient heatmap — `lr_coefficient_heatmap.png/.svg`

- Input: `model.coef_` shape (3, 768)
- Select top-50 embedding dimensions by max absolute coefficient across all classes
- Plot: heatmap, rows = classes, cols = top-50 dims (sorted by descending max |coef|)
- Colormap: `RdBu_r` (diverging, colourblind-safe), symmetric around 0 (`vmin=-max_abs_coef`, `vmax=+max_abs_coef`)
- X-axis: dim indices, Y-axis: class names
- Colorbar labeled "LR coefficient"

### 3. Per-class coefficient bar chart — `lr_coefficient_bars.png/.svg`

- One subplot per class (3 subplots, horizontal layout)
- For each class: top-30 dims by absolute coefficient value, bars colored by sign (positive = Wong orange, negative = Wong blue)
- X-axis: embedding dim index, Y-axis: coefficient value
- Title per subplot: class name

### 4. Image extremes gallery — `image_extremes.png/.svg`

Answers "what visual pattern does embedding dim N correspond to?"

- Identify top-3 most discriminative embedding dimensions: for each class in turn, take highest-|coef| dims (top-1, then top-2, etc.) until 3 unique dims are accumulated across all classes; if all three classes share the same top-1 dim, expand to top-2 per class and so on. Layout adapts to the actual number of unique dims found (may be 1–3 rows).
- For each dim: sort all 367 images by their raw activation value on that dim; take top-6 (highest) and bottom-6 (lowest)
- Layout: one row per dim (3 rows), left half = low activation (6 thumbnails), right half = high activation (6 thumbnails)
- Row label: "Dim {idx} — low ←→ high", annotated with which class it discriminates most
- Thumbnails: 112×112px

### 5. Class probability histograms — `class_probability_histograms.png/.svg`

- One subplot per class (3 subplots, horizontal)
- For each class: histogram of `predict_proba` for the correct class column, over images belonging to that class
- Bins: 20, range [0, 1]
- X-axis: predicted probability, Y-axis: count
- Vertical dashed line at median; title shows median value

### 6. Cosine similarity matrix — `cosine_similarity_matrix.png/.svg`

- Compute mean embedding per class (3 vectors of shape 768)
- Compute all pairwise cosine similarities → 3×3 matrix
- Also compute mean intra-class cosine similarity (mean pairwise within each class)
- Display as annotated heatmap: rows/cols = class names, values = cosine similarity
- Colormap: `viridis`, range [0, 1]
- Diagonal = mean intra-class similarity; off-diagonal = inter-class similarity

### 7. PCA projection — `pca_projection.png/.svg`

- Use the shared PCA(n_components=50) fit (see Implementation Notes); project onto first 2 components
- Scatter plot colored by class (Wong palette), s=20, alpha=0.7
- Overlay top-5 loading vectors as arrows from origin (scaled for visibility), labeled with dim index
- Axes: "PC1 (X% var)", "PC2 (Y% var)"
- Legend: class names

### 8. PCA component image strips — `pca_component_strips.png/.svg`

- Use the shared PCA(n_components=50) fit; project onto first 4 components
- For each of the 4 components: sort images by projection score; take top-5 (highest) and bottom-5 (lowest)
- Layout: 4 rows (one per PC), 10 thumbnails per row (5 low | 5 high), with a center divider label
- Row label: "PC{n} ({X}% var) — low ←→ high"
- Thumbnails: 112×112px

### 9. PCA variance explained — `pca_variance_explained.png/.svg`

- Fit PCA(n_components=50) on embeddings
- Plot: individual explained variance ratio (bar) + cumulative (line) for first 50 components
- Mark cumulative 90% and 95% thresholds with horizontal dashed lines
- X-axis: component index, Y-axis: variance explained (0–1)
- Compact figure: 7"×4"

---

## Implementation Notes

- **Label/path alignment**: Load `y` from `classifier_labels.npy` (authoritative). To get image paths, call `scan_labeled_images(image_dir, label_map)` — it returns paths in sorted rglob order, which matches the training script's cache order. Validate: `assert [lbl for _, lbl in image_list] == list(y)`, raising a descriptive error if they disagree (stale cache).
- Thumbnail loading: `PIL.Image.open(path).convert("RGB")`, resize to target size with `PIL.Image.LANCZOS`
- **Shared PCA**: Fit `PCA(n_components=50, random_state=42)` once at script start; slice components for Figs 7, 8, 9 to avoid redundant SVD decompositions.
- **Probability column selection**: `proba = model.predict_proba(X)`, then `proba[:, list(model.classes_).index(cls)]` for class `cls`.
- PCA from `sklearn.decomposition.PCA`, cosine similarity from `sklearn.metrics.pairwise.cosine_similarity`
- Use `src.visualization.style.save_figure(fig, path, formats, dpi)` for all saves
- Use `src.visualization.style.setup_matplotlib_style()` once at script start

---

## Testing

No automated tests — figures are visual outputs for personal analysis. Verify manually by running the script and inspecting outputs.

---

## Dependencies

No new dependencies. All present: PIL, sklearn, numpy, matplotlib, joblib, PyYAML.

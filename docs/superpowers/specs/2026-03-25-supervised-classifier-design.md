# Supervised Pattern Classifier Design

**Date:** 2026-03-25
**Status:** Approved
**Scope:** Train a logistic regression classifier on DINOv3 embeddings to distinguish mudcrack, big pool, and JBIO pattern types from labeled images in `checked-images/`.

---

## Problem Statement

A set of 369 labeled images has been organized under `data/kim_2023/checked-images/` into 8 sub-folders that map to 3 coarse pattern classes: **mudcrack**, **big_pool**, and **jbio**. The goal is to:

1. Extract DINOv3 CLS embeddings for all images
2. Train a logistic regression classifier on those embeddings
3. Evaluate with stratified 5-fold CV and produce a confusion matrix
4. Generate a UMAP plot colored by class
5. Persist the trained model so new images can be classified at inference time

---

## Label Mapping

| Folder | Class |
|--------|-------|
| `1609 Mudcrack no mat` | `mudcrack` |
| `Big Pool no mat 1` | `big_pool` |
| `Big Pool no mat 2` | `big_pool` |
| `Big Pool no mat 3` | `big_pool` |
| `Big Pool no mat 4` | `big_pool` |
| `JBIO mat` | `jbio` |
| `JBIO pond` | `jbio` |
| `JBIO wet mat` | `jbio` |

Image discovery scans `checked-images/` **recursively** (JBIO mat contains nested subdirectories). The top-level folder name determines the label. Class counts: mudcrack=98, big_pool=183, jbio=86. Use `class_weight='balanced'` in logistic regression to handle mild imbalance.

---

## Architecture

### New files

```
src/classification/
├── __init__.py
├── train.py          # ClassifierTrainer
└── predict.py        # ClassifierPredictor

scripts/train_classifier.py
configs/classifier_config.yaml
tests/test_classifier_train.py
tests/test_classifier_predict.py
```

### `src/classification/train.py` — `ClassifierTrainer`

- Accepts: `X` (np.ndarray, N×768), `y` (list[str] labels)
- `evaluate(cv_folds=5)` → runs `StratifiedKFold`, returns per-class precision/recall/F1 and fold-level confusion matrices
- `fit()` → fits final `LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)` on full data
- `save(output_dir)` → saves `lr_classifier.joblib` + `label_encoder.joblib` + `config.json`

### `src/classification/predict.py` — `ClassifierPredictor`

- `load(model_dir)` → loads joblib model + label encoder
- `predict(image_path)` → extracts DINOv3 embedding via `DinoEmbedder`, returns `(class_name: str, probability: float, all_probs: dict[str, float])`
- `predict_from_embedding(embedding)` → same but accepts pre-computed embedding (np.ndarray 768-d)

### `scripts/train_classifier.py`

CLI flags: `--config`, `--force` (re-extract embeddings even if cache exists), `--verbose`

Steps:
1. Load `configs/classifier_config.yaml`
2. Scan `checked-images/` recursively → build `[(path, label)]`
3. Extract DINOv3 embeddings via existing `DinoEmbedder` (batched); cache to `outputs/features/classifier_embeddings.npy` and `classifier_labels.npy`
4. Run `ClassifierTrainer.evaluate()` → log classification report, save normalized confusion matrix PNG+SVG
5. Run `ClassifierTrainer.fit()` + `save()` → persist model to `outputs/models/classifier/`
6. Fit UMAP (cosine, n_neighbors=30, min_dist=0.1, seed=42) on embeddings → scatter plot colored by true class → save PNG+SVG

---

## Configuration (`configs/classifier_config.yaml`)

```yaml
classifier:
  model_type: logistic_regression
  cv_folds: 5
  random_state: 42
  max_iter: 1000
  class_weight: balanced

dino:
  model_name: facebook/dinov3-vitb16-pretrain-lvd1689m
  input_size: 518
  batch_size: 32

data:
  image_dir: data/kim_2023/checked-images
  label_map:
    "1609 Mudcrack no mat": mudcrack
    "Big Pool no mat 1": big_pool
    "Big Pool no mat 2": big_pool
    "Big Pool no mat 3": big_pool
    "Big Pool no mat 4": big_pool
    "JBIO mat": jbio
    "JBIO pond": jbio
    "JBIO wet mat ": jbio

output:
  model_dir: outputs/models/classifier
  figures_dir: outputs/figures
  embedding_cache: outputs/features/classifier_embeddings.npy
  label_cache: outputs/features/classifier_labels.npy
```

---

## Outputs

| Path | Description |
|------|-------------|
| `outputs/models/classifier/lr_classifier.joblib` | Trained logistic regression model |
| `outputs/models/classifier/label_encoder.joblib` | sklearn LabelEncoder |
| `outputs/models/classifier/config.json` | Saved config for reproducibility |
| `outputs/features/classifier_embeddings.npy` | Cached DINOv3 embeddings (N×768) |
| `outputs/features/classifier_labels.npy` | Corresponding string labels |
| `outputs/figures/classifier_confusion_matrix.png/.svg` | Normalized confusion matrix (300 DPI) |
| `outputs/figures/classifier_umap.png/.svg` | UMAP scatter colored by true class (300 DPI) |

---

## Evaluation

- Stratified 5-fold CV → per-class precision, recall, F1 + macro/weighted averages
- Normalized confusion matrix (row-normalized, colourblind-safe palette per project conventions)
- UMAP 2D scatter: Wong palette, one color per class, noise points if any in gray

---

## Testing

- `tests/test_classifier_train.py`: label scanning from directory structure, CV fold structure, model fit with synthetic embeddings (no torch needed), output artifact creation
- `tests/test_classifier_predict.py`: load/predict round-trip with mock joblib model and synthetic embedding; `predict_from_embedding` shape/type checks

All tests use synthetic numpy arrays. DINOv3-dependent paths use `pytest.importorskip("torch")`.

---

## Dependencies

No new dependencies required. All needed packages already present:
- `scikit-learn` (logistic regression, cross-validation, label encoding)
- `umap-learn` (UMAP)
- `matplotlib` (confusion matrix + UMAP plot)
- `transformers` + `torch` (DINOv3, via `[ml]` extras)
- `joblib` (model persistence)

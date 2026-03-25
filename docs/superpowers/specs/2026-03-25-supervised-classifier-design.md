# Supervised Pattern Classifier Design

**Date:** 2026-03-25
**Status:** Approved
**Scope:** Train a logistic regression classifier on DINOv3 embeddings to distinguish mudcrack, big pool, and JBIO pattern types from labeled images in `checked-images/`.

---

## Problem Statement

A set of 367 labeled images has been organized under `data/kim_2023/checked-images/` into 8 sub-folders that map to 3 coarse pattern classes: **mudcrack**, **big_pool**, and **jbio**. The goal is to:

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

Image discovery scans `checked-images/` **recursively** (JBIO mat contains nested subdirectories). The top-level folder name determines the label; folder names are `.strip()`-ped before lookup to handle trailing whitespace (the `JBIO wet mat` folder has a trailing space on disk). Only files with extensions `.jpg` / `.jpeg` / `.png` are included; others are skipped with a logged warning. Class counts: mudcrack=98, big_pool=183, jbio=86. Use `class_weight='balanced'` in logistic regression to handle mild imbalance.

---

## Architecture

### New files

```
src/classification/
├── __init__.py
├── train.py          # ClassifierTrainer, ClassifierConfig, CVResult
└── predict.py        # ClassifierPredictor

scripts/train_classifier.py
configs/classifier_config.yaml
tests/test_classifier_train.py
tests/test_classifier_predict.py
```

### `ClassifierConfig` dataclass (`src/classification/train.py`)

```python
@dataclass
class ClassifierConfig:
    model_type: str = "logistic_regression"
    cv_folds: int = 5
    random_state: int = 42
    max_iter: int = 1000
    class_weight: str = "balanced"
    image_dir: str = "data/kim_2023/checked-images"
    label_map: dict[str, str] = field(default_factory=dict)
    model_dir: str = "outputs/models/classifier"
    figures_dir: str = "outputs/figures"
    embedding_cache: str = "outputs/features/classifier_embeddings.npy"
    label_cache: str = "outputs/features/classifier_labels.npy"
    dino_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    dino_input_size: int = 518
    dino_batch_size: int = 32
    dino_device: str = "auto"
```

`load_classifier_config(config_dict: dict) -> ClassifierConfig` mirrors the pattern of `load_dino_config()` and other loaders in the project. Accepts the **full parsed YAML dict** (all four top-level keys) and populates `ClassifierConfig` by reading `config["classifier"]`, `config["dino"]`, `config["data"]`, and `config["output"]` sub-dicts internally.

### `CVResult` dataclass (`src/classification/train.py`)

```python
@dataclass
class CVResult:
    report: dict                        # sklearn classification_report as dict
    confusion_matrices: list[np.ndarray]  # one per fold (raw counts)
    class_names: list[str]              # label order matching confusion matrix axes (from LabelEncoder.classes_)
    macro_f1_mean: float
    macro_f1_std: float
    cohen_kappa_mean: float
    cohen_kappa_std: float
```

The saved confusion matrix figure is the **sum of all fold matrices**, then row-normalized (standard practice for cross-validated confusion matrices).

### `src/classification/train.py` — `ClassifierTrainer`

- `__init__(self, X: np.ndarray, y: list[str], config: ClassifierConfig)`
- `evaluate() -> CVResult` — runs `StratifiedKFold(n_splits=config.cv_folds)`, accumulates per-fold confusion matrices and metrics, returns `CVResult` with mean ± std macro-F1 and Cohen's kappa
- `fit() -> None` — fits final `LogisticRegression(max_iter=config.max_iter, class_weight=config.class_weight, random_state=config.random_state)` on full data; encodes labels with `LabelEncoder`
- `save(output_dir: Path) -> None` — saves `lr_classifier.joblib` + `label_encoder.joblib` + `config.json`

### `src/classification/predict.py` — `ClassifierPredictor`

- `__init__(self, model_dir: Path)` — loads `lr_classifier.joblib` + `label_encoder.joblib` on construction (matches `PatternPredictor` convention in `src/inference/predict.py`)
- `predict(image_path: Path) -> tuple[str, float, dict[str, float]]` — extracts DINOv3 embedding via `DinoFeatureExtractor`, returns `(class_name, probability, all_class_probs)`
- `predict_from_embedding(embedding: np.ndarray) -> tuple[str, float, dict[str, float]]` — same but accepts pre-computed 768-d embedding

### `scripts/train_classifier.py`

CLI flags:
- `--config` (default: `configs/classifier_config.yaml`)
- `--force` — re-extract embeddings AND re-train model even if cached artifacts exist; without this flag, both extraction and training are skipped if their respective output files exist
- `--verbose`

Steps:
1. Load `configs/classifier_config.yaml` → `ClassifierConfig`
2. Scan `checked-images/` recursively; strip folder names; filter to image extensions; build `list[tuple[Path, str]]`
3. Extract DINOv3 embeddings via `DinoFeatureExtractor` (batched); cache to `outputs/features/classifier_embeddings.npy` + `classifier_labels.npy`; skip if cache exists and `--force` not set. When loading from cache, validate `len(embeddings) == len(labels)` and raise a descriptive `ValueError` if they differ (guards against stale cache from a different image set)
4. Run `ClassifierTrainer.evaluate()` → log classification report with macro-F1 mean ± std and Cohen's kappa; save sum-of-folds normalized confusion matrix PNG+SVG (`cmap='viridis'`, 300 DPI, Arial font)
5. Run `ClassifierTrainer.fit()` + `save()` → persist model to `outputs/models/classifier/`
6. Fit UMAP (cosine, n_neighbors=30, min_dist=0.1, seed=42) on embeddings → scatter plot colored by true class (Wong palette) → save PNG+SVG; UMAP is **not** persisted (visualization only — projecting new images into this space is out of scope)

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
  device: auto

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
    "JBIO wet mat ": jbio  # Note: trailing space matches filesystem folder name; code strips before lookup

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
| `outputs/figures/classifier_confusion_matrix.png/.svg` | Sum-of-folds row-normalized confusion matrix, viridis colormap, 300 DPI |
| `outputs/figures/classifier_umap.png/.svg` | UMAP scatter colored by true class, Wong palette, 300 DPI |

---

## Evaluation

- Stratified 5-fold CV → per-class precision, recall, F1 + macro/weighted averages
- Report macro-F1 mean ± std across folds and Cohen's kappa mean ± std
- Confusion matrix: sum of fold matrices, row-normalized, `cmap='viridis'`, 300 DPI PNG+SVG, Arial font, min 8pt
- UMAP 2D scatter: Wong palette, one color per class

---

## Testing

- `tests/test_classifier_train.py`: label scanning (directory structure, strip, extension filter), `CVResult` fields, model fit and `save()` artifacts — all with synthetic np.ndarray embeddings (no torch needed)
- `tests/test_classifier_predict.py`: `ClassifierPredictor.__init__` load round-trip with mock joblib files; `predict_from_embedding` return type/shape checks

All tests use synthetic numpy arrays. DINOv3-dependent paths (`predict()`) use `pytest.importorskip("torch")`.

---

## Dependencies

No new dependencies required. All needed packages already present:
- `scikit-learn` (logistic regression, cross-validation, label encoding, Cohen's kappa)
- `umap-learn` (UMAP)
- `matplotlib` (confusion matrix + UMAP plot)
- `transformers` + `torch` (DINOv3, via `[ml]` extras)
- `joblib` (model persistence)

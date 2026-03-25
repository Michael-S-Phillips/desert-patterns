# Supervised Pattern Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a logistic regression classifier on DINOv3 embeddings to distinguish mudcrack, big_pool, and jbio pattern classes from 367 labeled images in `data/kim_2023/checked-images/`.

**Architecture:** New `src/classification/` module with `train.py` (ClassifierConfig, CVResult, ClassifierTrainer) and `predict.py` (ClassifierPredictor). A training script `scripts/train_classifier.py` orchestrates embedding extraction, 5-fold CV evaluation, confusion matrix + UMAP figures, and final model persistence.

**Tech Stack:** scikit-learn (LogisticRegression, StratifiedKFold, LabelEncoder), umap-learn, matplotlib, joblib, PyYAML, existing DinoFeatureExtractor from `src/features/dino_embeddings.py`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/classification/__init__.py` | Package marker |
| Create | `src/classification/train.py` | ClassifierConfig, CVResult, ClassifierTrainer, load_classifier_config, scan_labeled_images |
| Create | `src/classification/predict.py` | ClassifierPredictor |
| Create | `configs/classifier_config.yaml` | Runtime configuration |
| Create | `scripts/train_classifier.py` | CLI: extract → evaluate → figures → fit → save |
| Create | `tests/test_classifier_train.py` | Tests for train.py |
| Create | `tests/test_classifier_predict.py` | Tests for predict.py |

---

## Task 1: Package scaffold + ClassifierConfig

**Files:**
- Create: `src/classification/__init__.py`
- Create: `src/classification/train.py`
- Create: `configs/classifier_config.yaml`
- Test: `tests/test_classifier_train.py`

- [ ] **Step 1: Write failing tests for ClassifierConfig loading**

Create `tests/test_classifier_train.py`:

```python
"""Tests for src/classification/train.py — config loading."""
from __future__ import annotations

import pytest
from src.classification.train import ClassifierConfig, load_classifier_config


def _minimal_config() -> dict:
    return {
        "classifier": {"cv_folds": 3, "max_iter": 500},
        "dino": {"model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m", "input_size": 518, "batch_size": 16, "device": "cpu"},
        "data": {
            "image_dir": "data/kim_2023/checked-images",
            "label_map": {"Folder A": "class_a", "Folder B ": "class_b"},
        },
        "output": {
            "model_dir": "outputs/models/classifier",
            "figures_dir": "outputs/figures",
            "embedding_cache": "outputs/features/emb.npy",
            "label_cache": "outputs/features/lbl.npy",
        },
    }


def test_load_classifier_config_overrides():
    cfg = load_classifier_config(_minimal_config())
    assert cfg.cv_folds == 3
    assert cfg.max_iter == 500
    assert cfg.dino_batch_size == 16
    assert cfg.dino_device == "cpu"
    assert cfg.image_dir == "data/kim_2023/checked-images"
    assert cfg.model_dir == "outputs/models/classifier"


def test_load_classifier_config_defaults():
    cfg = load_classifier_config({"classifier": {}, "dino": {}, "data": {}, "output": {}})
    assert cfg.model_type == "logistic_regression"
    assert cfg.cv_folds == 5
    assert cfg.random_state == 42
    assert cfg.class_weight == "balanced"
    assert cfg.dino_model_name == "facebook/dinov3-vitb16-pretrain-lvd1689m"


def test_load_classifier_config_strips_label_map_keys():
    cfg = load_classifier_config(_minimal_config())
    # "Folder B " (trailing space) should be stripped to "Folder B"
    assert "Folder B" in cfg.label_map
    assert "Folder B " not in cfg.label_map
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Volumes/Fangorn/desert_patterns && source .venv/bin/activate
pytest tests/test_classifier_train.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.classification'`

- [ ] **Step 3: Create package marker**

Create `src/classification/__init__.py` (empty):

```python
"""Supervised pattern classifier for desert ground types."""
```

- [ ] **Step 4: Create train.py with config dataclass + loader**

Create `src/classification/train.py`:

```python
"""Supervised classifier training: config, label scanning, CV evaluation, model fit."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ClassifierConfig:
    """Configuration for supervised pattern classifier."""

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


def load_classifier_config(config_dict: dict) -> ClassifierConfig:
    """Load ClassifierConfig from full YAML config dict.

    Accepts the full parsed YAML dict (all four top-level keys: classifier,
    dino, data, output) and fans out sub-dicts internally.

    Args:
        config_dict: Full parsed YAML dictionary.

    Returns:
        Populated ClassifierConfig.
    """
    clf = config_dict.get("classifier", {})
    dino = config_dict.get("dino", {})
    data = config_dict.get("data", {})
    out = config_dict.get("output", {})

    # Strip trailing/leading whitespace from label_map keys (filesystem quirk)
    raw_label_map: dict[str, str] = data.get("label_map", {})
    label_map = {k.strip(): v for k, v in raw_label_map.items()}

    return ClassifierConfig(
        model_type=clf.get("model_type", ClassifierConfig.model_type),
        cv_folds=clf.get("cv_folds", ClassifierConfig.cv_folds),
        random_state=clf.get("random_state", ClassifierConfig.random_state),
        max_iter=clf.get("max_iter", ClassifierConfig.max_iter),
        class_weight=clf.get("class_weight", ClassifierConfig.class_weight),
        image_dir=data.get("image_dir", ClassifierConfig.image_dir),
        label_map=label_map,
        model_dir=out.get("model_dir", ClassifierConfig.model_dir),
        figures_dir=out.get("figures_dir", ClassifierConfig.figures_dir),
        embedding_cache=out.get("embedding_cache", ClassifierConfig.embedding_cache),
        label_cache=out.get("label_cache", ClassifierConfig.label_cache),
        dino_model_name=dino.get("model_name", ClassifierConfig.dino_model_name),
        dino_input_size=dino.get("input_size", ClassifierConfig.dino_input_size),
        dino_batch_size=dino.get("batch_size", ClassifierConfig.dino_batch_size),
        dino_device=dino.get("device", ClassifierConfig.dino_device),
    )
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_classifier_train.py::test_load_classifier_config_overrides \
       tests/test_classifier_train.py::test_load_classifier_config_defaults \
       tests/test_classifier_train.py::test_load_classifier_config_strips_label_map_keys -v
```

Expected: 3 PASSED

- [ ] **Step 6: Create classifier_config.yaml**

Create `configs/classifier_config.yaml`:

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
    "JBIO wet mat ": jbio  # trailing space matches filesystem; code strips before lookup

output:
  model_dir: outputs/models/classifier
  figures_dir: outputs/figures
  embedding_cache: outputs/features/classifier_embeddings.npy
  label_cache: outputs/features/classifier_labels.npy
```

- [ ] **Step 7: Commit**

```bash
git add src/classification/__init__.py src/classification/train.py \
        configs/classifier_config.yaml tests/test_classifier_train.py
git commit -m "feat: add ClassifierConfig + load_classifier_config"
```

---

## Task 2: scan_labeled_images

**Files:**
- Modify: `src/classification/train.py` — add `scan_labeled_images()`
- Modify: `tests/test_classifier_train.py` — add scan tests

- [ ] **Step 1: Write failing scan tests**

Append to `tests/test_classifier_train.py`:

```python
# ---------------------------------------------------------------------------
# scan_labeled_images tests
# ---------------------------------------------------------------------------

import tempfile
import os
from src.classification.train import scan_labeled_images


def _make_fake_image_tree(tmp_path: Path) -> dict[str, str]:
    """Create a fake checked-images structure with dummy files."""
    label_map = {
        "Class A": "cat_a",
        "Class B": "cat_b",
    }
    # Flat files
    (tmp_path / "Class A").mkdir()
    (tmp_path / "Class A" / "img1.jpg").write_bytes(b"fake")
    (tmp_path / "Class A" / "img2.jpeg").write_bytes(b"fake")
    (tmp_path / "Class A" / "readme.txt").write_bytes(b"ignore")  # non-image
    # Nested subdir (like JBIO mat)
    (tmp_path / "Class B").mkdir()
    (tmp_path / "Class B" / "sub").mkdir()
    (tmp_path / "Class B" / "sub" / "img3.png").write_bytes(b"fake")
    (tmp_path / "Class B" / "img4.JPG").write_bytes(b"fake")  # uppercase ext
    return label_map


def test_scan_labeled_images_count(tmp_path):
    label_map = _make_fake_image_tree(tmp_path)
    result = scan_labeled_images(tmp_path, label_map)
    # 2 from Class A (txt excluded) + 2 from Class B
    assert len(result) == 4


def test_scan_labeled_images_labels(tmp_path):
    label_map = _make_fake_image_tree(tmp_path)
    result = scan_labeled_images(tmp_path, label_map)
    labels = {lbl for _, lbl in result}
    assert labels == {"cat_a", "cat_b"}


def test_scan_labeled_images_strips_folder_name(tmp_path):
    # Folder name with trailing space should match stripped label_map key
    folder = tmp_path / "Class C "
    folder.mkdir()
    (folder / "img.jpg").write_bytes(b"fake")
    label_map = {"Class C": "cat_c"}  # stripped key
    result = scan_labeled_images(tmp_path, label_map)
    assert len(result) == 1
    assert result[0][1] == "cat_c"


def test_scan_labeled_images_skips_unknown_folder(tmp_path, caplog):
    (tmp_path / "Unknown").mkdir()
    (tmp_path / "Unknown" / "img.jpg").write_bytes(b"fake")
    label_map = {"Known": "known"}
    import logging
    with caplog.at_level(logging.WARNING):
        result = scan_labeled_images(tmp_path, label_map)
    assert len(result) == 0
    assert any("Unknown" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_classifier_train.py::test_scan_labeled_images_count -v
```

Expected: `ImportError` or `AttributeError` — `scan_labeled_images` not yet defined

- [ ] **Step 3: Implement scan_labeled_images in train.py**

Add after `load_classifier_config` in `src/classification/train.py`:

```python
def scan_labeled_images(
    image_dir: Path,
    label_map: dict[str, str],
) -> list[tuple[Path, str]]:
    """Scan image_dir recursively and assign class labels from top-level folder names.

    The top-level folder name (relative to image_dir) is stripped of
    leading/trailing whitespace before lookup in label_map. Files with
    extensions not in IMAGE_EXTENSIONS are skipped with a warning.

    Args:
        image_dir: Root directory containing one sub-folder per class.
        label_map: Mapping of (stripped) folder name → class label.

    Returns:
        List of (image_path, class_label) tuples.
    """
    image_dir = Path(image_dir)
    result: list[tuple[Path, str]] = []

    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            if not path.name.startswith("."):  # silently skip hidden files like .DS_Store
                logger.warning("Skipping non-image file: %s", path)
            continue

        # Determine top-level folder relative to image_dir
        rel = path.relative_to(image_dir)
        top_folder = rel.parts[0].strip()

        if top_folder not in label_map:
            logger.warning("No label mapping for folder '%s', skipping %s", top_folder, path)
            continue

        result.append((path, label_map[top_folder]))

    return result
```

- [ ] **Step 4: Run scan tests**

```bash
pytest tests/test_classifier_train.py -k "scan" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/classification/train.py tests/test_classifier_train.py
git commit -m "feat: add scan_labeled_images with extension filter and strip"
```

---

## Task 3: CVResult dataclass + ClassifierTrainer.evaluate()

**Files:**
- Modify: `src/classification/train.py` — add CVResult, ClassifierTrainer.__init__, evaluate()
- Modify: `tests/test_classifier_train.py` — add evaluate tests

- [ ] **Step 1: Write failing evaluate tests**

Append to `tests/test_classifier_train.py`:

```python
# ---------------------------------------------------------------------------
# ClassifierTrainer.evaluate() tests
# ---------------------------------------------------------------------------

import numpy as np
from src.classification.train import ClassifierConfig, ClassifierTrainer, CVResult


def _make_synthetic_data(n: int = 90) -> tuple[np.ndarray, list[str]]:
    """30 samples per class, 768-d embeddings with a clear linear signal."""
    rng = np.random.default_rng(42)
    classes = ["mudcrack", "big_pool", "jbio"]
    X_parts, y_parts = [], []
    for i, cls in enumerate(classes):
        center = np.zeros(768)
        center[i * 10 : i * 10 + 10] = 5.0  # strong signal per class
        X_parts.append(rng.normal(center, 0.1, size=(n // 3, 768)))
        y_parts.extend([cls] * (n // 3))
    return np.vstack(X_parts).astype(np.float32), y_parts


def test_evaluate_returns_cvresult():
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig(cv_folds=3)
    trainer = ClassifierTrainer(X, y, cfg)
    result = trainer.evaluate()
    assert isinstance(result, CVResult)


def test_evaluate_cvresult_fields():
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig(cv_folds=3)
    trainer = ClassifierTrainer(X, y, cfg)
    result = trainer.evaluate()
    assert result.class_names == sorted({"mudcrack", "big_pool", "jbio"})
    assert len(result.confusion_matrices) == 3  # one per fold
    assert result.confusion_matrices[0].shape == (3, 3)
    assert 0.0 <= result.macro_f1_mean <= 1.0
    assert result.macro_f1_std >= 0.0
    assert -1.0 <= result.cohen_kappa_mean <= 1.0
    assert isinstance(result.report, dict)


def test_evaluate_separable_data_high_f1():
    """Linearly separable data should yield near-perfect CV F1."""
    X, y = _make_synthetic_data(n=90)
    cfg = ClassifierConfig(cv_folds=3, random_state=42)
    trainer = ClassifierTrainer(X, y, cfg)
    result = trainer.evaluate()
    assert result.macro_f1_mean > 0.90
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_classifier_train.py -k "evaluate" -v
```

Expected: `ImportError` — `CVResult`, `ClassifierTrainer` not yet defined

- [ ] **Step 3: Implement CVResult and ClassifierTrainer.evaluate() in train.py**

Append to `src/classification/train.py` (add imports at top of file, then append classes):

Add to imports section at top of file:

```python
from dataclasses import dataclass, field
# (already present — add these new imports below existing ones)
```

Add after `scan_labeled_images`:

```python
# ---------------------------------------------------------------------------
# CV result
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Results from stratified k-fold cross-validation."""

    report: dict                          # sklearn classification_report as dict
    confusion_matrices: list[np.ndarray]  # one per fold (raw counts), shape (n_classes, n_classes)
    class_names: list[str]                # label order for confusion matrix axes (from LabelEncoder.classes_)
    macro_f1_mean: float
    macro_f1_std: float
    cohen_kappa_mean: float
    cohen_kappa_std: float


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ClassifierTrainer:
    """Train and evaluate a logistic regression classifier on embedding features.

    Args:
        X: Feature matrix, shape (N, D).
        y: Class labels, length N.
        config: Classifier configuration.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: list[str],
        config: ClassifierConfig,
    ) -> None:
        self.X = X
        self.y = y
        self.config = config
        self._model = None
        self._le = None

    def evaluate(self) -> CVResult:
        """Run stratified k-fold cross-validation and return metrics.

        Each fold fits a fresh LogisticRegression on the train split and
        evaluates on the held-out split. The returned CVResult contains
        per-fold confusion matrices (raw counts), macro-F1 mean/std, and
        Cohen's kappa mean/std.

        Returns:
            CVResult with aggregated cross-validation statistics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            classification_report,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        le.fit(self.y)
        class_names: list[str] = list(le.classes_)

        skf = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        confusion_matrices: list[np.ndarray] = []
        f1_scores: list[float] = []
        kappa_scores: list[float] = []
        all_true: list[str] = []
        all_pred: list[str] = []

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train = [self.y[i] for i in train_idx]
            y_val = [self.y[i] for i in val_idx]

            model = LogisticRegression(
                max_iter=self.config.max_iter,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
            )
            model.fit(X_train, y_train)
            y_pred = list(model.predict(X_val))

            cm = confusion_matrix(y_val, y_pred, labels=class_names)
            confusion_matrices.append(cm)
            f1_scores.append(f1_score(y_val, y_pred, average="macro"))
            kappa_scores.append(cohen_kappa_score(y_val, y_pred))
            all_true.extend(y_val)
            all_pred.extend(y_pred)

        report = classification_report(all_true, all_pred, output_dict=True)

        return CVResult(
            report=report,
            confusion_matrices=confusion_matrices,
            class_names=class_names,
            macro_f1_mean=float(np.mean(f1_scores)),
            macro_f1_std=float(np.std(f1_scores)),
            cohen_kappa_mean=float(np.mean(kappa_scores)),
            cohen_kappa_std=float(np.std(kappa_scores)),
        )
```

- [ ] **Step 4: Run evaluate tests**

```bash
pytest tests/test_classifier_train.py -k "evaluate" -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run all train tests so far**

```bash
pytest tests/test_classifier_train.py -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add src/classification/train.py tests/test_classifier_train.py
git commit -m "feat: add CVResult and ClassifierTrainer.evaluate()"
```

---

## Task 4: ClassifierTrainer.fit() + save()

**Files:**
- Modify: `src/classification/train.py` — add fit() and save()
- Modify: `tests/test_classifier_train.py` — add fit/save tests

- [ ] **Step 1: Write failing fit/save tests**

Append to `tests/test_classifier_train.py`:

```python
# ---------------------------------------------------------------------------
# ClassifierTrainer.fit() + save() tests
# ---------------------------------------------------------------------------

import json
import tempfile
import joblib
from pathlib import Path


def test_fit_sets_model_attribute():
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig()
    trainer = ClassifierTrainer(X, y, cfg)
    assert trainer._model is None
    trainer.fit()
    assert trainer._model is not None


def test_fit_model_has_classes():
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig()
    trainer = ClassifierTrainer(X, y, cfg)
    trainer.fit()
    assert set(trainer._model.classes_) == {"mudcrack", "big_pool", "jbio"}


def test_save_creates_artifacts(tmp_path):
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig()
    trainer = ClassifierTrainer(X, y, cfg)
    trainer.fit()
    trainer.save(tmp_path)

    assert (tmp_path / "lr_classifier.joblib").exists()
    assert (tmp_path / "label_encoder.joblib").exists()
    assert (tmp_path / "config.json").exists()


def test_save_config_json_content(tmp_path):
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig(cv_folds=3)
    trainer = ClassifierTrainer(X, y, cfg)
    trainer.fit()
    trainer.save(tmp_path)

    with open(tmp_path / "config.json") as f:
        saved = json.load(f)
    assert saved["model_type"] == "logistic_regression"
    assert set(saved["classes"]) == {"mudcrack", "big_pool", "jbio"}


def test_save_model_is_loadable(tmp_path):
    X, y = _make_synthetic_data()
    cfg = ClassifierConfig()
    trainer = ClassifierTrainer(X, y, cfg)
    trainer.fit()
    trainer.save(tmp_path)

    loaded_model = joblib.load(tmp_path / "lr_classifier.joblib")
    preds = loaded_model.predict(X[:3])
    assert len(preds) == 3
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_classifier_train.py -k "fit or save" -v
```

Expected: `AttributeError` — `fit` / `save` not yet defined

- [ ] **Step 3: Implement fit() and save() in ClassifierTrainer**

Add these methods inside `ClassifierTrainer` in `src/classification/train.py`:

```python
    def fit(self) -> None:
        """Fit logistic regression on the full dataset.

        Stores the fitted model as ``self._model`` and a LabelEncoder as
        ``self._le``. Call :meth:`save` afterwards to persist.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder()
        self._le.fit(self.y)

        self._model = LogisticRegression(
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
        )
        self._model.fit(self.X, self.y)
        logger.info(
            "Fitted LogisticRegression on %d samples, classes: %s",
            len(self.y),
            list(self._model.classes_),
        )

    def save(self, output_dir: Path) -> None:
        """Persist fitted model, label encoder, and config to output_dir.

        Args:
            output_dir: Directory to write artifacts into (created if needed).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        import json
        import joblib

        if self._model is None:
            raise RuntimeError("Call fit() before save()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, output_dir / "lr_classifier.joblib")
        joblib.dump(self._le, output_dir / "label_encoder.joblib")

        config_data = {
            "model_type": self.config.model_type,
            "cv_folds": self.config.cv_folds,
            "random_state": self.config.random_state,
            "max_iter": self.config.max_iter,
            "class_weight": self.config.class_weight,
            "classes": list(self._model.classes_),
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info("Classifier saved to %s", output_dir)
```

- [ ] **Step 4: Run fit/save tests**

```bash
pytest tests/test_classifier_train.py -k "fit or save" -v
```

Expected: 5 PASSED

- [ ] **Step 5: Run full train test suite**

```bash
pytest tests/test_classifier_train.py -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add src/classification/train.py tests/test_classifier_train.py
git commit -m "feat: add ClassifierTrainer.fit() and save()"
```

---

## Task 5: ClassifierPredictor

**Files:**
- Create: `src/classification/predict.py`
- Create: `tests/test_classifier_predict.py`

- [ ] **Step 1: Write failing predict tests**

Create `tests/test_classifier_predict.py`:

```python
"""Tests for src/classification/predict.py — ClassifierPredictor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def _make_mock_model_dir(tmp_path: Path) -> Path:
    """Persist a tiny real sklearn model to tmp_path."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 768)).astype(np.float32)
    y = ["mudcrack"] * 10 + ["big_pool"] * 10 + ["jbio"] * 10

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)

    le = LabelEncoder()
    le.fit(y)

    joblib.dump(model, tmp_path / "lr_classifier.joblib")
    joblib.dump(le, tmp_path / "label_encoder.joblib")

    config = {"model_type": "logistic_regression", "classes": list(model.classes_)}
    (tmp_path / "config.json").write_text(json.dumps(config))

    return tmp_path


def test_predictor_loads_from_dir(tmp_path):
    model_dir = _make_mock_model_dir(tmp_path)
    from src.classification.predict import ClassifierPredictor
    predictor = ClassifierPredictor(model_dir)
    assert predictor._model is not None


def test_predictor_raises_on_missing_dir(tmp_path):
    from src.classification.predict import ClassifierPredictor
    with pytest.raises(FileNotFoundError):
        ClassifierPredictor(tmp_path / "nonexistent")


def test_predict_from_embedding_returns_tuple(tmp_path):
    model_dir = _make_mock_model_dir(tmp_path)
    from src.classification.predict import ClassifierPredictor
    predictor = ClassifierPredictor(model_dir)
    rng = np.random.default_rng(1)
    embedding = rng.standard_normal(768).astype(np.float32)
    class_name, prob, all_probs = predictor.predict_from_embedding(embedding)
    assert isinstance(class_name, str)
    assert class_name in {"mudcrack", "big_pool", "jbio"}
    assert 0.0 <= prob <= 1.0
    assert set(all_probs.keys()) == {"mudcrack", "big_pool", "jbio"}
    assert abs(sum(all_probs.values()) - 1.0) < 1e-5


def test_predict_from_embedding_argmax_consistent(tmp_path):
    """class_name returned should match the key with the highest probability."""
    model_dir = _make_mock_model_dir(tmp_path)
    from src.classification.predict import ClassifierPredictor
    predictor = ClassifierPredictor(model_dir)
    rng = np.random.default_rng(7)
    embedding = rng.standard_normal(768).astype(np.float32)
    class_name, prob, all_probs = predictor.predict_from_embedding(embedding)
    best_from_dict = max(all_probs, key=all_probs.__getitem__)
    assert class_name == best_from_dict
    assert abs(prob - all_probs[class_name]) < 1e-6


def test_predict_requires_torch(tmp_path):
    """predict() (not predict_from_embedding) requires torch — skip if absent."""
    torch = pytest.importorskip("torch")
    # If torch is available, just verify the method exists and is callable.
    model_dir = _make_mock_model_dir(tmp_path)
    from src.classification.predict import ClassifierPredictor
    predictor = ClassifierPredictor(model_dir)
    assert callable(predictor.predict)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_classifier_predict.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.classification.predict'`

- [ ] **Step 3: Create predict.py**

Create `src/classification/predict.py`:

```python
"""Supervised pattern classifier predictor — loads persisted model and classifies images."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

if TYPE_CHECKING:
    from src.features.dino_embeddings import DinoFeatureExtractor

logger = logging.getLogger(__name__)


class ClassifierPredictor:
    """Classify new images using a persisted logistic regression model.

    Loads ``lr_classifier.joblib`` and ``label_encoder.joblib`` from
    ``model_dir`` on construction. DINOv3 extraction is lazy: the model is
    not loaded until :meth:`predict` is first called.

    Args:
        model_dir: Directory produced by :meth:`ClassifierTrainer.save`.
    """

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self._model = joblib.load(model_dir / "lr_classifier.joblib")
        self._le = joblib.load(model_dir / "label_encoder.joblib")
        self._extractor: DinoFeatureExtractor | None = None

        logger.info(
            "ClassifierPredictor loaded from %s, classes: %s",
            model_dir,
            list(self._model.classes_),
        )

    def predict_from_embedding(
        self,
        embedding: np.ndarray,
    ) -> tuple[str, float, dict[str, float]]:
        """Classify a pre-computed DINOv3 embedding.

        Args:
            embedding: 1-D float32 array of shape ``(768,)``.

        Returns:
            Tuple of (class_name, probability, all_class_probs) where
            all_class_probs maps each class name to its predicted probability.
        """
        emb = embedding.reshape(1, -1)
        probs = self._model.predict_proba(emb)[0]
        class_names: list[str] = list(self._model.classes_)
        best_idx = int(np.argmax(probs))
        return (
            class_names[best_idx],
            float(probs[best_idx]),
            {cls: float(p) for cls, p in zip(class_names, probs)},
        )

    def predict(
        self,
        image_path: Path,
    ) -> tuple[str, float, dict[str, float]]:
        """Extract a DINOv3 embedding and classify an image.

        Requires ``torch`` and ``transformers`` (installed via
        ``pip install -e ".[ml]"``).

        Args:
            image_path: Path to a JPEG or PNG image file.

        Returns:
            Tuple of (class_name, probability, all_class_probs).
        """
        from PIL import Image

        from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor

        if self._extractor is None:
            self._extractor = DinoFeatureExtractor()

        img = Image.open(image_path).convert("RGB")
        embedding = self._extractor.extract_cls(img)
        return self.predict_from_embedding(embedding)
```

- [ ] **Step 4: Run predict tests**

```bash
pytest tests/test_classifier_predict.py -v
```

Expected: 4 PASSED (test_predict_requires_torch skipped if torch not installed, otherwise PASSED)

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/test_classifier_train.py tests/test_classifier_predict.py -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add src/classification/predict.py tests/test_classifier_predict.py
git commit -m "feat: add ClassifierPredictor with predict_from_embedding and predict"
```

---

## Task 6: Training script

**Files:**
- Create: `scripts/train_classifier.py`

- [ ] **Step 1: Create the training script**

Create `scripts/train_classifier.py`:

```python
#!/usr/bin/env python3
"""Train a logistic regression classifier on DINOv3 embeddings.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --force        # re-extract + re-train
    python scripts/train_classifier.py --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.classification.train import (
    ClassifierTrainer,
    load_classifier_config,
    scan_labeled_images,
)
from src.features.dino_embeddings import DinoConfig, DinoFeatureExtractor
from src.visualization.style import (
    WONG_PALETTE,
    FigureStyle,
    save_figure,
    setup_matplotlib_style,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_or_load_embeddings(
    config,
    image_list: list,
    force: bool,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, y) — from cache if available and force=False."""
    cache_emb = Path(config.embedding_cache)
    cache_lbl = Path(config.label_cache)

    if not force and cache_emb.exists() and cache_lbl.exists():
        logger.info("Loading cached embeddings from %s", cache_emb)
        X = np.load(cache_emb)
        y = list(np.load(cache_lbl))
        if len(X) != len(y):
            raise ValueError(
                f"Embedding cache shape mismatch: {len(X)} embeddings vs {len(y)} labels. "
                "Delete cache files or re-run with --force."
            )
        logger.info("Loaded %d cached embeddings", len(X))
        return X, y

    paths = [p for p, _ in image_list]
    labels = [lbl for _, lbl in image_list]

    dino_cfg = DinoConfig(
        model_name=config.dino_model_name,
        input_size=config.dino_input_size,
        batch_size=config.dino_batch_size,
        device=config.dino_device,
    )
    extractor = DinoFeatureExtractor(dino_cfg)
    X = extractor.extract_batch(paths, batch_size=config.dino_batch_size)
    y = labels

    cache_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_emb, X)
    np.save(cache_lbl, np.array(y))
    logger.info("Embeddings cached to %s (%d images)", cache_emb, len(X))

    return X, y


def _save_confusion_matrix(cv_result, figures_dir: Path, style: FigureStyle) -> None:
    """Save sum-of-folds row-normalised confusion matrix as PNG+SVG."""
    from sklearn.metrics import ConfusionMatrixDisplay

    cm_sum = sum(cv_result.confusion_matrices)
    row_sums = cm_sum.sum(axis=1, keepdims=True)
    cm_norm = cm_sum / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=cv_result.class_names,
    )
    disp.plot(ax=ax, cmap="viridis", values_format=".2f", colorbar=True)
    ax.set_title("Cross-validated confusion matrix (row-normalised)")

    out = figures_dir / "classifier_confusion_matrix"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("Confusion matrix saved to %s", out)


def _save_umap_plot(
    X: np.ndarray,
    y: list[str],
    figures_dir: Path,
    style: FigureStyle,
) -> None:
    """Fit UMAP on embeddings and save scatter coloured by class (Wong palette)."""
    import umap

    logger.info("Fitting UMAP for visualisation…")
    reducer = umap.UMAP(
        metric="cosine",
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(X)

    classes = sorted(set(y))
    colors = {cls: WONG_PALETTE[i % len(WONG_PALETTE)] for i, cls in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_width * 0.85))
    for cls in classes:
        mask = np.array([lbl == cls for lbl in y])
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=colors[cls],
            label=cls,
            s=20,
            alpha=0.7,
            linewidths=0,
        )
    ax.legend(frameon=False)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("DINOv3 embeddings by class")

    out = figures_dir / "classifier_umap"
    save_figure(fig, out, formats=style.export_formats, dpi=style.dpi)
    logger.info("UMAP plot saved to %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pattern classifier")
    parser.add_argument(
        "--config",
        default="configs/classifier_config.yaml",
        help="Path to classifier config YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract embeddings and re-train model even if cached",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config = load_classifier_config(config_dict)

    # 2. Scan images
    image_dir = Path(config.image_dir)
    image_list = scan_labeled_images(image_dir, config.label_map)
    class_counts = {}
    for _, lbl in image_list:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    logger.info("Found %d images: %s", len(image_list), class_counts)

    # 3. Extract / load embeddings
    X, y = _extract_or_load_embeddings(config, image_list, args.force)

    # 4. Evaluate with CV
    trainer = ClassifierTrainer(X, y, config)
    cv_result = trainer.evaluate()
    logger.info(
        "CV macro-F1: %.3f ± %.3f | Cohen κ: %.3f ± %.3f",
        cv_result.macro_f1_mean,
        cv_result.macro_f1_std,
        cv_result.cohen_kappa_mean,
        cv_result.cohen_kappa_std,
    )

    # 5. Figures
    style = FigureStyle()
    setup_matplotlib_style(style)
    figures_dir = Path(config.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_confusion_matrix(cv_result, figures_dir, style)
    _save_umap_plot(X, y, figures_dir, style)

    # 6. Fit + save final model
    model_path = Path(config.model_dir) / "lr_classifier.joblib"
    if not args.force and model_path.exists():
        logger.info("Model already exists at %s — skipping fit (use --force to retrain)", model_path)
    else:
        trainer.fit()
        trainer.save(Path(config.model_dir))

    logger.info(
        "Done. Figures → %s  |  Model → %s",
        config.figures_dir,
        config.model_dir,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable (no syntax errors)**

```bash
python -c "import scripts.train_classifier" 2>&1 || python scripts/train_classifier.py --help
```

Expected: prints usage/help with no errors

- [ ] **Step 3: Run the full pipeline (requires torch + real images)**

This step requires the `[ml]` extras and the real image directory. Run only when torch is available:

```bash
python scripts/train_classifier.py --verbose
```

Expected log output (approximate):
```
Found 367 images: {'mudcrack': 98, 'big_pool': 183, 'jbio': 86}
Loading DINOv3 model … on device=mps
DINOv3 batch 1/...
Embeddings cached to outputs/features/classifier_embeddings.npy
CV macro-F1: 0.8XX ± 0.0XX | Cohen κ: 0.7XX ± 0.0XX
Confusion matrix saved to outputs/figures/classifier_confusion_matrix
UMAP plot saved to outputs/figures/classifier_umap
Classifier saved to outputs/models/classifier
Done.
```

Verify output artifacts:
```bash
ls outputs/models/classifier/
# lr_classifier.joblib  label_encoder.joblib  config.json
ls outputs/figures/classifier*
# classifier_confusion_matrix.png  classifier_confusion_matrix.svg
# classifier_umap.png  classifier_umap.svg
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_classifier.py
git commit -m "feat: add train_classifier.py script with embed/eval/fit/figures pipeline"
```

---

## Task 7: Final test run + cleanup

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/test_classifier_train.py tests/test_classifier_predict.py -v
```

Expected: all PASSED (torch-dependent tests skipped or passed depending on environment)

- [ ] **Step 2: Run existing project tests to check for regressions**

```bash
pytest tests/ -v --ignore=tests/test_umap_segfault.py -q
```

Expected: no new failures

- [ ] **Step 3: Final commit**

```bash
git add -A
git status  # verify nothing unexpected is staged
git commit -m "feat: supervised pattern classifier complete (Task 7 cleanup)"
```

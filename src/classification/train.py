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
    dino, data, output) and fans out sub-dicts internally.  Label map keys
    are stripped of leading/trailing whitespace to handle filesystem quirks.

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


# ---------------------------------------------------------------------------
# Label scanning
# ---------------------------------------------------------------------------


def scan_labeled_images(
    image_dir: Path,
    label_map: dict[str, str],
) -> list[tuple[Path, str]]:
    """Scan image_dir recursively and assign class labels from top-level folder names.

    The top-level folder name (relative to image_dir) is stripped of
    leading/trailing whitespace before lookup in label_map.  Files with
    extensions not in IMAGE_EXTENSIONS are skipped with a warning.
    Hidden files (e.g. .DS_Store) are silently ignored.

    Args:
        image_dir: Root directory containing one sub-folder per class.
        label_map: Mapping of (stripped) folder name → class label.

    Returns:
        List of (image_path, class_label) tuples, sorted by path.
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
            logger.warning(
                "No label mapping for folder '%s', skipping %s", top_folder, path
            )
            continue

        result.append((path, label_map[top_folder]))

    return result


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
        evaluates on the held-out split.  The returned CVResult contains
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

    def fit(self) -> None:
        """Fit logistic regression on the full dataset.

        Stores the fitted model as ``self._model`` and a LabelEncoder as
        ``self._le``.  Call :meth:`save` afterwards to persist.
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

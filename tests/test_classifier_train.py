"""Tests for src/classification/train.py."""
from __future__ import annotations

from pathlib import Path

import pytest
from src.classification.train import ClassifierConfig, load_classifier_config


def _minimal_config() -> dict:
    return {
        "classifier": {"cv_folds": 3, "max_iter": 500},
        "dino": {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "input_size": 518,
            "batch_size": 16,
            "device": "cpu",
        },
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


# ---------------------------------------------------------------------------
# scan_labeled_images tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ClassifierTrainer.evaluate() tests
# ---------------------------------------------------------------------------

import numpy as np
from src.classification.train import ClassifierTrainer, CVResult


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


# ---------------------------------------------------------------------------
# ClassifierTrainer.fit() + save() tests
# ---------------------------------------------------------------------------

import json
import joblib


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

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
    pytest.importorskip("torch")
    # If torch is available, just verify the method exists and is callable.
    model_dir = _make_mock_model_dir(tmp_path)
    from src.classification.predict import ClassifierPredictor
    predictor = ClassifierPredictor(model_dir)
    assert callable(predictor.predict)

"""Tests for HDF5 feature store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.features.feature_store import FeatureStore


@pytest.fixture
def store(tmp_path: Path) -> FeatureStore:
    """Create a FeatureStore in a temp directory."""
    return FeatureStore(tmp_path / "test_store.h5")


@pytest.fixture
def populated_store(store: FeatureStore) -> FeatureStore:
    """Store with image_ids, dino, and texture data saved."""
    ids = ["img_001", "img_002", "img_003"]
    dino = np.random.RandomState(42).randn(3, 768).astype(np.float32)
    texture = np.random.RandomState(43).randn(3, 90).astype(np.float32)
    names = [f"feat_{i}" for i in range(90)]

    store.save_image_ids(ids)
    store.save_dino(dino)
    store.save_texture(texture, names)
    return store


# ---------------------------------------------------------------------------
# Save/load roundtrips
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """Verify save then load returns equivalent data."""

    def test_image_ids_roundtrip(self, store):
        ids = ["a", "b", "c"]
        store.save_image_ids(ids)
        loaded = store.load_image_ids()
        assert loaded == ids

    def test_dino_roundtrip(self, store):
        store.save_image_ids(["a", "b"])
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        store.save_dino(data)
        loaded = store.load_dino()
        np.testing.assert_array_equal(loaded, data)

    def test_texture_roundtrip(self, store):
        store.save_image_ids(["a"])
        data = np.random.randn(1, 90).astype(np.float32)
        names = [f"f{i}" for i in range(90)]
        store.save_texture(data, names)
        loaded = store.load_texture()
        np.testing.assert_array_equal(loaded, data)
        loaded_names = store.load_texture_feature_names()
        assert loaded_names == names

    def test_fused_roundtrip(self, store):
        store.save_image_ids(["a", "b"])
        fused = np.random.randn(2, 50).astype(np.float32)
        dino_pca = np.random.randn(2, 30).astype(np.float32)
        texture_pca = np.random.randn(2, 20).astype(np.float32)

        store.save_fused(fused, dino_pca, texture_pca)

        np.testing.assert_array_equal(store.load_fused(), fused)
        np.testing.assert_array_equal(store.load_dino_pca(), dino_pca)
        np.testing.assert_array_equal(store.load_texture_pca(), texture_pca)


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


class TestExistenceChecks:
    """has_* methods should reflect store contents."""

    def test_empty_store(self, store):
        assert not store.has_dino()
        assert not store.has_texture()
        assert not store.has_fused()

    def test_after_save(self, populated_store):
        assert populated_store.has_dino()
        assert populated_store.has_texture()
        assert not populated_store.has_fused()

    def test_n_images_empty(self, store):
        assert store.get_n_images() == 0

    def test_n_images_populated(self, populated_store):
        assert populated_store.get_n_images() == 3


# ---------------------------------------------------------------------------
# Overwrite / incremental
# ---------------------------------------------------------------------------


class TestOverwrite:
    """Overwriting existing datasets should replace data."""

    def test_overwrite_dino(self, store):
        store.save_image_ids(["a"])
        v1 = np.array([[1.0, 2.0]], dtype=np.float32)
        v2 = np.array([[3.0, 4.0]], dtype=np.float32)
        store.save_dino(v1)
        store.save_dino(v2)
        loaded = store.load_dino()
        np.testing.assert_array_equal(loaded, v2)

    def test_overwrite_image_ids(self, store):
        store.save_image_ids(["a", "b"])
        store.save_image_ids(["x", "y", "z"])
        assert store.load_image_ids() == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """validate() should detect common issues."""

    def test_clean_store(self, populated_store):
        issues = populated_store.validate()
        assert issues == []

    def test_missing_file(self, tmp_path):
        store = FeatureStore(tmp_path / "nonexistent.h5")
        issues = store.validate()
        assert any("does not exist" in i for i in issues)

    def test_nan_detection(self, store):
        store.save_image_ids(["a"])
        data = np.array([[float("nan"), 1.0]], dtype=np.float32)
        store.save_dino(data)
        issues = store.validate()
        assert any("NaN" in i for i in issues)

    def test_inf_detection(self, store):
        store.save_image_ids(["a"])
        data = np.array([[float("inf"), 1.0]], dtype=np.float32)
        store.save_dino(data)
        issues = store.validate()
        assert any("Inf" in i for i in issues)

    def test_shape_mismatch(self, store):
        store.save_image_ids(["a", "b"])
        # Save dino with 3 rows but only 2 image_ids
        data = np.random.randn(3, 768).astype(np.float32)
        store.save_dino(data)
        issues = store.validate()
        assert any("rows" in i for i in issues)

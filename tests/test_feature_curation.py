"""Tests for src.features.feature_curation — scene direction removal."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.feature_curation import (
    CurationConfig,
    compute_scene_directions,
    curate_features,
    load_curation_config,
    project_out_directions,
)


# =====================================================================
# TestComputeSceneDirections
# =====================================================================


class TestComputeSceneDirections:
    """Tests for compute_scene_directions()."""

    def test_single_scene_cluster(self, staged_clustering_data):
        """1 scene cluster -> 1 direction = normalized centroid diff."""
        data = staged_clustering_data
        directions, eigenvalues, scene_c, pattern_c = compute_scene_directions(
            data["features"],
            data["labels"],
            scene_cluster_ids=[10],
            n_directions=1,
        )
        assert directions.shape == (1, 50)
        assert eigenvalues.shape == (1,)
        # Direction should be unit-length
        np.testing.assert_allclose(np.linalg.norm(directions[0]), 1.0, atol=1e-10)

    def test_multiple_scene_clusters(self, staged_clustering_data):
        """3 directions requested from 2 clusters -> up to 2 directions."""
        data = staged_clustering_data
        directions, eigenvalues, _, _ = compute_scene_directions(
            data["features"],
            data["labels"],
            scene_cluster_ids=[10, 11],
            n_directions=3,
        )
        # Clamped to 2 (n_scene_clusters)
        assert directions.shape[0] <= 2
        assert directions.shape[1] == 50
        assert len(eigenvalues) == directions.shape[0]

    def test_directions_orthonormal(self, staged_clustering_data):
        """V @ V^T should approximate I for the returned directions."""
        data = staged_clustering_data
        directions, _, _, _ = compute_scene_directions(
            data["features"],
            data["labels"],
            scene_cluster_ids=[10, 11],
            n_directions=2,
        )
        if directions.shape[0] > 1:
            gram = directions @ directions.T
            np.testing.assert_allclose(gram, np.eye(directions.shape[0]), atol=1e-10)

    def test_n_directions_clamped(self, staged_clustering_data):
        """n_directions=5 with 2 clusters -> actual <= 2."""
        data = staged_clustering_data
        directions, _, _, _ = compute_scene_directions(
            data["features"],
            data["labels"],
            scene_cluster_ids=[10, 11],
            n_directions=5,
        )
        assert directions.shape[0] <= 2

    def test_empty_scene_list_raises(self, staged_clustering_data):
        """Empty scene_cluster_ids should raise ValueError."""
        data = staged_clustering_data
        with pytest.raises(ValueError, match="must not be empty"):
            compute_scene_directions(
                data["features"],
                data["labels"],
                scene_cluster_ids=[],
            )

    def test_all_scene_raises(self, staged_clustering_data):
        """All non-noise clusters as scene -> ValueError."""
        data = staged_clustering_data
        all_ids = [0, 1, 2, 3, 10, 11]
        with pytest.raises(ValueError, match="All non-noise clusters"):
            compute_scene_directions(
                data["features"],
                data["labels"],
                scene_cluster_ids=all_ids,
            )

    def test_missing_cluster_id_skipped(self, staged_clustering_data):
        """Scene ID not in labels -> skip it, remaining IDs still work."""
        data = staged_clustering_data
        # ID 99 doesn't exist — should log warning but not crash
        directions, _, _, _ = compute_scene_directions(
            data["features"],
            data["labels"],
            scene_cluster_ids=[10, 99],
            n_directions=2,
        )
        # Only 1 valid scene cluster (10), so 1 direction
        assert directions.shape[0] == 1


# =====================================================================
# TestProjectOutDirections
# =====================================================================


class TestProjectOutDirections:
    """Tests for project_out_directions()."""

    def test_removes_component(self):
        """Dot product with removed direction should be ~0."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 50)
        direction = rng.randn(50)
        direction = direction / np.linalg.norm(direction)
        directions = direction.reshape(1, -1)

        curated = project_out_directions(features, directions)
        dots = curated @ direction
        np.testing.assert_allclose(dots, 0.0, atol=1e-10)

    def test_preserves_orthogonal(self):
        """Components orthogonal to removed direction should be unchanged."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 50)

        # Direction to remove: unit vector along dim 0
        direction = np.zeros(50)
        direction[0] = 1.0
        directions = direction.reshape(1, -1)

        curated = project_out_directions(features, directions)
        # Dims 1-49 should be identical
        np.testing.assert_allclose(curated[:, 1:], features[:, 1:], atol=1e-10)
        # Dim 0 should be zeroed
        np.testing.assert_allclose(curated[:, 0], 0.0, atol=1e-10)

    def test_idempotent(self):
        """Applying projection twice should equal applying once."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 50)
        direction = rng.randn(50)
        direction = direction / np.linalg.norm(direction)
        directions = direction.reshape(1, -1)

        once = project_out_directions(features, directions)
        twice = project_out_directions(once, directions)
        np.testing.assert_allclose(once, twice, atol=1e-10)

    def test_zero_directions(self):
        """Empty V -> features unchanged (returned as copy)."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 50)
        empty_dirs = np.empty((0, 50))

        curated = project_out_directions(features, empty_dirs)
        np.testing.assert_array_equal(curated, features)
        # Should be a copy, not same object
        assert curated is not features


# =====================================================================
# TestCurateFeatures
# =====================================================================


class TestCurateFeatures:
    """Tests for curate_features() — the main entry point."""

    def test_output_shape(self, staged_clustering_data):
        """Curated features should have the same shape as input."""
        data = staged_clustering_data
        result = curate_features(
            data["features"],
            data["labels"],
            scene_cluster_ids=data["scene_ids"],
        )
        assert result.curated_features.shape == data["features"].shape

    def test_variance_fraction_positive(self, staged_clustering_data):
        """total_variance_fraction should be > 0 for real scene clusters."""
        data = staged_clustering_data
        result = curate_features(
            data["features"],
            data["labels"],
            scene_cluster_ids=data["scene_ids"],
        )
        assert result.total_variance_fraction > 0

    def test_scene_patches_move_more(self, staged_clustering_data):
        """Scene cluster patches should be displaced further than pattern patches."""
        data = staged_clustering_data
        result = curate_features(
            data["features"],
            data["labels"],
            scene_cluster_ids=data["scene_ids"],
        )

        displacement = np.linalg.norm(
            result.curated_features - data["features"], axis=1
        )

        scene_mask = np.isin(data["labels"], data["scene_ids"])
        pattern_mask = np.isin(data["labels"], data["keep_ids"])

        mean_scene_disp = displacement[scene_mask].mean()
        mean_pattern_disp = displacement[pattern_mask].mean()

        assert mean_scene_disp > mean_pattern_disp

    def test_deterministic(self, staged_clustering_data):
        """Same input -> same output."""
        data = staged_clustering_data
        r1 = curate_features(
            data["features"],
            data["labels"],
            scene_cluster_ids=data["scene_ids"],
        )
        r2 = curate_features(
            data["features"],
            data["labels"],
            scene_cluster_ids=data["scene_ids"],
        )
        np.testing.assert_array_equal(
            r1.curated_features, r2.curated_features
        )

    def test_with_noise_labels(self, staged_clustering_data):
        """Noise labels (-1) should be excluded from centroid computation."""
        data = staged_clustering_data
        labels = data["labels"].copy()
        # Mark first 5 samples as noise
        labels[:5] = -1

        result = curate_features(
            data["features"],
            labels,
            scene_cluster_ids=data["scene_ids"],
        )
        # Should still work — noise is ignored
        assert result.curated_features.shape == data["features"].shape
        assert result.n_directions_removed > 0


# =====================================================================
# TestCurationConfig
# =====================================================================


class TestCurationConfig:
    """Tests for CurationConfig and load_curation_config."""

    def test_default_config(self):
        """Default config values are correct."""
        cfg = CurationConfig()
        assert cfg.n_directions == 3
        assert cfg.method == "centroid_pca"
        assert cfg.min_eigenvalue == 1e-10

    def test_load_from_dict(self):
        """Custom values from dict."""
        cfg = load_curation_config({
            "n_directions": 5,
            "method": "centroid_pca",
            "min_eigenvalue": 1e-8,
        })
        assert cfg.n_directions == 5
        assert cfg.min_eigenvalue == 1e-8

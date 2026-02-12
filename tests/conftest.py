"""Shared test fixtures for desert_patterns test suite."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs


@pytest.fixture
def project_root() -> Path:
    """Root directory of the project."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_data_root(project_root: Path) -> Path:
    """Root of the raw image data directory."""
    return project_root / "data" / "kim_2023"


@pytest.fixture
def sample_drone_jpg(sample_data_root: Path) -> Path:
    """A sample drone JPG file for testing. Skips if data unavailable."""
    path = sample_data_root / "Kim_BigPool_Drone" / "Big Pool 1 1130 am" / "1 m DJI_0904.JPG"
    if not path.exists():
        pytest.skip("Sample data not available")
    return path


@pytest.fixture
def sample_ground_jpg(sample_data_root: Path) -> Path:
    """A sample ground-level JPG file for testing. Skips if data unavailable."""
    path = sample_data_root / "Kim_BigPool_Ground" / "Kim_Playa_11am" / "IMG_8143.jpg"
    if not path.exists():
        pytest.skip("Sample data not available")
    return path


# ---- Phase 2 synthetic images ----


@pytest.fixture
def synthetic_drone_image() -> np.ndarray:
    """Synthetic 4000x3000 BGR drone image with varied texture.

    Contains a grid of random-intensity blocks to ensure non-trivial
    tile statistics (non-zero contrast, edges, etc.).
    """
    rng = np.random.RandomState(42)
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    block_h, block_w = 500, 500
    for y in range(0, 3000, block_h):
        for x in range(0, 4000, block_w):
            colour = rng.randint(40, 220, size=3).tolist()
            img[y : y + block_h, x : x + block_w] = colour
    return img


@pytest.fixture
def synthetic_ground_image() -> np.ndarray:
    """Synthetic 640x480 BGR ground image with sky/ground boundary.

    Top 40% is pale blue (sky), bottom 60% is brown (ground) with
    a sharp horizontal edge between them to test horizon detection.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boundary = int(480 * 0.4)  # y=192
    # Sky (BGR: light blue)
    img[:boundary, :] = [230, 200, 160]
    # Ground (BGR: brown/tan)
    img[boundary:, :] = [60, 100, 140]
    return img


@pytest.fixture
def synthetic_drone_jpg(tmp_path: Path, synthetic_drone_image: np.ndarray) -> Path:
    """Write synthetic drone image to a temp JPG file."""
    path = tmp_path / "drone_test.jpg"
    cv2.imwrite(str(path), synthetic_drone_image)
    return path


@pytest.fixture
def synthetic_ground_jpg(tmp_path: Path, synthetic_ground_image: np.ndarray) -> Path:
    """Write synthetic ground image to a temp JPG file."""
    path = tmp_path / "ground_test.jpg"
    cv2.imwrite(str(path), synthetic_ground_image)
    return path


# ---- Phase 3 synthetic images ----


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """518x518 grayscale checkerboard with high contrast/edges.

    Alternating 37x37-ish blocks of 0 and 255.
    """
    img = np.zeros((518, 518), dtype=np.uint8)
    block = 37
    for y in range(0, 518, block):
        for x in range(0, 518, block):
            if ((y // block) + (x // block)) % 2 == 0:
                img[y : y + block, x : x + block] = 255
    return img


@pytest.fixture
def uniform_gray_image() -> np.ndarray:
    """518x518 uniform gray (128) image — zero contrast/max energy."""
    return np.full((518, 518), 128, dtype=np.uint8)


@pytest.fixture
def gradient_image() -> np.ndarray:
    """518x518 horizontal gradient from 0 to 255."""
    row = np.linspace(0, 255, 518, dtype=np.uint8)
    return np.tile(row, (518, 1))


@pytest.fixture
def circle_image() -> np.ndarray:
    """518x518 image with a white circle on black background.

    Fractal dimension of a circle boundary should be ~1.0.
    """
    img = np.zeros((518, 518), dtype=np.uint8)
    cv2.circle(img, (259, 259), 200, 255, thickness=2)
    return img


@pytest.fixture
def synthetic_518_bgr() -> np.ndarray:
    """518x518 BGR image with varied texture for feature extraction tests."""
    rng = np.random.RandomState(42)
    img = np.zeros((518, 518, 3), dtype=np.uint8)
    block = 130
    for y in range(0, 518, block):
        for x in range(0, 518, block):
            colour = rng.randint(40, 220, size=3).tolist()
            img[y : y + block, x : x + block] = colour
    return img


# ---- Phase 4 clustering fixtures ----


@pytest.fixture
def clusterable_features() -> np.ndarray:
    """200 samples × 50 features from 4 well-separated blobs."""
    X, _ = make_blobs(n_samples=200, n_features=50, centers=4, random_state=42)
    return X.astype(np.float32)


@pytest.fixture
def cluster_labels() -> np.ndarray:
    """Pre-computed labels for 200 samples: 4 clusters + some noise."""
    rng = np.random.RandomState(42)
    labels = np.repeat([0, 1, 2, 3], 50)
    # Mark 10 random points as noise
    noise_idx = rng.choice(200, 10, replace=False)
    labels[noise_idx] = -1
    return labels


@pytest.fixture
def mock_metadata_df() -> pd.DataFrame:
    """200-row metadata DataFrame for clustering tests."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "image_id": [f"img_{i:04d}" for i in range(n)],
        "source_type": rng.choice(["drone", "ground"], n),
        "altitude_group": rng.choice(["low", "mid", "high", "ground"], n),
        "temporal_phase": rng.choice(
            ["pre_rain", "during_rain", "post_rain", "unknown"],
            n,
            p=[0.05, 0.15, 0.10, 0.70],
        ),
        "lat": rng.uniform(-24.09, -24.08, n),
        "lon": rng.uniform(-69.92, -69.91, n),
    })


@pytest.fixture
def mock_texture_features() -> np.ndarray:
    """200 × 90 random texture features for characterization testing."""
    rng = np.random.RandomState(42)
    return rng.randn(200, 90).astype(np.float32)


# ---- Phase 5 visualization fixtures ----


@pytest.fixture
def mock_assignments_df() -> pd.DataFrame:
    """200-row assignments DataFrame matching Phase 4 output format."""
    rng = np.random.RandomState(42)
    n = 200
    labels = np.repeat([0, 1, 2, 3], 50)
    noise_idx = rng.choice(n, 10, replace=False)
    labels[noise_idx] = -1

    umap_2d = rng.randn(n, 2).astype(np.float32)
    return pd.DataFrame({
        "image_id": [f"patch_{i:04d}" for i in range(n)],
        "cluster_id": labels,
        "probability": rng.uniform(0.5, 1.0, n),
        "umap_x": umap_2d[:, 0],
        "umap_y": umap_2d[:, 1],
    })


@pytest.fixture
def mock_profiles() -> list[dict]:
    """List of cluster profile dicts matching Phase 4 JSON format."""
    profiles = []
    for cid in range(4):
        profiles.append({
            "cluster_id": cid,
            "size": 50,
            "texture_mean": list(np.random.RandomState(cid).randn(90)),
            "texture_std": list(np.abs(np.random.RandomState(cid + 10).randn(90))),
            "distinguishing_features": [
                {"name": f"texture_{j}", "value": float(j), "z_score": 2.5 - j * 0.3}
                for j in range(5)
            ],
            "metadata_distribution": {"source_type": {"drone": 0.6, "ground": 0.4}},
            "representative_ids": [f"patch_{cid * 50 + j:04d}" for j in range(10)],
            "boundary_ids": [f"patch_{cid * 50 + 40 + j:04d}" for j in range(5)],
            "description": f"Cluster {cid}: moderate texture with distinct pattern.",
        })
    return profiles


@pytest.fixture
def mock_patch_paths(tmp_path: Path) -> dict[str, Path]:
    """200 tiny PNG files with paths keyed by patch ID."""
    paths: dict[str, Path] = {}
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    for i in range(200):
        pid = f"patch_{i:04d}"
        p = tmp_path / f"{pid}.png"
        cv2.imwrite(str(p), img)
        paths[pid] = p
    return paths


@pytest.fixture
def mock_temporal_data() -> dict:
    """Temporal data dict matching Phase 4 JSON format."""
    return {
        "phase_distributions": {
            "during_rain": {"0": 0.3, "1": 0.5, "2": 0.1, "3": 0.1},
            "post_rain": {"0": 0.2, "1": 0.2, "2": 0.4, "3": 0.2},
        },
        "transition_matrix": [[0.6, 0.2, 0.1, 0.1], [0.1, 0.5, 0.3, 0.1]],
        "embedding_drift": {
            "0": {"centroids": [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]},
            "1": {"centroids": [[-1.0, 0.0], [-0.5, 0.5]]},
        },
        "test_results": [
            {"test": "chi2", "statistic": 12.5, "p_value": 0.01, "effect_size": 0.3}
        ],
    }


@pytest.fixture
def mock_continuous_space_df() -> pd.DataFrame:
    """Continuous space DataFrame matching Phase 4 CSV format."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "image_id": [f"patch_{i:04d}" for i in range(n)],
        "pattern_x": rng.randn(n),
        "pattern_y": rng.randn(n),
    })


# ---- Feature curation fixtures ----


@pytest.fixture
def staged_clustering_data():
    """200 samples with 4 pattern clusters (150) + 2 scene clusters (50).

    Features are 50-d.  Scene clusters are offset in the first 3 dimensions
    to simulate scene-content encoding (e.g. tape measure direction).
    Pattern clusters are separated only in dimensions 3+.

    Returns dict with ``features``, ``labels``, ``scene_ids``, ``keep_ids``.
    """
    rng = np.random.RandomState(42)
    n_pattern = 150  # 4 clusters of ~37-38
    n_scene = 50  # 2 clusters of 25
    d = 50

    # Pattern clusters: separated in dims 3-10
    pattern_features = rng.randn(n_pattern, d).astype(np.float64) * 0.5
    pattern_labels = np.repeat([0, 1, 2, 3], [38, 38, 37, 37])
    for cid in range(4):
        mask = pattern_labels == cid
        # Offset in dims 3+cid*2 and 4+cid*2
        pattern_features[mask, 3 + cid * 2] += 5.0
        pattern_features[mask, 4 + cid * 2] += 5.0

    # Scene clusters: offset in dims 0, 1, 2 (the "scene directions")
    scene_features = rng.randn(n_scene, d).astype(np.float64) * 0.5
    scene_labels = np.repeat([10, 11], 25)
    # Cluster 10: strong offset in dim 0
    scene_features[:25, 0] += 10.0
    scene_features[:25, 1] += 3.0
    # Cluster 11: strong offset in dim 1 and 2
    scene_features[25:, 1] += 10.0
    scene_features[25:, 2] += 5.0

    features = np.vstack([pattern_features, scene_features])
    labels = np.concatenate([pattern_labels, scene_labels])

    return {
        "features": features,
        "labels": labels,
        "scene_ids": [10, 11],
        "keep_ids": [0, 1, 2, 3],
    }

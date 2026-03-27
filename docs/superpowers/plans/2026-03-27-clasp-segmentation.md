# CLASP Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CLASP (Clustering via Adaptive Spectral Processing) — a fully unsupervised per-image segmentation alternative to the existing DINOv3-attention + SAM pipeline.

**Architecture:** `src/segmentation/clasp.py` holds `ClaspConfig`, `ClaspResult`, `load_clasp_config`, and `ClaspSegmenter`. The segmenter reuses `DinoFeatureExtractor.extract_patch_tokens()` already in the codebase, builds a cosine affinity matrix, eigendecomposes it, selects K clusters via eigengap elbow + silhouette sweep, and optionally refines with DenseCRF. `scripts/generate_clasp_segmentations.py` runs batch segmentation over all labeled images.

**Tech Stack:** `scipy.linalg.eigh` (eigendecomposition), `sklearn.cluster.KMeans` + `sklearn.metrics.silhouette_score` + `sklearn.preprocessing.normalize`, `pydensecrf` (C extension, install via git URL), `PIL`, `numpy`, `matplotlib`.

**Spec:** `docs/superpowers/specs/2026-03-27-clasp-segmentation-design.md`

**Note on pydensecrf:** The PyPI package is broken on Apple Silicon. Install from source:
```bash
pip install "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"
```
The existing `.venv` already has it installed. The `pyproject.toml` update in Task 4 uses this git URL.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/segmentation/clasp.py` | Create | `ClaspConfig`, `ClaspResult`, `load_clasp_config`, `ClaspSegmenter` |
| `configs/classifier_config.yaml` | Modify | Add `clasp:` block |
| `pyproject.toml` | Modify | Add `pydensecrf` git URL to `[ml]` extras |
| `tests/test_clasp.py` | Create | 15 unit tests, no model loading for most |
| `scripts/generate_clasp_segmentations.py` | Create | Batch script: scan images, run CLASP, save outputs |

---

## Task 1: Config, dataclasses, and config tests

**Files:**
- Create: `src/segmentation/clasp.py`
- Modify: `configs/classifier_config.yaml`
- Create: `tests/test_clasp.py`

- [ ] **Step 1: Write failing config tests**

```python
# tests/test_clasp.py
"""Tests for src/segmentation/clasp.py."""
from __future__ import annotations

import numpy as np
import pytest


def test_clasp_config_defaults():
    from src.segmentation.clasp import ClaspConfig
    cfg = ClaspConfig()
    assert cfg.output_dir == "outputs/segmentations_clasp"
    assert cfg.bandwidth == pytest.approx(0.5)
    assert cfg.min_clusters == 2
    assert cfg.max_clusters == 15
    assert cfg.dense_crf is True
    assert cfg.crf_iterations == 20
    assert cfg.crf_gt_prob == pytest.approx(0.8)
    assert cfg.crf_gaussian_sxy == 4
    assert cfg.crf_gaussian_compat == 4
    assert cfg.crf_bilateral_sxy == 80
    assert cfg.crf_bilateral_srgb == 13
    assert cfg.crf_bilateral_compat == 10


def test_load_clasp_config_overrides():
    from src.segmentation.clasp import load_clasp_config
    cfg = load_clasp_config({"clasp": {"bandwidth": 0.3, "min_clusters": 3, "dense_crf": False}})
    assert cfg.bandwidth == pytest.approx(0.3)
    assert cfg.min_clusters == 3
    assert cfg.dense_crf is False
    assert cfg.max_clusters == 15   # default unchanged


def test_load_clasp_config_missing_section():
    from src.segmentation.clasp import ClaspConfig, load_clasp_config
    cfg = load_clasp_config({})
    assert cfg.bandwidth == ClaspConfig.bandwidth
    assert cfg.crf_iterations == ClaspConfig.crf_iterations
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate
pytest tests/test_clasp.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.segmentation.clasp'`

- [ ] **Step 3: Create `src/segmentation/clasp.py` with config and dataclasses**

```python
# src/segmentation/clasp.py
"""CLASP: Adaptive Spectral Clustering for unsupervised per-image segmentation.

Implements the algorithm from Curie & da Costa (2025), arXiv:2509.25016v2,
using DINOv3 patch tokens instead of dinov2_vits14_reg.

Usage:
    from src.segmentation.clasp import ClaspSegmenter, load_clasp_config
    import yaml
    with open("configs/classifier_config.yaml") as f:
        config = yaml.safe_load(f)
    segmenter = ClaspSegmenter(load_clasp_config(config), load_dino_config(config["dino"]))
    result = segmenter.segment(Path("image.jpg"))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ClaspConfig:
    """Configuration for CLASP segmentation."""

    output_dir: str = "outputs/segmentations_clasp"
    bandwidth: float = 0.5          # β for K search range around K_opt
    min_clusters: int = 2           # hard floor
    max_clusters: int = 15          # hard ceiling
    dense_crf: bool = True          # apply DenseCRF boundary refinement
    crf_iterations: int = 20
    crf_gt_prob: float = 0.8        # unary potential ground-truth probability
    crf_gaussian_sxy: int = 4       # Gaussian kernel spatial σ
    crf_gaussian_compat: int = 4
    crf_bilateral_sxy: int = 80     # bilateral kernel spatial σ
    crf_bilateral_srgb: int = 13    # bilateral kernel color σ
    crf_bilateral_compat: int = 10


def load_clasp_config(config_dict: dict) -> ClaspConfig:
    """Load ClaspConfig from the full parsed YAML config dict.

    Reads the ``clasp:`` sub-dict internally; same pattern as
    ``load_segmentation_config()``.

    Args:
        config_dict: Full parsed YAML dictionary.

    Returns:
        Populated ClaspConfig.
    """
    cl = config_dict.get("clasp", {})
    return ClaspConfig(
        output_dir=cl.get("output_dir", ClaspConfig.output_dir),
        bandwidth=cl.get("bandwidth", ClaspConfig.bandwidth),
        min_clusters=cl.get("min_clusters", ClaspConfig.min_clusters),
        max_clusters=cl.get("max_clusters", ClaspConfig.max_clusters),
        dense_crf=cl.get("dense_crf", ClaspConfig.dense_crf),
        crf_iterations=cl.get("crf_iterations", ClaspConfig.crf_iterations),
        crf_gt_prob=cl.get("crf_gt_prob", ClaspConfig.crf_gt_prob),
        crf_gaussian_sxy=cl.get("crf_gaussian_sxy", ClaspConfig.crf_gaussian_sxy),
        crf_gaussian_compat=cl.get("crf_gaussian_compat", ClaspConfig.crf_gaussian_compat),
        crf_bilateral_sxy=cl.get("crf_bilateral_sxy", ClaspConfig.crf_bilateral_sxy),
        crf_bilateral_srgb=cl.get("crf_bilateral_srgb", ClaspConfig.crf_bilateral_srgb),
        crf_bilateral_compat=cl.get("crf_bilateral_compat", ClaspConfig.crf_bilateral_compat),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ClaspResult:
    """Output of CLASP segmentation for a single image."""

    label_map: np.ndarray       # int32, shape (H, W) at native resolution, values 0…K-1
    k_chosen: int               # final cluster count after silhouette search
    segment_areas: list[int]    # pixel count per segment, sorted descending
    image_path: Path


# ---------------------------------------------------------------------------
# Segmenter (stub — methods added in later tasks)
# ---------------------------------------------------------------------------


class ClaspSegmenter:
    """Segment images via CLASP: DINOv3 patch cosine affinity + spectral clustering.

    DINOv3 is lazy-loaded on first use.

    Args:
        seg_config: CLASP segmentation configuration.
        dino_config: DINOv3 configuration (from ``load_dino_config()``).
    """

    def __init__(self, seg_config: ClaspConfig, dino_config: Any) -> None:
        self._clasp_config = seg_config
        self._dino_config = dino_config
        self._extractor: Any = None

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            from src.features.dino_embeddings import DinoFeatureExtractor
            self._extractor = DinoFeatureExtractor(self._dino_config)
        return self._extractor
```

- [ ] **Step 4: Add `clasp:` block to `configs/classifier_config.yaml`**

Append to the end of `configs/classifier_config.yaml`:

```yaml

clasp:
  output_dir: outputs/segmentations_clasp
  bandwidth: 0.5
  min_clusters: 2
  max_clusters: 15
  dense_crf: true
  crf_iterations: 20
  crf_gt_prob: 0.8
  crf_gaussian_sxy: 4
  crf_gaussian_compat: 4
  crf_bilateral_sxy: 80
  crf_bilateral_srgb: 13
  crf_bilateral_compat: 10
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_clasp.py::test_clasp_config_defaults tests/test_clasp.py::test_load_clasp_config_overrides tests/test_clasp.py::test_load_clasp_config_missing_section -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/segmentation/clasp.py configs/classifier_config.yaml tests/test_clasp.py
git commit -m "feat: add CLASP config, dataclasses, and config tests"
```

---

## Task 2: Affinity matrix and eigengap heuristic

**Files:**
- Modify: `src/segmentation/clasp.py` (add `_build_affinity`, `_eigengap_k`)
- Modify: `tests/test_clasp.py` (add 5 tests)

- [ ] **Step 1: Write failing affinity + eigengap tests**

Append to `tests/test_clasp.py`:

```python
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from src.features.dino_embeddings import DinoConfig


def _make_segmenter(dense_crf: bool = False) -> "ClaspSegmenter":
    from src.segmentation.clasp import ClaspConfig, ClaspSegmenter
    cfg = ClaspConfig(dense_crf=dense_crf)
    return ClaspSegmenter(cfg, DinoConfig())


# ---------------------------------------------------------------------------
# Affinity matrix tests
# ---------------------------------------------------------------------------


def test_build_affinity_shape():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((100, 768)).astype(np.float32)
    A = segmenter._build_affinity(tokens)
    assert A.shape == (100, 100)


def test_build_affinity_diagonal():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(1)
    tokens = rng.standard_normal((50, 768)).astype(np.float32)
    A = segmenter._build_affinity(tokens)
    np.testing.assert_allclose(np.diag(A), 1.0, atol=1e-5)


def test_build_affinity_range():
    segmenter = _make_segmenter()
    rng = np.random.default_rng(2)
    tokens = rng.standard_normal((80, 768)).astype(np.float32)
    A = segmenter._build_affinity(tokens)
    assert A.min() >= 0.0
    assert A.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Eigengap tests
# ---------------------------------------------------------------------------


def test_eigengap_k_clear_drop():
    """Sharp drop between λ_3 and λ_4 → K_opt = 4."""
    segmenter = _make_segmenter()
    # gaps: [2, 1, 6, 0.1, 0.1, 0.1]  → biggest gap at 1-based index 3 → K_opt=4
    eigenvalues = np.array([10.0, 8.0, 7.0, 1.0, 0.9, 0.8, 0.7], dtype=np.float32)
    k = segmenter._eigengap_k(eigenvalues)
    assert k == 4


def test_eigengap_k_monotone():
    """Monotone decay still returns a valid int."""
    segmenter = _make_segmenter()
    eigenvalues = np.linspace(10.0, 1.0, 20).astype(np.float32)
    k = segmenter._eigengap_k(eigenvalues)
    assert isinstance(k, int)
    assert 1 <= k <= len(eigenvalues)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_clasp.py -k "affinity or eigengap" -v
```
Expected: `AttributeError: 'ClaspSegmenter' object has no attribute '_build_affinity'`

- [ ] **Step 3: Add `_build_affinity` and `_eigengap_k` to `ClaspSegmenter`**

Add these methods inside the `ClaspSegmenter` class in `src/segmentation/clasp.py`, after `_get_extractor`:

```python
    # ------------------------------------------------------------------
    # Affinity matrix
    # ------------------------------------------------------------------

    def _build_affinity(self, patch_tokens: np.ndarray) -> np.ndarray:
        """Build cosine affinity matrix clipped to [0, 1].

        Args:
            patch_tokens: shape (n_patches, 768)

        Returns:
            float32 affinity matrix of shape (n_patches, n_patches), values in [0, 1].
        """
        norms = np.linalg.norm(patch_tokens, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = (patch_tokens / norms).astype(np.float32)
        A = normed @ normed.T
        return np.maximum(A, 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Eigengap elbow heuristic
    # ------------------------------------------------------------------

    def _eigengap_k(self, eigenvalues: np.ndarray) -> int:
        """Eigengap elbow heuristic on descending eigenvalue sequence.

        Gaps: δ_i = λ_i − λ_{i+1} (1-based i = 1…n−1, stored at 0-index j = i−1).
        Fits a line from (1, δ_1) to (n−1, δ_{n−1}) and returns K_opt = i* + 1
        where i* (1-based) is the index of maximum perpendicular distance.

        Args:
            eigenvalues: 1-D array in descending order.

        Returns:
            K_opt (int ≥ 1).
        """
        n = len(eigenvalues)
        if n <= 2:
            return 1

        # gaps[j] = eigenvalues[j] - eigenvalues[j+1], 0-indexed; j = i-1
        gaps = eigenvalues[:-1] - eigenvalues[1:]   # shape (n-1,)

        # Line from P_0 = (1, gaps[0]) to P_{n-2} = (n-1, gaps[n-2])
        x1, y1 = 1.0, float(gaps[0])
        x2, y2 = float(n - 1), float(gaps[-1])
        dx, dy = x2 - x1, y2 - y1
        line_len = float(np.sqrt(dx ** 2 + dy ** 2))

        if line_len < 1e-10:
            return 2  # all gaps equal; no clear elbow

        # Perpendicular distance from point (j+1, gaps[j]) to the line
        distances = np.array(
            [abs(dy * (j + 1 - x1) - dx * (float(gaps[j]) - y1)) for j in range(n - 1)],
            dtype=np.float64,
        ) / line_len

        j_star = int(np.argmax(distances))   # 0-indexed
        k_opt = j_star + 2                   # i* = j*+1 (1-based), K_opt = i*+1
        return int(k_opt)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_clasp.py -k "affinity or eigengap" -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/clasp.py tests/test_clasp.py
git commit -m "feat: add CLASP affinity matrix and eigengap heuristic with tests"
```

---

## Task 3: Silhouette K-search and label map upsample

**Files:**
- Modify: `src/segmentation/clasp.py` (add `_search_best_k`, `_make_label_map`)
- Modify: `tests/test_clasp.py` (add 4 tests)

- [ ] **Step 1: Write failing silhouette + label map tests**

Append to `tests/test_clasp.py`:

```python
# ---------------------------------------------------------------------------
# Silhouette K-search tests
# ---------------------------------------------------------------------------


def test_search_best_k_in_range():
    """Returns K within the bandwidth-clamped range."""
    segmenter = _make_segmenter()
    rng = np.random.default_rng(42)
    # Three well-separated clusters in 2D
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
    pts = np.vstack([c + rng.standard_normal((20, 2)) * 0.3 for c in centers]).astype(np.float32)
    k_opt = 3
    k, labels = segmenter._search_best_k(pts, k_opt)
    k_lo = max(2, int(np.floor(k_opt * (1 - 0.5))))   # 2
    k_hi = min(15, int(np.ceil(k_opt * (1 + 0.5))))   # 5
    assert k_lo <= k <= k_hi
    assert labels.shape == (60,)


def test_search_best_k_respects_bounds():
    """K_opt near min_clusters clamps correctly."""
    segmenter = _make_segmenter()
    rng = np.random.default_rng(7)
    # k_opt=2; bandwidth=0.5 → k_lo = max(2, floor(1)) = 2
    pts = rng.standard_normal((30, 3)).astype(np.float32)
    k, labels = segmenter._search_best_k(pts, k_opt=2)
    assert k >= segmenter._clasp_config.min_clusters
    assert k <= segmenter._clasp_config.max_clusters


# ---------------------------------------------------------------------------
# Label map upsample tests
# ---------------------------------------------------------------------------


def test_make_label_map_shape():
    segmenter = _make_segmenter()
    labels = np.zeros(196, dtype=np.int32)
    lm = segmenter._make_label_map(labels, image_size=(320, 240), n_patches=196)
    assert lm.shape == (240, 320)   # (height, width)


def test_make_label_map_label_range():
    segmenter = _make_segmenter()
    k = 4
    labels = (np.arange(196) % k).astype(np.int32)
    lm = segmenter._make_label_map(labels, image_size=(224, 224), n_patches=196)
    assert lm.min() >= 0
    assert lm.max() < k
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_clasp.py -k "search_best_k or label_map" -v
```
Expected: `AttributeError: 'ClaspSegmenter' object has no attribute '_search_best_k'`

- [ ] **Step 3: Add `_search_best_k` and `_make_label_map` to `ClaspSegmenter`**

Add after `_eigengap_k` in `src/segmentation/clasp.py`:

```python
    # ------------------------------------------------------------------
    # K-search via silhouette score
    # ------------------------------------------------------------------

    def _search_best_k(
        self,
        eigenvectors: np.ndarray,
        k_opt: int,
    ) -> tuple[int, np.ndarray]:
        """Sweep K around K_opt, pick the clustering with the best silhouette score.

        Args:
            eigenvectors: shape (n_patches, n_eigs) — the leading eigenvectors
                          from the eigendecomposition (already in descending order).
            k_opt: candidate K from the eigengap heuristic.

        Returns:
            (best_k, labels) where labels is int32 of shape (n_patches,).
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize

        cfg = self._clasp_config
        k_lo = max(cfg.min_clusters, int(np.floor(k_opt * (1 - cfg.bandwidth))))
        k_hi = min(cfg.max_clusters, int(np.ceil(k_opt * (1 + cfg.bandwidth))))
        k_hi = max(k_lo, k_hi)  # guarantee at least one candidate

        best_k = k_lo
        best_score = -np.inf
        best_labels: np.ndarray | None = None

        for k in range(k_lo, k_hi + 1):
            # Take first k eigenvectors and row-normalize
            U = normalize(eigenvectors[:, :k], norm="l2")
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(U)

            if len(np.unique(labels)) < 2:
                continue  # silhouette requires ≥ 2 distinct clusters

            score = silhouette_score(U, labels, metric="euclidean")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            # Fallback: all K values produced degenerate clustering
            U = normalize(eigenvectors[:, :k_lo], norm="l2")
            km = KMeans(n_clusters=k_lo, random_state=42, n_init=10)
            best_labels = km.fit_predict(U)
            best_k = k_lo

        return best_k, best_labels.astype(np.int32)

    # ------------------------------------------------------------------
    # Label map upsample
    # ------------------------------------------------------------------

    def _make_label_map(
        self,
        labels: np.ndarray,
        image_size: tuple[int, int],
        n_patches: int,
    ) -> np.ndarray:
        """Upsample patch-level cluster labels to native image resolution.

        Args:
            labels: int32 of shape (n_patches,)
            image_size: (width, height) in PIL convention
            n_patches: total patch count (grid_size = round(sqrt(n_patches)))

        Returns:
            int32 label map of shape (height, width).
        """
        from PIL import Image

        grid_size = int(round(np.sqrt(n_patches)))
        width, height = image_size
        label_grid = labels.reshape(grid_size, grid_size).astype(np.uint8)
        upsampled = Image.fromarray(label_grid).resize((width, height), Image.NEAREST)
        return np.asarray(upsampled, dtype=np.int32)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_clasp.py -k "search_best_k or label_map" -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/segmentation/clasp.py tests/test_clasp.py
git commit -m "feat: add CLASP silhouette K-search and label map upsample with tests"
```

---

## Task 4: DenseCRF, `segment()`, lazy init test, and pyproject.toml

**Files:**
- Modify: `src/segmentation/clasp.py` (add `_apply_dense_crf`, `segment`)
- Modify: `pyproject.toml` (add pydensecrf to `[ml]`)
- Modify: `tests/test_clasp.py` (add 3 tests)

- [ ] **Step 1: Write failing DenseCRF + lazy-init tests**

Append to `tests/test_clasp.py`:

```python
# ---------------------------------------------------------------------------
# DenseCRF tests
# ---------------------------------------------------------------------------


def test_apply_dense_crf_skipped():
    """dense_crf=False returns label_map unchanged without touching pydensecrf."""
    segmenter = _make_segmenter(dense_crf=False)
    label_map = np.zeros((100, 100), dtype=np.int32)
    image_np = np.zeros((100, 100, 3), dtype=np.uint8)
    result = segmenter._apply_dense_crf(image_np, label_map, k=2)
    np.testing.assert_array_equal(result, label_map)


def test_apply_dense_crf_shape():
    """DenseCRF refinement preserves label map shape."""
    pytest.importorskip("pydensecrf")
    segmenter = _make_segmenter(dense_crf=True)
    rng = np.random.default_rng(0)
    h, w = 56, 56
    image_np = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    label_map = (rng.integers(0, 3, (h, w))).astype(np.int32)
    result = segmenter._apply_dense_crf(image_np, label_map, k=3)
    assert result.shape == (h, w)
    assert result.dtype == np.int32


# ---------------------------------------------------------------------------
# Lazy-init test
# ---------------------------------------------------------------------------


def test_clasp_segmenter_lazy_init():
    """ClaspSegmenter does not load DINOv3 at construction."""
    from src.segmentation.clasp import ClaspConfig, ClaspSegmenter
    segmenter = ClaspSegmenter(ClaspConfig(), DinoConfig())
    assert segmenter._extractor is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_clasp.py -k "dense_crf or lazy_init" -v
```
Expected: `AttributeError: 'ClaspSegmenter' object has no attribute '_apply_dense_crf'`

- [ ] **Step 3: Add `_apply_dense_crf` and `segment` to `ClaspSegmenter`**

Add after `_make_label_map` in `src/segmentation/clasp.py`:

```python
    # ------------------------------------------------------------------
    # DenseCRF boundary refinement
    # ------------------------------------------------------------------

    def _apply_dense_crf(
        self,
        image_np: np.ndarray,
        label_map: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Refine segment boundaries with fully-connected DenseCRF.

        Parameters follow §4.3 of Curie & da Costa (2025). Uncertain labeling
        is disabled (``zero_unsure=False``). No-op when ``dense_crf=False``.

        Args:
            image_np: uint8 RGB array of shape (H, W, 3) at native resolution.
            label_map: int32 label map of shape (H, W).
            k: number of classes.

        Returns:
            Refined int32 label map of shape (H, W).
        """
        if not self._clasp_config.dense_crf:
            return label_map

        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels
        except ImportError as exc:
            raise ImportError(
                "pydensecrf is required for DenseCRF refinement. "
                "Install with: pip install "
                "\"pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git\""
            ) from exc

        cfg = self._clasp_config
        h, w = image_np.shape[:2]

        d = dcrf.DenseCRF2D(w, h, k)

        U = unary_from_labels(
            label_map.astype(np.int32),
            k,
            gt_prob=cfg.crf_gt_prob,
            zero_unsure=False,  # disable uncertain labeling
        )
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(
            sxy=cfg.crf_gaussian_sxy,
            compat=cfg.crf_gaussian_compat,
        )
        d.addPairwiseBilateral(
            sxy=cfg.crf_bilateral_sxy,
            srgb=cfg.crf_bilateral_srgb,
            rgbim=image_np,
            compat=cfg.crf_bilateral_compat,
        )

        Q = d.inference(cfg.crf_iterations)
        return np.argmax(Q, axis=0).reshape(h, w).astype(np.int32)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment(self, image_path: Path) -> ClaspResult:
        """Run full CLASP pipeline for a single image.

        Loads the image at native resolution, extracts DINOv3 patch tokens,
        builds the cosine affinity matrix, eigendecomposes, selects K via
        eigengap + silhouette, upsamples, and optionally applies DenseCRF.

        Args:
            image_path: Path to the image file (JPEG or PNG).

        Returns:
            ClaspResult with label map, K chosen, and per-segment areas.
        """
        from PIL import Image
        from scipy.linalg import eigh

        pil_img = Image.open(image_path).convert("RGB")
        image_size = pil_img.size          # (width, height) PIL convention
        image_np = np.asarray(pil_img)     # uint8 (H, W, 3) for DenseCRF

        patch_tokens = self._get_extractor().extract_patch_tokens(pil_img)
        n_patches = patch_tokens.shape[0]

        A = self._build_affinity(patch_tokens)

        # scipy.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = eigh(A)
        eigenvalues = eigenvalues[::-1].copy()
        eigenvectors = eigenvectors[:, ::-1].copy()

        k_opt = self._eigengap_k(eigenvalues)
        k_chosen, labels = self._search_best_k(eigenvectors, k_opt)

        label_map = self._make_label_map(labels, image_size, n_patches)
        label_map = self._apply_dense_crf(image_np, label_map, k_chosen)

        segment_areas = sorted(
            [int((label_map == i).sum()) for i in range(k_chosen)],
            reverse=True,
        )

        return ClaspResult(
            label_map=label_map,
            k_chosen=k_chosen,
            segment_areas=segment_areas,
            image_path=image_path,
        )
```

- [ ] **Step 4: Update `pyproject.toml` — add pydensecrf to `[ml]` extras**

In `pyproject.toml`, find the `ml = [` block and add the pydensecrf git URL:

```toml
ml = [
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers>=4.40",
    "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
    "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git",
]
```

- [ ] **Step 5: Run all CLASP tests**

```bash
pytest tests/test_clasp.py -v
```
Expected: 15 passed

- [ ] **Step 6: Commit**

```bash
git add src/segmentation/clasp.py tests/test_clasp.py pyproject.toml
git commit -m "feat: add CLASP DenseCRF refinement, segment() entry point, and pyproject dep"
```

---

## Task 5: Batch script `generate_clasp_segmentations.py`

**Files:**
- Create: `scripts/generate_clasp_segmentations.py`

- [ ] **Step 1: Create the batch script**

```python
#!/usr/bin/env python
"""Batch CLASP segmentation of desert pattern images using DINOv3 + spectral clustering.

For each labeled image in checked-images/, runs CLASP to produce segment labels,
saves per-image overlays (Wong palette), label map PNGs, and metadata JSON.

Usage:
    python scripts/generate_clasp_segmentations.py [--force] [--verbose]
    python scripts/generate_clasp_segmentations.py --image IMG_8346.jpeg
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def make_clasp_overlay(image_np: np.ndarray, result) -> np.ndarray:
    """Blend original image with per-segment colors at 50% alpha (Wong palette).

    Args:
        image_np: uint8 RGB array, shape (H, W, 3)
        result: ClaspResult

    Returns:
        uint8 RGB composite, same shape as image_np.
    """
    from src.visualization.style import WONG_PALETTE

    out = image_np.astype(np.float32) / 255.0
    for seg_id in range(result.k_chosen):
        r, g, b = _hex_to_rgb(WONG_PALETTE[seg_id % len(WONG_PALETTE)])
        color = np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float32)
        mask = result.label_map == seg_id
        out[mask] = out[mask] * 0.5 + color * 0.5

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def save_clasp_outputs(result, class_dir: Path) -> None:
    """Write overlay PNG, label map PNG, and segments JSON.

    Args:
        result: ClaspResult
        class_dir: Per-class output directory (e.g. outputs/segmentations_clasp/mudcrack/)
    """
    stem = result.image_path.stem
    image_np = np.asarray(Image.open(result.image_path).convert("RGB"))

    # Segment color overlay
    overlay = make_clasp_overlay(image_np, result)
    Image.fromarray(overlay).save(class_dir / f"{stem}_clasp_overlay.png")

    # Raw label map as indexed-color PNG
    Image.fromarray(result.label_map.astype(np.uint8)).save(
        class_dir / f"{stem}_clasp_labelmap.png"
    )

    # Metadata JSON
    with open(class_dir / f"{stem}_clasp_segments.json", "w") as f:
        json.dump(
            {
                "k": result.k_chosen,
                "segment_areas": result.segment_areas,
                "image_path": str(result.image_path),
            },
            f,
            indent=2,
        )

    logger.info("Saved %s — k=%d segments", stem, result.k_chosen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-segment desert pattern images with CLASP."
    )
    parser.add_argument("--config", default="configs/classifier_config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if _clasp_overlay.png already exists")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--image", nargs="+", default=None,
        help="Process only these image files (filename or stem).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    from src.classification.train import load_classifier_config, scan_labeled_images
    from src.features.dino_embeddings import load_dino_config
    from src.segmentation.clasp import ClaspSegmenter, load_clasp_config

    clf_config = load_classifier_config(config_dict)
    clasp_config = load_clasp_config(config_dict)
    dino_config = load_dino_config(config_dict.get("dino", {}))

    image_list = scan_labeled_images(Path(clf_config.image_dir), clf_config.label_map)

    if args.image:
        targets = set(args.image)
        image_list = [
            (p, lbl) for p, lbl in image_list
            if p.name in targets or p.stem in targets
        ]
        if not image_list:
            raise ValueError(
                f"No images matched --image {args.image!r}. "
                "Check filenames against the checked-images/ directory."
            )
        logger.info("Processing %d specific image(s)", len(image_list))
    else:
        logger.info("Processing all %d images", len(image_list))

    # Create output directories
    output_dir = Path(clasp_config.output_dir)
    for _, class_name in image_list:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    segmenter = ClaspSegmenter(clasp_config, dino_config)

    for img_path, class_name in image_list:
        class_dir = output_dir / class_name
        overlay_path = class_dir / f"{img_path.stem}_clasp_overlay.png"
        if overlay_path.exists() and not args.force:
            logger.debug("Skipping %s (overlay exists; use --force to re-run)", img_path.stem)
            continue

        logger.info("Segmenting %s (%s)", img_path.stem, class_name)
        try:
            result = segmenter.segment(img_path)
            save_clasp_outputs(result, class_dir)
        except Exception as exc:
            logger.error("Failed to segment %s: %s", img_path.stem, exc, exc_info=True)

    logger.info("Done — CLASP outputs at %s", output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite to confirm nothing broken**

```bash
pytest tests/test_clasp.py -v
```
Expected: 15 passed

- [ ] **Step 3: Smoke-test on one image**

```bash
source .venv/bin/activate
python scripts/generate_clasp_segmentations.py --image IMG_8223.jpeg --verbose
```
Expected: `INFO __main__: Saved IMG_8223 — k=N segments` (N typically 3–8)
Check `outputs/segmentations_clasp/mudcrack/IMG_8223_clasp_overlay.png` exists.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_clasp_segmentations.py
git commit -m "feat: add generate_clasp_segmentations.py batch script"
```

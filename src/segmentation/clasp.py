"""CLASP: Adaptive Spectral Clustering for unsupervised per-image segmentation.

Implements the algorithm from Curie & da Costa (2025), arXiv:2509.25016v2,
using DINOv3 patch tokens instead of dinov2_vits14_reg.

Usage:
    from src.segmentation.clasp import ClaspSegmenter, load_clasp_config
    from src.features.dino_embeddings import load_dino_config
    import yaml
    with open("configs/classifier_config.yaml") as f:
        config = yaml.safe_load(f)
    segmenter = ClaspSegmenter(load_clasp_config(config), load_dino_config(config.get("dino", {})))
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
# Segmenter
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
                "Install with: "
                'pip install "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git"'
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
        image_np = np.array(pil_img)        # uint8 (H, W, 3) for DenseCRF; writeable copy required by pydensecrf

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

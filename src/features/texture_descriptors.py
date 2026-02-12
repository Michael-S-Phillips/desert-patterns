"""Classical texture feature extraction for desert pattern images.

Extracts 90 interpretable texture features per image:
  - GLCM (20): 5 properties × 4 distances, angles averaged
  - Gabor (48): 4 frequencies × 6 orientations × (mean, var)
  - Fractal dimension (1): box-counting on Canny edges
  - Lacunarity (5): sliding-box at 5 scales
  - LBP (10): uniform LBP histogram (P=8, R=1)
  - Global statistics (6): mean, std, skew, kurtosis, edge_density, dominant_freq
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import stats as sp_stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GLCMConfig:
    """GLCM feature extraction parameters."""

    distances: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    angles: list[float] = field(
        default_factory=lambda: [0.0, 0.785, 1.571, 2.356]
    )
    properties: list[str] = field(
        default_factory=lambda: ["contrast", "correlation", "energy", "homogeneity"]
    )
    levels: int = 256


@dataclass
class GaborConfig:
    """Gabor filter bank parameters."""

    frequencies: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.4]
    )
    orientations: list[float] = field(
        default_factory=lambda: [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    )


@dataclass
class FractalConfig:
    """Fractal dimension parameters."""

    method: str = "box_counting"
    threshold: str = "otsu"


@dataclass
class LacunarityConfig:
    """Lacunarity parameters."""

    box_sizes: list[int] = field(default_factory=lambda: [5, 10, 20, 40, 80])


@dataclass
class LBPConfig:
    """Local Binary Pattern parameters."""

    radii: list[int] = field(default_factory=lambda: [1])
    method: str = "uniform"


@dataclass
class GlobalStatsConfig:
    """Global statistics parameters."""

    compute_fft_dominant_freq: bool = True
    edge_detector: str = "canny"


@dataclass
class TextureConfig:
    """Full texture descriptor configuration."""

    glcm: GLCMConfig = field(default_factory=GLCMConfig)
    gabor: GaborConfig = field(default_factory=GaborConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    lacunarity: LacunarityConfig = field(default_factory=LacunarityConfig)
    lbp: LBPConfig = field(default_factory=LBPConfig)
    global_stats: GlobalStatsConfig = field(default_factory=GlobalStatsConfig)


def load_texture_config(config_dict: dict) -> TextureConfig:
    """Create a TextureConfig from a config dictionary.

    Args:
        config_dict: Dictionary with texture configuration values.

    Returns:
        Populated TextureConfig.
    """
    cfg = TextureConfig()

    if "glcm" in config_dict:
        g = config_dict["glcm"]
        cfg.glcm = GLCMConfig(
            distances=g.get("distances", cfg.glcm.distances),
            angles=g.get("angles", cfg.glcm.angles),
            properties=g.get("properties", cfg.glcm.properties),
            levels=g.get("levels", cfg.glcm.levels),
        )

    if "gabor" in config_dict:
        g = config_dict["gabor"]
        cfg.gabor = GaborConfig(
            frequencies=g.get("frequencies", cfg.gabor.frequencies),
            orientations=g.get("orientations", cfg.gabor.orientations),
        )

    if "fractal" in config_dict:
        f = config_dict["fractal"]
        cfg.fractal = FractalConfig(
            method=f.get("method", cfg.fractal.method),
            threshold=f.get("threshold", cfg.fractal.threshold),
        )

    if "lacunarity" in config_dict:
        lc = config_dict["lacunarity"]
        cfg.lacunarity = LacunarityConfig(
            box_sizes=lc.get("box_sizes", cfg.lacunarity.box_sizes),
        )

    if "lbp" in config_dict:
        lb = config_dict["lbp"]
        cfg.lbp = LBPConfig(
            radii=lb.get("radii", cfg.lbp.radii),
            method=lb.get("method", cfg.lbp.method),
        )

    if "global" in config_dict:
        gs = config_dict["global"]
        cfg.global_stats = GlobalStatsConfig(
            compute_fft_dominant_freq=gs.get(
                "compute_fft_dominant_freq",
                cfg.global_stats.compute_fft_dominant_freq,
            ),
            edge_detector=gs.get("edge_detector", cfg.global_stats.edge_detector),
        )

    return cfg


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _box_counting_dimension(binary_image: np.ndarray) -> float:
    """Compute fractal dimension via box-counting on a binary image.

    Args:
        binary_image: 2-D boolean or uint8 array (non-zero = foreground).

    Returns:
        Estimated fractal dimension (typically 1.0–2.0 for edge maps).
        Returns 0.0 if the image has no foreground pixels.
    """
    pixels = binary_image > 0
    if not np.any(pixels):
        return 0.0

    # Pad to square with side = next power of 2
    max_dim = max(pixels.shape)
    p = int(np.ceil(np.log2(max_dim)))
    size = 2**p

    padded = np.zeros((size, size), dtype=bool)
    padded[: pixels.shape[0], : pixels.shape[1]] = pixels

    counts = []
    sizes = []
    for k in range(1, p + 1):
        box_size = 2**k
        n_boxes_per_side = size // box_size
        # Reshape into boxes and check occupancy
        reshaped = padded.reshape(
            n_boxes_per_side, box_size, n_boxes_per_side, box_size
        )
        box_occupied = reshaped.any(axis=(1, 3))
        count = int(box_occupied.sum())
        if count > 0:
            counts.append(np.log(count))
            sizes.append(np.log(1.0 / box_size))

    if len(counts) < 2:
        return 0.0

    # Linear regression: log(count) vs log(1/box_size)
    coeffs = np.polyfit(sizes, counts, 1)
    return float(coeffs[0])


def _sliding_box_lacunarity(
    binary_image: np.ndarray, box_size: int
) -> float:
    """Compute lacunarity at a given box size using an integral image.

    Lacunarity = var(mass) / mean(mass)^2 + 1 for sliding boxes.

    Args:
        binary_image: 2-D boolean or uint8 array (non-zero = foreground).
        box_size: Side length of the sliding box.

    Returns:
        Lacunarity value (≥ 1.0).  Returns 1.0 if the image is too small
        or has uniform density.
    """
    h, w = binary_image.shape
    if box_size > h or box_size > w:
        return 1.0

    img_float = (binary_image > 0).astype(np.float64)
    # Integral image for fast box sums
    integral = cv2.integral(img_float)

    # Box sum using integral image corners
    y_max = h - box_size + 1
    x_max = w - box_size + 1

    # Vectorized box sums via integral image
    # integral[y2+1, x2+1] - integral[y1, x2+1] - integral[y2+1, x1] + integral[y1, x1]
    y1 = np.arange(y_max)[:, None]
    x1 = np.arange(x_max)[None, :]
    y2 = y1 + box_size
    x2 = x1 + box_size

    box_sums = (
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )

    mean_mass = box_sums.mean()
    if mean_mass == 0:
        return 1.0

    var_mass = box_sums.var()
    lacunarity = var_mass / (mean_mass**2) + 1.0
    return float(lacunarity)


def _compute_dominant_frequency(gray: np.ndarray) -> float:
    """Compute the dominant spatial frequency from the 2D FFT magnitude spectrum.

    Args:
        gray: 2-D uint8 grayscale image.

    Returns:
        Dominant spatial frequency as fraction of Nyquist (0.0–1.0).
        Returns 0.0 for uniform images.
    """
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Zero out DC component
    magnitude[cy, cx] = 0.0

    if magnitude.max() == 0:
        return 0.0

    # Find peak
    peak_y, peak_x = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    # Distance from center (normalized by half the image size)
    dist = np.sqrt((peak_y - cy) ** 2 + (peak_x - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)
    return float(dist / max_dist) if max_dist > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class TextureDescriptorExtractor:
    """Extract classical texture features from grayscale images.

    Produces a 90-dimensional feature vector per image.

    Args:
        config: Texture extraction configuration.
    """

    # Ordered feature names (must match extract() output ordering)
    FEATURE_NAMES: list[str] = []

    def __init__(self, config: TextureConfig | None = None):
        self.config = config or TextureConfig()
        self.FEATURE_NAMES = self._build_feature_names()

    def _build_feature_names(self) -> list[str]:
        """Build the ordered list of feature names."""
        names: list[str] = []

        # GLCM: 5 props × 4 distances = 20
        for prop in self.config.glcm.properties:
            for d in self.config.glcm.distances:
                names.append(f"glcm_{prop}_d{d}")
        # GLCM entropy × 4 distances
        for d in self.config.glcm.distances:
            names.append(f"glcm_entropy_d{d}")

        # Gabor: 4 freqs × 6 orientations × 2 stats = 48
        for freq in self.config.gabor.frequencies:
            for ori in self.config.gabor.orientations:
                names.append(f"gabor_f{freq}_o{int(ori)}_mean")
                names.append(f"gabor_f{freq}_o{int(ori)}_var")

        # Fractal dimension: 1
        names.append("fractal_dimension")

        # Lacunarity: 5
        for bs in self.config.lacunarity.box_sizes:
            names.append(f"lacunarity_b{bs}")

        # LBP: 10 (P=8, uniform → P+2=10 bins)
        for i in range(10):
            names.append(f"lbp_bin{i}")

        # Global stats: 6
        names.extend([
            "global_mean",
            "global_std",
            "global_skewness",
            "global_kurtosis",
            "global_edge_density",
            "global_dominant_freq",
        ])

        return names

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract all 90 texture features from a single image.

        Args:
            image: Grayscale uint8 image (2-D array).  If a 3-channel BGR
                image is passed, it is converted to grayscale automatically.

        Returns:
            1-D numpy array of shape ``(90,)`` with float64 values.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        features: list[np.ndarray] = [
            self._compute_glcm(gray),
            self._compute_gabor(gray),
            np.array([self._compute_fractal_dim(gray)]),
            self._compute_lacunarity(gray),
            self._compute_lbp(gray),
            self._compute_global_stats(gray),
        ]
        return np.concatenate(features).astype(np.float64)

    def extract_batch(
        self,
        image_paths: list[Path] | list[str],
        checkpoint_path: Path | str | None = None,
        checkpoint_interval: int = 50,
    ) -> np.ndarray:
        """Extract texture features for a batch of images.

        Args:
            image_paths: List of paths to image files.
            checkpoint_path: Path for checkpoint file (.npy). If provided,
                partial results are saved every *checkpoint_interval* images
                and a prior checkpoint is resumed automatically.
            checkpoint_interval: Save checkpoint every N images.

        Returns:
            2-D numpy array of shape ``(N, 90)`` with float64 values.
        """
        n = len(image_paths)
        n_features = len(self.FEATURE_NAMES)
        result = np.empty((n, n_features), dtype=np.float64)
        log_interval = max(1, n // 10)
        start_idx = 0

        # Resume from checkpoint if available
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                saved = np.load(str(checkpoint_path))
                if saved.shape[0] == n and saved.shape[1] == n_features:
                    # Count how many rows are not all-zero (already computed)
                    filled = np.any(saved != 0.0, axis=1) | np.all(np.isnan(saved), axis=1)
                    start_idx = int(np.max(np.where(filled)[0])) + 1 if filled.any() else 0
                    result[:start_idx] = saved[:start_idx]
                    logger.info(
                        "Resumed from checkpoint: %d/%d images already done",
                        start_idx, n,
                    )
                else:
                    logger.warning(
                        "Checkpoint shape mismatch (%s vs expected (%d, %d)) — starting fresh",
                        saved.shape, n, n_features,
                    )

        for i in range(start_idx, n):
            path = image_paths[i]
            if i % log_interval == 0 or (i == start_idx and start_idx > 0):
                logger.info("Texture features: image %d/%d", i + 1, n)
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Could not read image %s — filling with NaN", path)
                result[i] = np.full(n_features, np.nan)
                continue
            result[i] = self.extract(img)

            # Save checkpoint periodically
            if checkpoint_path is not None and (i + 1) % checkpoint_interval == 0:
                np.save(str(checkpoint_path), result)
                logger.info("Checkpoint saved at image %d/%d", i + 1, n)

        # Clean up checkpoint on successful completion
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Texture extraction complete — checkpoint removed")

        return result

    # ----- GLCM (20 features) -----

    def _compute_glcm(self, gray: np.ndarray) -> np.ndarray:
        """Compute GLCM features: 4 properties × 4 distances + entropy × 4 distances.

        Angles are averaged for each (property, distance) pair.

        Returns:
            1-D array of 20 float values.
        """
        cfg = self.config.glcm
        glcm = graycomatrix(
            gray,
            distances=cfg.distances,
            angles=cfg.angles,
            levels=cfg.levels,
            symmetric=True,
            normed=True,
        )
        # glcm shape: (levels, levels, n_distances, n_angles)

        features: list[float] = []

        # Standard properties (averaged over angles)
        for prop in cfg.properties:
            props = graycoprops(glcm, prop)  # (n_distances, n_angles)
            for d_idx in range(len(cfg.distances)):
                features.append(float(props[d_idx, :].mean()))

        # GLCM entropy (manual, averaged over angles)
        for d_idx in range(len(cfg.distances)):
            entropies = []
            for a_idx in range(len(cfg.angles)):
                mat = glcm[:, :, d_idx, a_idx].astype(np.float64)
                mat = mat / (mat.sum() + 1e-12)
                nonzero = mat[mat > 0]
                entropy = -np.sum(nonzero * np.log2(nonzero))
                entropies.append(entropy)
            features.append(float(np.mean(entropies)))

        return np.array(features)

    # ----- Gabor (48 features) -----

    def _compute_gabor(self, gray: np.ndarray) -> np.ndarray:
        """Compute Gabor filter bank responses.

        For each (frequency, orientation) pair: mean and variance of response.

        Returns:
            1-D array of 48 float values.
        """
        cfg = self.config.gabor
        gray_float = gray.astype(np.float64) / 255.0
        features: list[float] = []

        for freq in cfg.frequencies:
            for ori_deg in cfg.orientations:
                theta = np.radians(ori_deg)
                filt_real, filt_imag = gabor(gray_float, frequency=freq, theta=theta)
                magnitude = np.sqrt(filt_real**2 + filt_imag**2)
                features.append(float(magnitude.mean()))
                features.append(float(magnitude.var()))

        return np.array(features)

    # ----- Fractal dimension (1 feature) -----

    def _compute_fractal_dim(self, gray: np.ndarray) -> float:
        """Compute fractal dimension via Otsu → Canny → box-counting.

        Returns:
            Fractal dimension estimate (typically 1.0–2.0).
        """
        # Otsu threshold to get binary, then Canny for edges
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary, 50, 150)
        return _box_counting_dimension(edges)

    # ----- Lacunarity (5 features) -----

    def _compute_lacunarity(self, gray: np.ndarray) -> np.ndarray:
        """Compute lacunarity at multiple box sizes.

        Returns:
            1-D array of 5 float values.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        features = []
        for bs in self.config.lacunarity.box_sizes:
            features.append(_sliding_box_lacunarity(binary, bs))
        return np.array(features)

    # ----- LBP (10 features) -----

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Compute LBP uniform histogram (P=8, R=1 → 10 bins).

        Returns:
            1-D array of 10 float values (normalized histogram).
        """
        radius = self.config.lbp.radii[0]
        n_points = 8 * radius
        lbp_image = local_binary_pattern(
            gray, P=n_points, R=radius, method=self.config.lbp.method
        )
        # Uniform LBP with P=8: P+2 = 10 bins (0..9)
        n_bins = n_points + 2
        hist, _ = np.histogram(
            lbp_image.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        return hist.astype(np.float64)

    # ----- Global statistics (6 features) -----

    def _compute_global_stats(self, gray: np.ndarray) -> np.ndarray:
        """Compute global image statistics.

        Returns:
            1-D array of 6 float values:
            [mean, std, skewness, kurtosis, edge_density, dominant_freq].
        """
        pixels = gray.astype(np.float64).ravel()

        mean_val = float(np.mean(pixels))
        std_val = float(np.std(pixels))
        # skew/kurtosis return NaN for constant arrays; fall back to 0
        if std_val == 0.0:
            skew_val = 0.0
            kurt_val = 0.0
        else:
            skew_val = float(sp_stats.skew(pixels))
            kurt_val = float(sp_stats.kurtosis(pixels))

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size

        # Dominant frequency
        if self.config.global_stats.compute_fft_dominant_freq:
            dominant_freq = _compute_dominant_frequency(gray)
        else:
            dominant_freq = 0.0

        return np.array([
            mean_val, std_val, skew_val, kurt_val, edge_density, dominant_freq
        ])

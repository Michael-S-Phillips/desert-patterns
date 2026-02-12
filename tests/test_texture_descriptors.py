"""Tests for classical texture descriptor extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.features.texture_descriptors import (
    TextureConfig,
    TextureDescriptorExtractor,
    _box_counting_dimension,
    _compute_dominant_frequency,
    _sliding_box_lacunarity,
    load_texture_config,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestTextureConfig:
    """Tests for TextureConfig and load_texture_config."""

    def test_defaults(self):
        cfg = TextureConfig()
        assert cfg.glcm.distances == [1, 3, 5, 10]
        assert len(cfg.gabor.orientations) == 6
        assert cfg.lbp.method == "uniform"

    def test_load_from_dict(self):
        d = {
            "glcm": {"distances": [1, 2]},
            "gabor": {"frequencies": [0.1, 0.2]},
        }
        cfg = load_texture_config(d)
        assert cfg.glcm.distances == [1, 2]
        assert cfg.gabor.frequencies == [0.1, 0.2]
        # Defaults preserved
        assert cfg.lbp.radii == [1]

    def test_load_empty_dict(self):
        cfg = load_texture_config({})
        assert cfg.glcm.levels == 256


# ---------------------------------------------------------------------------
# Feature count tests
# ---------------------------------------------------------------------------


class TestFeatureCounts:
    """Verify the expected 90 features are produced."""

    @pytest.fixture
    def extractor(self):
        return TextureDescriptorExtractor()

    def test_total_feature_count(self, extractor, checkerboard_image):
        result = extractor.extract(checkerboard_image)
        assert result.shape == (90,)

    def test_feature_names_count(self, extractor):
        assert len(extractor.FEATURE_NAMES) == 90

    def test_feature_names_match_output(self, extractor, checkerboard_image):
        result = extractor.extract(checkerboard_image)
        assert len(extractor.FEATURE_NAMES) == result.shape[0]

    def test_glcm_count(self, extractor):
        """GLCM: 4 props × 4 distances + 1 entropy × 4 distances = 20."""
        glcm_names = [n for n in extractor.FEATURE_NAMES if n.startswith("glcm_")]
        assert len(glcm_names) == 20

    def test_gabor_count(self, extractor):
        """Gabor: 4 freqs × 6 orientations × 2 (mean+var) = 48."""
        gabor_names = [n for n in extractor.FEATURE_NAMES if n.startswith("gabor_")]
        assert len(gabor_names) == 48

    def test_fractal_count(self, extractor):
        """Fractal: 1 feature."""
        fractal_names = [n for n in extractor.FEATURE_NAMES if n.startswith("fractal_")]
        assert len(fractal_names) == 1

    def test_lacunarity_count(self, extractor):
        """Lacunarity: 5 features."""
        lac_names = [n for n in extractor.FEATURE_NAMES if n.startswith("lacunarity_")]
        assert len(lac_names) == 5

    def test_lbp_count(self, extractor):
        """LBP: 10 bins (P=8 uniform → P+2=10)."""
        lbp_names = [n for n in extractor.FEATURE_NAMES if n.startswith("lbp_")]
        assert len(lbp_names) == 10

    def test_global_count(self, extractor):
        """Global stats: 6 features."""
        global_names = [n for n in extractor.FEATURE_NAMES if n.startswith("global_")]
        assert len(global_names) == 6


# ---------------------------------------------------------------------------
# Texture property tests
# ---------------------------------------------------------------------------


class TestCheckerboard:
    """Checkerboard should produce high contrast, many edges."""

    @pytest.fixture
    def features(self, checkerboard_image):
        ext = TextureDescriptorExtractor()
        return dict(zip(ext.FEATURE_NAMES, ext.extract(checkerboard_image)))

    def test_high_contrast(self, features):
        assert features["glcm_contrast_d1"] > 0

    def test_edge_density(self, features):
        assert features["global_edge_density"] > 0

    def test_no_nan(self, features):
        for name, val in features.items():
            assert np.isfinite(val), f"{name} is not finite: {val}"


class TestUniformImage:
    """Uniform gray image should have zero contrast, high energy."""

    @pytest.fixture
    def features(self, uniform_gray_image):
        ext = TextureDescriptorExtractor()
        return dict(zip(ext.FEATURE_NAMES, ext.extract(uniform_gray_image)))

    def test_zero_contrast(self, features):
        assert features["glcm_contrast_d1"] == 0.0

    def test_high_energy(self, features):
        # Energy = 1.0 for a uniform image (all probability in one cell)
        assert features["glcm_energy_d1"] == pytest.approx(1.0, abs=0.01)

    def test_zero_std(self, features):
        assert features["global_std"] == 0.0


class TestCircleImage:
    """Circle boundary should have fractal dimension around 1.0."""

    def test_fractal_dim_near_one(self, circle_image):
        ext = TextureDescriptorExtractor()
        features = dict(zip(ext.FEATURE_NAMES, ext.extract(circle_image)))
        # A circle boundary has fractal dim ~1.0
        assert 0.7 <= features["fractal_dimension"] <= 1.5


class TestLBPHistogram:
    """LBP histogram should sum to approximately 1 (density normalization)."""

    def test_lbp_sums_to_one(self, checkerboard_image):
        ext = TextureDescriptorExtractor()
        features = ext.extract(checkerboard_image)
        # LBP bins are indices 69..78 (after 20 GLCM + 48 Gabor + 1 fractal + 5 lacunarity = 74, then 69..78)
        lbp_start = 20 + 48 + 1 + 5  # = 74
        lbp_end = lbp_start + 10  # = 84
        lbp_hist = features[lbp_start:lbp_end]
        # density=True → integrates to 1 over bins with width 1
        assert np.sum(lbp_hist) == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Feature extraction should be deterministic."""

    def test_same_results(self, checkerboard_image):
        ext = TextureDescriptorExtractor()
        f1 = ext.extract(checkerboard_image)
        f2 = ext.extract(checkerboard_image)
        np.testing.assert_array_equal(f1, f2)


# ---------------------------------------------------------------------------
# No NaN on valid images
# ---------------------------------------------------------------------------


class TestNoNaN:
    """Features should not contain NaN on any valid synthetic image."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["checkerboard_image", "uniform_gray_image", "gradient_image", "circle_image"],
    )
    def test_no_nan(self, fixture_name, request):
        img = request.getfixturevalue(fixture_name)
        ext = TextureDescriptorExtractor()
        features = ext.extract(img)
        assert not np.any(np.isnan(features)), f"NaN in features for {fixture_name}"


# ---------------------------------------------------------------------------
# BGR auto-conversion
# ---------------------------------------------------------------------------


class TestBGRConversion:
    """Passing a 3-channel BGR image should work transparently."""

    def test_bgr_input(self, synthetic_518_bgr):
        ext = TextureDescriptorExtractor()
        features = ext.extract(synthetic_518_bgr)
        assert features.shape == (90,)
        assert not np.any(np.isnan(features))


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    """Batch extraction should match single-image extraction."""

    def test_batch_shape(self, tmp_path, checkerboard_image):
        ext = TextureDescriptorExtractor()
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(p), checkerboard_image)
            paths.append(p)

        result = ext.extract_batch(paths)
        assert result.shape == (3, 90)

    def test_batch_matches_single(self, tmp_path, checkerboard_image):
        ext = TextureDescriptorExtractor()
        single = ext.extract(checkerboard_image)

        p = tmp_path / "img.png"
        cv2.imwrite(str(p), checkerboard_image)
        batch = ext.extract_batch([p])

        np.testing.assert_allclose(batch[0], single, atol=1e-10)


# ---------------------------------------------------------------------------
# Standalone helper tests
# ---------------------------------------------------------------------------


class TestBoxCountingDimension:
    """Tests for _box_counting_dimension."""

    def test_empty_image(self):
        assert _box_counting_dimension(np.zeros((256, 256), dtype=np.uint8)) == 0.0

    def test_full_image(self):
        full = np.ones((256, 256), dtype=np.uint8) * 255
        dim = _box_counting_dimension(full)
        # A completely filled square has fractal dim ~2.0
        assert 1.5 <= dim <= 2.1


class TestSlidingBoxLacunarity:
    """Tests for _sliding_box_lacunarity."""

    def test_uniform_low_lacunarity(self):
        # Uniform image → lacunarity ~1.0
        uniform = np.ones((100, 100), dtype=np.uint8) * 255
        lac = _sliding_box_lacunarity(uniform, 10)
        assert lac == pytest.approx(1.0, abs=0.01)

    def test_box_too_large(self):
        small = np.ones((5, 5), dtype=np.uint8)
        assert _sliding_box_lacunarity(small, 10) == 1.0


class TestDominantFrequency:
    """Tests for _compute_dominant_frequency."""

    def test_uniform_zero_freq(self):
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        assert _compute_dominant_frequency(uniform) == 0.0

    def test_returns_bounded(self):
        rng = np.random.RandomState(42)
        noisy = rng.randint(0, 255, (64, 64), dtype=np.uint8)
        freq = _compute_dominant_frequency(noisy)
        assert 0.0 <= freq <= 1.0

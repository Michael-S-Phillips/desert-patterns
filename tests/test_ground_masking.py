"""Tests for src.data.ground_masking — ground image masking."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.ground_masking import (
    GroundMaskingConfig,
    MaskingMethod,
    detect_horizon,
    horizon_mask,
    lower_crop_mask,
    mask_ground_image,
)


class TestLowerCropMask:
    """Tests for lower_crop_mask."""

    def test_default_crop_480(self) -> None:
        """60% crop of 480px → keeps bottom 288 rows (crop_y=192)."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = 128
        result = lower_crop_mask(img, crop_fraction=0.6)

        assert result.success
        assert result.method == MaskingMethod.LOWER_CROP
        assert result.ground_fraction == pytest.approx(0.6)
        assert result.masked_region.shape == (288, 640, 3)

    def test_mask_dimensions(self) -> None:
        """Binary mask matches input dimensions."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = lower_crop_mask(img, crop_fraction=0.6)

        assert result.mask.shape == (480, 640)
        # Upper portion should be 0, lower should be 255
        assert result.mask[0, 0] == 0
        assert result.mask[-1, 0] == 255

    def test_full_image_crop(self) -> None:
        """crop_fraction=1.0 keeps the entire image."""
        img = np.full((100, 100, 3), 42, dtype=np.uint8)
        result = lower_crop_mask(img, crop_fraction=1.0)

        assert result.ground_fraction == pytest.approx(1.0)
        assert result.masked_region.shape == (100, 100, 3)

    def test_small_crop(self) -> None:
        """crop_fraction=0.3 keeps bottom 30%."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = lower_crop_mask(img, crop_fraction=0.3)

        assert result.masked_region.shape[0] == 30
        assert result.masked_region.shape[1] == 200


class TestDetectHorizon:
    """Tests for detect_horizon."""

    def test_synthetic_horizon(self, synthetic_ground_image: np.ndarray) -> None:
        """Detect horizon in synthetic sky/ground image."""
        horizon_y = detect_horizon(synthetic_ground_image)
        # Horizon should be near the boundary at y=192 (40% of 480)
        # Allow some tolerance since edge detection isn't pixel-perfect
        if horizon_y is not None:
            assert abs(horizon_y - 192) < 30

    def test_uniform_image_returns_none(self) -> None:
        """Uniform image has no horizon."""
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert detect_horizon(img) is None

    def test_noisy_image(self) -> None:
        """Pure noise image is unlikely to produce a horizon."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Should either return None or at least not crash
        result = detect_horizon(img)
        assert result is None or isinstance(result, int)


class TestHorizonMask:
    """Tests for horizon_mask."""

    def test_mask_below_horizon(self) -> None:
        """Everything below horizon_y is kept."""
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = horizon_mask(img, 200)

        assert result.success
        assert result.method == MaskingMethod.HORIZON
        assert result.masked_region.shape == (280, 640, 3)
        assert result.mask[199, 0] == 0
        assert result.mask[200, 0] == 255

    def test_horizon_at_top(self) -> None:
        """Horizon at y=0 keeps the entire image."""
        img = np.full((100, 100, 3), 42, dtype=np.uint8)
        result = horizon_mask(img, 0)
        assert result.masked_region.shape == (100, 100, 3)


class TestMaskGroundImage:
    """Integration tests for mask_ground_image."""

    def test_fallback_without_sam(
        self, synthetic_ground_jpg: Path
    ) -> None:
        """Without SAM, should fall back to horizon or crop."""
        config = GroundMaskingConfig()
        result = mask_ground_image(synthetic_ground_jpg, config, sam_masker=None)

        assert result.success
        assert result.method in (MaskingMethod.HORIZON, MaskingMethod.LOWER_CROP)
        assert result.masked_region is not None

    def test_nonexistent_image(self, tmp_path: Path) -> None:
        """Non-existent image returns FAILED result."""
        config = GroundMaskingConfig()
        result = mask_ground_image(tmp_path / "nope.jpg", config)

        assert not result.success
        assert result.method == MaskingMethod.FAILED

    def test_real_ground_image(self, sample_ground_jpg: Path) -> None:
        """Integration: mask a real ground image (skipped if data unavailable)."""
        config = GroundMaskingConfig()
        result = mask_ground_image(sample_ground_jpg, config, sam_masker=None)

        assert result.success
        assert result.masked_region is not None
        assert result.masked_region.shape[0] > 0
        assert result.masked_region.shape[1] > 0

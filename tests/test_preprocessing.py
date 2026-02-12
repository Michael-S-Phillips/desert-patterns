"""Tests for src.data.preprocessing — standardization and pipeline orchestration."""

import numpy as np
import pytest

from src.data.preprocessing import (
    generate_ground_patch_id,
    generate_tile_id,
    gray_world_balance,
    per_image_normalize,
    standardize_image,
)


class TestStandardizeImage:
    """Tests for standardize_image."""

    def test_output_shape_518(self) -> None:
        """Output is always 518x518x3."""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = standardize_image(img, target_size=518, normalize=False)
        assert result.shape == (518, 518, 3)

    def test_downscale_large(self) -> None:
        """Downscaling a large image produces 518x518."""
        img = np.full((3000, 4000, 3), 128, dtype=np.uint8)
        result = standardize_image(img, target_size=518, normalize=False)
        assert result.shape == (518, 518, 3)

    def test_upscale_small(self) -> None:
        """Upscaling a small image produces 518x518."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = standardize_image(img, target_size=518, normalize=False)
        assert result.shape == (518, 518, 3)

    def test_exact_size_unchanged(self) -> None:
        """518x518 input produces 518x518 output."""
        img = np.full((518, 518, 3), 128, dtype=np.uint8)
        result = standardize_image(img, target_size=518, normalize=False)
        assert result.shape == (518, 518, 3)

    def test_output_dtype(self) -> None:
        """Output is always uint8."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = standardize_image(img, target_size=518, normalize=True)
        assert result.dtype == np.uint8

    def test_with_white_balance(self) -> None:
        """White balance + normalize doesn't crash."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = standardize_image(
            img, target_size=518, normalize=True, white_balance=True
        )
        assert result.shape == (518, 518, 3)
        assert result.dtype == np.uint8


class TestGrayWorldBalance:
    """Tests for gray_world_balance."""

    def test_already_balanced(self) -> None:
        """Image with equal channel means is unchanged."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = gray_world_balance(img)
        np.testing.assert_array_equal(result, img)

    def test_blue_cast_correction(self) -> None:
        """Blue-heavy image has blue channel reduced."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # Blue channel high (BGR)
        img[:, :, 1] = 100  # Green
        img[:, :, 2] = 100  # Red

        result = gray_world_balance(img)
        # After correction, blue channel mean should decrease
        assert result[:, :, 0].mean() < 200
        # All channel means should be closer to each other
        means = [result[:, :, c].mean() for c in range(3)]
        assert max(means) - min(means) < 5

    def test_output_range(self) -> None:
        """Output stays in 0-255."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 200
        img[:, :, 2] = 100
        result = gray_world_balance(img)
        assert result.min() >= 0
        assert result.max() <= 255


class TestPerImageNormalize:
    """Tests for per_image_normalize."""

    def test_output_range(self) -> None:
        """Normalized image stays in 0-255."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = per_image_normalize(img)
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.dtype == np.uint8

    def test_uniform_produces_128(self) -> None:
        """Uniform image normalizes to constant 128."""
        img = np.full((100, 100, 3), 42, dtype=np.uint8)
        result = per_image_normalize(img)
        # All channels should be 128 (std=0 fallback)
        np.testing.assert_array_equal(result, 128)

    def test_varied_input(self) -> None:
        """Non-uniform image produces non-constant output."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50] = 50
        img[50:] = 200
        result = per_image_normalize(img)
        assert result.std() > 0


class TestGenerateTileId:
    """Tests for generate_tile_id."""

    def test_format(self) -> None:
        assert generate_tile_id("abc123", 0, 0) == "abc123_r00_c00"
        assert generate_tile_id("abc123", 7, 10) == "abc123_r07_c10"

    def test_zero_padding(self) -> None:
        result = generate_tile_id("x", 3, 5)
        assert result == "x_r03_c05"


class TestGenerateGroundPatchId:
    """Tests for generate_ground_patch_id."""

    def test_format(self) -> None:
        assert generate_ground_patch_id("abc123") == "abc123_gnd"

    def test_different_parents(self) -> None:
        id1 = generate_ground_patch_id("aaa")
        id2 = generate_ground_patch_id("bbb")
        assert id1 != id2

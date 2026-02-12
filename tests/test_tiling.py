"""Tests for src.data.tiling — drone image tiling."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.tiling import (
    TilingConfig,
    compute_tile_grid,
    compute_tile_stats,
    extract_tile,
    tile_drone_image,
)


class TestComputeTileGrid:
    """Tests for compute_tile_grid."""

    def test_standard_drone_4000x3000(self) -> None:
        """4000x3000 with 512 tiles and 25% overlap → 88 tiles (11×8)."""
        positions = compute_tile_grid(4000, 3000, 512, 0.25)
        assert len(positions) == 88
        rows = {p[2] for p in positions}
        cols = {p[3] for p in positions}
        assert len(rows) == 8
        assert len(cols) == 11

    def test_last_tile_positions(self) -> None:
        """Last column at x=3488, last row at y=2488."""
        positions = compute_tile_grid(4000, 3000, 512, 0.25)
        max_x = max(p[0] for p in positions)
        max_y = max(p[1] for p in positions)
        assert max_x == 3488  # 4000 - 512
        assert max_y == 2488  # 3000 - 512

    def test_no_overlap(self) -> None:
        """1024x1024, no overlap → 4 tiles (2×2)."""
        positions = compute_tile_grid(1024, 1024, 512, 0.0)
        assert len(positions) == 4

    def test_too_small(self) -> None:
        """Image smaller than tile size → 0 tiles."""
        positions = compute_tile_grid(100, 100, 512, 0.25)
        assert len(positions) == 0

    def test_exact_tile_size(self) -> None:
        """Image exactly tile_size → 1 tile."""
        positions = compute_tile_grid(512, 512, 512, 0.25)
        assert len(positions) == 1
        assert positions[0] == (0, 0, 0, 0)

    def test_all_tiles_within_bounds(self) -> None:
        """Every tile fits within the image boundaries."""
        for w, h in [(4000, 3000), (1920, 1080), (800, 600)]:
            positions = compute_tile_grid(w, h, 512, 0.25)
            for x, y, _, _ in positions:
                assert x >= 0, f"x={x} < 0"
                assert y >= 0, f"y={y} < 0"
                assert x + 512 <= w, f"x+512={x + 512} > {w}"
                assert y + 512 <= h, f"y+512={y + 512} > {h}"

    def test_no_duplicate_positions(self) -> None:
        """No duplicate (x, y) positions in the grid."""
        positions = compute_tile_grid(4000, 3000, 512, 0.25)
        xy_pairs = [(p[0], p[1]) for p in positions]
        assert len(xy_pairs) == len(set(xy_pairs))


class TestExtractTile:
    """Tests for extract_tile."""

    def test_correct_region(self) -> None:
        """Extracted tile matches the expected region of the source."""
        img = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
        tile = extract_tile(img, 10, 20, 30)
        np.testing.assert_array_equal(tile, img[20:50, 10:40])

    def test_returns_copy(self) -> None:
        """Extracted tile is a copy, not a view."""
        img = np.zeros((100, 100), dtype=np.uint8)
        tile = extract_tile(img, 0, 0, 50)
        tile[:] = 255
        assert img[0, 0] == 0

    def test_colour_image(self) -> None:
        """Works with 3-channel images."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:20, 10:20] = [100, 150, 200]
        tile = extract_tile(img, 10, 10, 10)
        assert tile.shape == (10, 10, 3)
        np.testing.assert_array_equal(tile[0, 0], [100, 150, 200])


class TestComputeTileStats:
    """Tests for compute_tile_stats."""

    def test_uniform_white(self) -> None:
        """All-white tile: high brightness, low contrast, all valid."""
        tile = np.full((512, 512, 3), 255, dtype=np.uint8)
        stats = compute_tile_stats(tile)
        assert stats["brightness_mean"] == pytest.approx(255.0)
        assert stats["contrast_std"] == pytest.approx(0.0)
        assert stats["valid_fraction"] == pytest.approx(1.0)

    def test_all_black(self) -> None:
        """All-black tile: zero brightness, zero valid fraction."""
        tile = np.zeros((512, 512, 3), dtype=np.uint8)
        stats = compute_tile_stats(tile)
        assert stats["brightness_mean"] == pytest.approx(0.0)
        assert stats["valid_fraction"] == pytest.approx(0.0)

    def test_half_and_half(self) -> None:
        """Half-white, half-black: ~50% valid fraction, some contrast."""
        tile = np.zeros((512, 512, 3), dtype=np.uint8)
        tile[:, :256] = 255
        stats = compute_tile_stats(tile)
        assert stats["valid_fraction"] == pytest.approx(0.5, abs=0.01)
        assert stats["contrast_std"] > 0

    def test_checkerboard_has_edges(self) -> None:
        """Checkerboard pattern produces non-zero edge density."""
        tile = np.zeros((512, 512), dtype=np.uint8)
        tile[::2, ::2] = 255
        tile[1::2, 1::2] = 255
        stats = compute_tile_stats(tile)
        assert stats["edge_density"] > 0


class TestTileDroneImage:
    """Integration tests for tile_drone_image."""

    def test_synthetic_image(
        self, tmp_path: Path, synthetic_drone_image: np.ndarray
    ) -> None:
        """Tile a synthetic 4000x3000 image and verify output."""
        # Save synthetic image to disk
        img_path = tmp_path / "test_drone.jpg"
        cv2.imwrite(str(img_path), synthetic_drone_image)

        output_dir = tmp_path / "tiles"
        config = TilingConfig()
        tiles = tile_drone_image(img_path, "test123", config, output_dir)

        # Should produce 88 tiles total
        assert len(tiles) == 88

        # All tiles should be valid (no black borders in synthetic image)
        valid_tiles = [t for t in tiles if t.is_valid]
        assert len(valid_tiles) == 88

        # Check tile files exist
        for t in valid_tiles:
            tile_path = output_dir / f"{t.tile_id}.png"
            assert tile_path.exists(), f"Missing tile file: {tile_path}"
            tile_img = cv2.imread(str(tile_path))
            assert tile_img.shape == (512, 512, 3)

    def test_tile_id_format(
        self, tmp_path: Path, synthetic_drone_image: np.ndarray
    ) -> None:
        """Tile IDs follow expected naming convention."""
        img_path = tmp_path / "test_drone.jpg"
        cv2.imwrite(str(img_path), synthetic_drone_image)

        output_dir = tmp_path / "tiles"
        tiles = tile_drone_image(img_path, "abc123", TilingConfig(), output_dir)

        for t in tiles:
            assert t.tile_id.startswith("abc123_r")
            assert "_c" in t.tile_id

    def test_real_drone_image(
        self, sample_drone_jpg: Path, tmp_path: Path
    ) -> None:
        """Integration: tile a real drone image (skipped if data unavailable)."""
        output_dir = tmp_path / "tiles"
        tiles = tile_drone_image(sample_drone_jpg, "real_drone", TilingConfig(), output_dir)
        assert len(tiles) == 88
        valid_count = sum(1 for t in tiles if t.is_valid)
        assert valid_count > 0

    def test_nonexistent_image(self, tmp_path: Path) -> None:
        """Returns empty list for non-existent image."""
        tiles = tile_drone_image(
            tmp_path / "nope.jpg", "nope", TilingConfig(), tmp_path / "out"
        )
        assert tiles == []

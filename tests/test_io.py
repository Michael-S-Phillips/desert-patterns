"""Tests for src.utils.io — file scanning and ID generation."""

from pathlib import Path

import pytest

from src.utils.io import ImageFileInfo, generate_image_id, scan_directory


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mixed files for testing."""
    d = tmp_path / "images"
    d.mkdir()

    # Create test files
    (d / "photo1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    (d / "photo2.JPG").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)
    (d / "raw.DNG").write_bytes(b"\x00" * 50)
    (d / "video.MP4").write_bytes(b"\x00" * 50)
    (d / ".hidden.jpg").write_bytes(b"\x00" * 10)
    (d / ".DS_Store").write_bytes(b"\x00" * 10)
    (d / "readme.txt").write_text("test")

    return d


class TestScanDirectory:
    def test_extension_filtering(self, tmp_image_dir: Path) -> None:
        """Only files matching specified extensions are returned."""
        results = scan_directory(tmp_image_dir, extensions=[".jpg", ".JPG"])
        filenames = [r.filename for r in results]
        assert "photo1.jpg" in filenames
        assert "photo2.JPG" in filenames
        assert "raw.DNG" not in filenames
        assert "video.MP4" not in filenames
        assert "readme.txt" not in filenames

    def test_hidden_file_skipping(self, tmp_image_dir: Path) -> None:
        """Hidden files (starting with .) are skipped."""
        results = scan_directory(tmp_image_dir, extensions=[".jpg", ".JPG"])
        filenames = [r.filename for r in results]
        assert ".hidden.jpg" not in filenames
        assert ".DS_Store" not in filenames

    def test_all_files_when_no_filter(self, tmp_image_dir: Path) -> None:
        """All non-hidden files returned when no extension filter."""
        results = scan_directory(tmp_image_dir)
        filenames = [r.filename for r in results]
        assert len(filenames) == 5  # photo1.jpg, photo2.JPG, raw.DNG, video.MP4, readme.txt
        assert ".hidden.jpg" not in filenames

    def test_deterministic_sort(self, tmp_image_dir: Path) -> None:
        """Results are sorted by filename."""
        results = scan_directory(tmp_image_dir, extensions=[".jpg", ".JPG"])
        filenames = [r.filename for r in results]
        assert filenames == sorted(filenames)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        d = tmp_path / "empty"
        d.mkdir()
        results = scan_directory(d, extensions=[".jpg"])
        assert results == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Non-existent directory returns empty list."""
        results = scan_directory(tmp_path / "nope", extensions=[".jpg"])
        assert results == []

    def test_paths_with_spaces_and_parens(self, tmp_path: Path) -> None:
        """Handles directory names with spaces and parentheses."""
        d = tmp_path / "Big Pool 1 (drone)"
        d.mkdir()
        (d / "1 m DJI_0904.JPG").write_bytes(b"\xff\xd8" + b"\x00" * 50)
        results = scan_directory(d, extensions=[".jpg", ".JPG"])
        assert len(results) == 1
        assert results[0].filename == "1 m DJI_0904.JPG"

    def test_file_info_fields(self, tmp_image_dir: Path) -> None:
        """ImageFileInfo fields are correctly populated."""
        results = scan_directory(tmp_image_dir, extensions=[".jpg", ".JPG"])
        info = next(r for r in results if r.filename == "photo1.jpg")
        assert info.extension == ".jpg"
        assert info.file_size_bytes == 104
        assert info.path.is_absolute()


class TestGenerateImageId:
    def test_deterministic(self) -> None:
        """Same inputs produce the same ID."""
        id1 = generate_image_id("path/to/file.jpg", 12345)
        id2 = generate_image_id("path/to/file.jpg", 12345)
        assert id1 == id2

    def test_length(self) -> None:
        """IDs are exactly 12 hex characters."""
        img_id = generate_image_id("test.jpg", 100)
        assert len(img_id) == 12
        assert all(c in "0123456789abcdef" for c in img_id)

    def test_different_inputs_different_ids(self) -> None:
        """Different inputs produce different IDs."""
        id1 = generate_image_id("file_a.jpg", 100)
        id2 = generate_image_id("file_b.jpg", 100)
        id3 = generate_image_id("file_a.jpg", 200)
        assert id1 != id2
        assert id1 != id3

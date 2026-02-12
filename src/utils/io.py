"""File I/O utilities for the desert patterns pipeline."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageFileInfo:
    """Basic file-level information for an image."""

    path: Path
    relative_path: str
    filename: str
    extension: str
    file_size_bytes: int


def scan_directory(
    directory: Path,
    extensions: list[str] | None = None,
) -> list[ImageFileInfo]:
    """Scan a directory for image files, filtering by extension.

    Args:
        directory: Directory to scan.
        extensions: Allowed file extensions (e.g. [".jpg", ".JPG"]).
            If None, all files are returned.

    Returns:
        Sorted list of ImageFileInfo for matching files.
    """
    if not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return []

    ext_set = {e.lower() for e in extensions} if extensions else None
    results: list[ImageFileInfo] = []

    for path in directory.iterdir():
        # Skip hidden files and directories
        if path.name.startswith("."):
            continue
        if not path.is_file():
            continue
        if ext_set is not None and path.suffix.lower() not in ext_set:
            continue

        results.append(
            ImageFileInfo(
                path=path.resolve(),
                relative_path=str(path.relative_to(directory.parent) if directory.parent != path else path.name),
                filename=path.name,
                extension=path.suffix,
                file_size_bytes=path.stat().st_size,
            )
        )

    # Deterministic ordering: sort by filename
    results.sort(key=lambda f: f.filename)
    return results


def generate_image_id(relative_path: str, file_size: int) -> str:
    """Generate a stable image ID from relative path and file size.

    Uses SHA-256 of (relative_path + file_size), truncated to 12 hex chars.
    Stable across runs as long as the file hasn't moved or changed size.

    Args:
        relative_path: Path relative to data root.
        file_size: File size in bytes.

    Returns:
        12-character hexadecimal string.
    """
    content = f"{relative_path}:{file_size}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def ensure_directories(*paths: Path) -> None:
    """Create directories if they don't exist.

    Args:
        paths: Directory paths to create.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", path)

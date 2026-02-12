"""EXIF and XMP metadata extraction utilities."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# XMP tag patterns for DJI drone metadata
_XMP_RELATIVE_ALT = re.compile(r'drone-dji:RelativeAltitude="([^"]+)"')
_XMP_ABSOLUTE_ALT = re.compile(r'drone-dji:AbsoluteAltitude="([^"]+)"')
_XMP_GPS_LAT = re.compile(r'drone-dji:GpsLatitude="([^"]+)"')
_XMP_GPS_LON = re.compile(r'drone-dji:GpsLongitude="([^"]+)"')

# How many bytes to read from the file header for XMP extraction
_XMP_READ_BYTES = 200_000


@dataclass
class ExifData:
    """Parsed EXIF/XMP metadata from an image file."""

    camera_make: str | None = None
    camera_model: str | None = None
    datetime_original: datetime | None = None
    focal_length_mm: float | None = None
    image_width_px: int | None = None
    image_height_px: int | None = None
    exposure_time: float | None = None
    f_number: float | None = None
    iso: int | None = None

    # GPS from EXIF
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    gps_altitude: float | None = None

    # DJI XMP-specific fields
    xmp_relative_altitude: float | None = None
    xmp_absolute_altitude: float | None = None
    xmp_gps_latitude: float | None = None
    xmp_gps_longitude: float | None = None

    # Raw tag access for debugging
    raw_tags: dict[str, Any] = field(default_factory=dict, repr=False)


def extract_exif(image_path: Path) -> ExifData:
    """Extract EXIF and XMP metadata from an image file.

    Uses Pillow for standard EXIF tags and raw byte scanning for DJI XMP.

    Args:
        image_path: Path to the image file.

    Returns:
        ExifData with all extractable metadata.
    """
    data = ExifData()

    # Try Pillow EXIF first
    _extract_pillow_exif(image_path, data)

    # Try DJI XMP block (read first 200KB for embedded XML)
    _parse_dji_xmp(image_path, data)

    return data


def _extract_pillow_exif(image_path: Path, data: ExifData) -> None:
    """Extract standard EXIF tags using Pillow."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
    except ImportError:
        logger.warning("Pillow not available for EXIF extraction")
        return

    try:
        img = Image.open(image_path)
        data.image_width_px = img.width
        data.image_height_px = img.height

        exif_dict = img._getexif()
        if not exif_dict:
            return

        tags: dict[str, Any] = {}
        gps_info: dict[str, Any] = {}

        for tag_id, value in exif_dict.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if tag_name == "GPSInfo":
                for gps_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_id, str(gps_id))
                    gps_info[gps_tag] = gps_value
            else:
                tags[tag_name] = value

        data.raw_tags = tags
        data.camera_make = tags.get("Make")
        data.camera_model = tags.get("Model")
        data.focal_length_mm = _to_float(tags.get("FocalLength"))
        data.exposure_time = _to_float(tags.get("ExposureTime"))
        data.f_number = _to_float(tags.get("FNumber"))
        data.iso = tags.get("ISOSpeedRatings")
        data.datetime_original = _parse_timestamp(
            tags.get("DateTimeOriginal") or tags.get("DateTime")
        )

        if gps_info:
            data.gps_latitude = _parse_gps_coordinate(
                gps_info.get("GPSLatitude"),
                gps_info.get("GPSLatitudeRef"),
            )
            data.gps_longitude = _parse_gps_coordinate(
                gps_info.get("GPSLongitude"),
                gps_info.get("GPSLongitudeRef"),
            )
            data.gps_altitude = _to_float(gps_info.get("GPSAltitude"))

    except Exception:
        logger.warning("Failed to extract Pillow EXIF from %s", image_path, exc_info=True)


def _parse_dji_xmp(image_path: Path, data: ExifData) -> None:
    """Parse DJI-specific XMP metadata from file header bytes.

    DJI drones embed XMP XML in the JPEG header. We read the first 200KB
    and search for drone-dji: namespace tags.
    """
    try:
        with open(image_path, "rb") as f:
            header = f.read(_XMP_READ_BYTES)

        # Decode as latin-1 (safe for binary with ASCII XML embedded)
        text = header.decode("latin-1")

        match = _XMP_RELATIVE_ALT.search(text)
        if match:
            data.xmp_relative_altitude = _parse_signed_float(match.group(1))

        match = _XMP_ABSOLUTE_ALT.search(text)
        if match:
            data.xmp_absolute_altitude = _parse_signed_float(match.group(1))

        match = _XMP_GPS_LAT.search(text)
        if match:
            data.xmp_gps_latitude = _parse_signed_float(match.group(1))

        match = _XMP_GPS_LON.search(text)
        if match:
            data.xmp_gps_longitude = _parse_signed_float(match.group(1))

    except Exception:
        logger.debug("No DJI XMP data in %s", image_path, exc_info=True)


def _parse_gps_coordinate(
    dms_tuple: tuple | None,
    ref: str | None,
) -> float | None:
    """Convert EXIF GPS DMS (degrees, minutes, seconds) to decimal degrees.

    Args:
        dms_tuple: Tuple of (degrees, minutes, seconds) as IFDRational or float.
        ref: Reference direction ('N', 'S', 'E', 'W').

    Returns:
        Decimal degrees (negative for S/W), or None if input is invalid.
    """
    if dms_tuple is None or len(dms_tuple) != 3:
        return None

    try:
        degrees = float(dms_tuple[0])
        minutes = float(dms_tuple[1])
        seconds = float(dms_tuple[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0

        if ref in ("S", "W"):
            decimal = -decimal

        return decimal
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse EXIF timestamp string to datetime.

    Handles the standard EXIF format "YYYY:MM:DD HH:MM:SS".
    """
    if value is None:
        return None

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue

    logger.debug("Could not parse timestamp: %r", value)
    return None


def _parse_signed_float(value: str) -> float | None:
    """Parse a signed float string like '+240.20' or '-24.0953266'."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _to_float(value: Any) -> float | None:
    """Convert an EXIF value (possibly IFDRational) to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None

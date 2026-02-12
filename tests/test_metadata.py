"""Tests for src.data.metadata_extractor — altitude parsing, classification, temporal phase, GSD."""

from datetime import date, datetime
from pathlib import Path

import pytest

from src.data.metadata_extractor import (
    AltitudeGroupConfig,
    DataConfig,
    MetadataExtractor,
    RainEventConfig,
    _parse_altitude_from_filename,
    assign_temporal_phase,
    classify_altitude_group,
    compute_gsd,
    load_config,
)


# --- Altitude parsing from filenames ---


class TestParseAltitudeFromFilename:
    """Test the two filename altitude conventions."""

    # Convention A: "N m DJI_XXXX.JPG"
    def test_convention_a_1m(self) -> None:
        assert _parse_altitude_from_filename("1 m DJI_0904.JPG") == 1.0

    def test_convention_a_240m(self) -> None:
        assert _parse_altitude_from_filename("240 m DJI_0899.JPG") == 240.0

    def test_convention_a_120m(self) -> None:
        assert _parse_altitude_from_filename("120 m DJI_0900.JPG") == 120.0

    def test_convention_a_5m(self) -> None:
        assert _parse_altitude_from_filename("5 m DJI_0903.JPG") == 5.0

    # Convention B: "N_DJI_XXXX.JPG" or "Nb_DJI_XXXX.JPG"
    def test_convention_b_numeric(self) -> None:
        assert _parse_altitude_from_filename("1_DJI_0311.JPG") == 1.0

    def test_convention_b_with_letter(self) -> None:
        assert _parse_altitude_from_filename("1b_DJI_0312.JPG") == 1.0

    def test_convention_b_120(self) -> None:
        assert _parse_altitude_from_filename("120_DJI_0307.JPG") == 120.0

    def test_convention_b_240(self) -> None:
        assert _parse_altitude_from_filename("240_DJI_0306.JPG") == 240.0

    # No altitude in filename
    def test_no_prefix_dji(self) -> None:
        assert _parse_altitude_from_filename("DJI_0897.JPG") is None

    def test_iphone_filename(self) -> None:
        assert _parse_altitude_from_filename("IMG_8143.jpg") is None


# --- Altitude group classification ---

_TEST_GROUPS = {
    "high": AltitudeGroupConfig(min=100, max=999),
    "mid": AltitudeGroupConfig(min=20, max=100),
    "low": AltitudeGroupConfig(min=1, max=20),
    "ground": AltitudeGroupConfig(min=0, max=1),
}


class TestClassifyAltitudeGroup:
    def test_high_altitude(self) -> None:
        assert classify_altitude_group(240.0, _TEST_GROUPS) == "high"

    def test_mid_altitude(self) -> None:
        assert classify_altitude_group(60.0, _TEST_GROUPS) == "mid"

    def test_low_altitude(self) -> None:
        assert classify_altitude_group(5.0, _TEST_GROUPS) == "low"

    def test_ground(self) -> None:
        assert classify_altitude_group(0.5, _TEST_GROUPS) == "ground"

    # Boundary values (inclusive)
    def test_boundary_high_min(self) -> None:
        assert classify_altitude_group(100.0, _TEST_GROUPS) == "high"

    def test_boundary_mid_max(self) -> None:
        """100m falls into high (checked first) since boundaries overlap at 100."""
        assert classify_altitude_group(100.0, _TEST_GROUPS) == "high"

    def test_boundary_mid_min(self) -> None:
        assert classify_altitude_group(20.0, _TEST_GROUPS) == "mid"

    def test_boundary_low_max(self) -> None:
        """20m falls into mid (checked first) since boundaries overlap at 20."""
        assert classify_altitude_group(20.0, _TEST_GROUPS) == "mid"

    def test_boundary_low_min(self) -> None:
        assert classify_altitude_group(1.0, _TEST_GROUPS) == "low"

    def test_boundary_ground_max(self) -> None:
        """1m falls into low (checked first) since boundaries overlap at 1."""
        assert classify_altitude_group(1.0, _TEST_GROUPS) == "low"

    def test_none_altitude(self) -> None:
        assert classify_altitude_group(None, _TEST_GROUPS) is None


# --- Temporal phase assignment ---


class TestAssignTemporalPhase:
    _RAIN = RainEventConfig(start=date(2023, 8, 24), end=date(2023, 8, 28))

    def test_pre_rain(self) -> None:
        ts = datetime(2023, 8, 23, 12, 0)
        assert assign_temporal_phase(ts, self._RAIN) == "pre_rain"

    def test_during_rain_start(self) -> None:
        ts = datetime(2023, 8, 24, 0, 0)
        assert assign_temporal_phase(ts, self._RAIN) == "during_rain"

    def test_during_rain_middle(self) -> None:
        ts = datetime(2023, 8, 26, 15, 30)
        assert assign_temporal_phase(ts, self._RAIN) == "during_rain"

    def test_during_rain_end(self) -> None:
        ts = datetime(2023, 8, 28, 23, 59)
        assert assign_temporal_phase(ts, self._RAIN) == "during_rain"

    def test_post_rain(self) -> None:
        ts = datetime(2023, 8, 29, 8, 0)
        assert assign_temporal_phase(ts, self._RAIN) == "post_rain"


# --- GSD computation ---


class TestComputeGSD:
    def test_known_values(self) -> None:
        """GSD formula: (alt * sensor_w) / (focal * img_w) * 100 cm/px."""
        # At 100m altitude, 6.17mm sensor, 4.58mm focal, 4000px width
        gsd = compute_gsd(
            altitude_m=100.0,
            focal_length_mm=4.58,
            image_width_px=4000,
            sensor_width_mm=6.17,
        )
        expected = (100.0 * 6.17) / (4.58 * 4000) * 100.0
        assert gsd == pytest.approx(expected)

    def test_1m_altitude(self) -> None:
        """At 1m, GSD should be very small."""
        gsd = compute_gsd(1.0, 4.58, 4000, 6.17)
        assert gsd < 0.05  # sub-millimeter

    def test_240m_altitude(self) -> None:
        """At 240m, GSD should be several cm."""
        gsd = compute_gsd(240.0, 4.58, 4000, 6.17)
        assert 5.0 < gsd < 15.0


# --- Config loading ---


class TestLoadConfig:
    def test_load_default_config(self, project_root: Path) -> None:
        """Default data_config.yaml loads without error."""
        config_path = project_root / "configs" / "data_config.yaml"
        if not config_path.exists():
            pytest.skip("Config file not available")
        config = load_config(config_path)
        assert len(config.sources) == 6
        assert config.rain_event.start == date(2023, 8, 24)
        assert config.rain_event.end == date(2023, 8, 28)
        assert "high" in config.altitude_groups


# --- Integration tests (require actual data) ---


class TestIntegration:
    """Integration tests that need the actual image data."""

    @pytest.fixture
    def config(self, project_root: Path) -> DataConfig:
        config_path = project_root / "configs" / "data_config.yaml"
        if not config_path.exists():
            pytest.skip("Config not available")
        return load_config(config_path)

    @pytest.fixture
    def catalog(self, config: DataConfig, project_root: Path):
        data_root = project_root / config.data_root
        if not data_root.exists():
            pytest.skip("Data not available")
        extractor = MetadataExtractor(config, project_root, skip_quality=True)
        return extractor.extract_all()

    def test_total_image_count(self, catalog) -> None:
        """Catalog should contain exactly 195 JPG images."""
        assert len(catalog) == 195

    def test_no_duplicate_ids(self, catalog) -> None:
        """All image IDs must be unique."""
        assert catalog["image_id"].nunique() == len(catalog)

    def test_all_paths_exist(self, catalog) -> None:
        """All image_path values point to existing files."""
        for path_str in catalog["image_path"]:
            assert Path(path_str).exists(), f"Missing: {path_str}"

    def test_drone_count(self, catalog) -> None:
        """Correct number of drone images."""
        drone = catalog[catalog["source_type"] == "drone"]
        assert len(drone) == 25

    def test_ground_count(self, catalog) -> None:
        """Correct number of ground images."""
        ground = catalog[catalog["source_type"] == "ground"]
        assert len(ground) == 170

    def test_drone_has_gps(self, catalog) -> None:
        """All drone images should have GPS coordinates."""
        drone = catalog[catalog["source_type"] == "drone"]
        assert drone["lat"].notna().all()
        assert drone["lon"].notna().all()

    def test_drone_has_altitude(self, catalog) -> None:
        """All drone images should have altitude."""
        drone = catalog[catalog["source_type"] == "drone"]
        assert drone["altitude_m"].notna().all()

    def test_drone_has_gsd(self, catalog) -> None:
        """All drone images with altitude should have GSD."""
        drone = catalog[catalog["source_type"] == "drone"]
        assert drone["gsd_cm_per_px"].notna().all()

    def test_drone_has_timestamps(self, catalog) -> None:
        """All drone images should have timestamps."""
        drone = catalog[catalog["source_type"] == "drone"]
        assert drone["timestamp"].notna().all()

    def test_site_names_valid(self, catalog) -> None:
        """Site names match expected values."""
        sites = set(catalog["site_name"].unique())
        assert sites == {"big_pool", "biofilm_pool", "mudcrack"}

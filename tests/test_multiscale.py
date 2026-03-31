"""Tests for multi-scale segmentation helpers (no torch dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.generate_multiscale_segmentations import (
    compute_altitude_summary,
    group_images_by_site_altitude,
    select_representative_images,
)


def _make_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a1", "a2", "b1", "b2", "b3"],
            "site_name": ["big_pool", "big_pool", "big_pool", "big_pool", "biofilm_pool"],
            "altitude_m": [1.0, 5.0, 1.0, 1.0, 20.0],
        }
    )


def test_group_by_site_altitude():
    """group_images_by_site_altitude returns {site: {alt: [rows]}} structure."""
    df = _make_catalog()
    groups = group_images_by_site_altitude(df)
    assert set(groups.keys()) == {"big_pool", "biofilm_pool"}
    assert set(groups["big_pool"].keys()) == {1.0, 5.0}
    # a1, b1, b2 all have site=big_pool, altitude=1.0
    assert len(groups["big_pool"][1.0]) == 3
    assert len(groups["big_pool"][5.0]) == 1
    assert len(groups["biofilm_pool"][20.0]) == 1


def test_representative_image_selection():
    """When multiple images share altitude, first by image_id is selected."""
    df = _make_catalog()
    groups = group_images_by_site_altitude(df)
    reps = select_representative_images(groups["big_pool"])
    # For altitude 1.0m: three rows with image_ids "a1", "b1", "b2"; "a1" < "b1" < "b2"
    assert reps[1.0]["image_id"] == "a1"
    assert reps[5.0]["image_id"] == "a2"


def test_multiscale_summary_keys():
    """compute_altitude_summary produces expected keys for SAM results."""
    from unittest.mock import MagicMock

    def make_result(n_instances, mask_fraction, entropy):
        r = MagicMock()
        r.instance_masks = [np.ones((10, 10), dtype=bool)] * n_instances
        r.attention_entropy = entropy
        r.attention_map = np.full((10, 10), mask_fraction, dtype=np.float32)
        return r

    results_by_altitude = {
        1.0: [make_result(3, 0.2, 5.1), make_result(2, 0.3, 4.9)],
        5.0: [make_result(5, 0.4, 6.0)],
    }

    summary = compute_altitude_summary(results_by_altitude, include_entropy=True)
    assert set(summary.keys()) == {"1.0m", "5.0m"}
    assert "n_images" in summary["1.0m"]
    assert "mean_n_instances" in summary["1.0m"]
    assert "mean_mask_area_fraction" in summary["1.0m"]
    assert "mean_attention_entropy" in summary["1.0m"]
    assert summary["1.0m"]["n_images"] == 2
    assert summary["5.0m"]["n_images"] == 1
    # No entropy for CLASP
    summary_no_entropy = compute_altitude_summary(results_by_altitude, include_entropy=False)
    assert "mean_attention_entropy" not in summary_no_entropy["1.0m"]

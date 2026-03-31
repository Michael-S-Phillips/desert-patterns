"""Multi-scale segmentation analysis across site altitudes.

Groups images by (site_name, altitude_m) from the image catalog, runs the
configured segmenter at each altitude, and produces comparison figures and
per-site JSON summaries.

Usage::

    python scripts/generate_multiscale_segmentations.py \\
        --config configs/classifier_config.yaml \\
        --segmenter sam \\
        --site big_pool \\
        --force

"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions (pure Python, no torch dependency — independently testable)
# ---------------------------------------------------------------------------


def group_images_by_site_altitude(
    catalog: pd.DataFrame,
) -> dict[str, dict[float, list[dict]]]:
    """Group catalog rows by (site_name, altitude_m).

    Args:
        catalog: DataFrame with at least ``site_name`` and ``altitude_m`` columns.

    Returns:
        Nested dict ``{site_name: {altitude_m: [row_dicts]}}``.
    """
    result: dict[str, dict[float, list[dict]]] = {}
    for _, row in catalog.iterrows():
        site = str(row["site_name"])
        alt = float(row["altitude_m"])
        result.setdefault(site, {}).setdefault(alt, []).append(row.to_dict())
    return result


def select_representative_images(
    altitude_groups: dict[float, list[dict]],
) -> dict[float, dict]:
    """Select one representative image per altitude (first by image_id).

    Args:
        altitude_groups: ``{altitude_m: [row_dicts]}`` for a single site.

    Returns:
        ``{altitude_m: row_dict}`` — one row per distinct altitude.
    """
    reps: dict[float, dict] = {}
    for alt, rows in altitude_groups.items():
        reps[alt] = min(rows, key=lambda r: str(r["image_id"]))
    return reps


def compute_altitude_summary(
    results_by_altitude: dict[float, list[Any]],
    include_entropy: bool = True,
) -> dict[str, dict]:
    """Compute per-altitude summary statistics from segmentation results.

    Args:
        results_by_altitude: ``{altitude_m: [SegmentationResult or ClaspResult]}``.
        include_entropy: If True, includes ``mean_attention_entropy`` (SAM path).
            Pass False for CLASP results that have no attention map.

    Returns:
        Dict keyed by ``"{altitude_m}m"`` with per-altitude stats.
    """
    summary: dict[str, dict] = {}
    for alt, results in results_by_altitude.items():
        key = f"{alt}m"
        n_instances = [len(r.instance_masks) for r in results]
        area_fractions = [float(r.attention_map.mean()) for r in results]
        entry: dict = {
            "n_images": len(results),
            "mean_n_instances": float(np.mean(n_instances)),
            "mean_mask_area_fraction": float(np.mean(area_fractions)),
        }
        if include_entropy:
            entropies = [r.attention_entropy for r in results]
            entry["mean_attention_entropy"] = float(np.mean(entropies))
        summary[key] = entry
    return summary


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

# Colourblind-safe palette (Wong 2011)
_WONG_COLORS = [
    "#0072B2", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#D55E00", "#CC79A7", "#000000",
]


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def _save_per_image_outputs(
    result: Any,
    output_dir: Path,
    stem: str,
    segmenter_type: str,
) -> None:
    """Save overlay and (optionally) attention map PNGs for a single image."""
    from PIL import Image as PILImage

    output_dir.mkdir(parents=True, exist_ok=True)

    pil_orig = PILImage.open(result.image_path).convert("RGB")
    overlay = np.array(pil_orig)
    for i, mask in enumerate(result.instance_masks):
        color = tuple(
            int(c * 255)
            for c in _hex_to_rgb(_WONG_COLORS[i % len(_WONG_COLORS)])
        )
        overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    PILImage.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")

    if segmenter_type == "sam" and result.attention_map is not None:
        attn_uint8 = (result.attention_map * 255).astype(np.uint8)
        PILImage.fromarray(attn_uint8, mode="L").save(output_dir / f"{stem}_attn.png")


def generate_comparison_figure(
    site_name: str,
    representatives: dict[float, dict],
    results_by_altitude: dict[float, Any],
    output_dir: Path,
    segmenter_type: str,
) -> None:
    """Generate side-by-side comparison figure for one site.

    SAM: columns = (thumbnail, attention map, mask overlay); rows = altitude.
    CLASP: columns = (thumbnail, label map overlay); rows = altitude.

    Saved as 300 DPI PNG + SVG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    altitudes = sorted(representatives.keys())
    n_rows = len(altitudes)
    n_cols = 3 if segmenter_type == "sam" else 2

    fig_width = 7.0
    cell_h = fig_width / n_cols
    fig_height = cell_h * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Thumbnail", "Attention Map", "Mask Overlay"] if segmenter_type == "sam" else ["Thumbnail", "Label Map"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=9, fontfamily="DejaVu Sans")

    for row_idx, alt in enumerate(altitudes):
        result = results_by_altitude.get(alt)
        rep_row = representatives[alt]
        image_path = rep_row.get("file_path", rep_row.get("image_path", ""))

        ax_row = axes[row_idx]
        for ax in ax_row:
            ax.set_axis_off()
        ax_row[0].set_ylabel(f"{alt:.0f}m", fontsize=8, rotation=0, labelpad=30)

        # Column 0: thumbnail
        try:
            thumb = PILImage.open(image_path).convert("RGB")
            thumb.thumbnail((400, 400))
            ax_row[0].imshow(np.asarray(thumb))
        except Exception:
            ax_row[0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_row[0].transAxes)

        if result is None:
            continue

        if segmenter_type == "sam":
            if result.attention_map is not None:
                ax_row[1].imshow(result.attention_map, cmap="cividis", vmin=0, vmax=1)
            pil_orig = PILImage.open(result.image_path).convert("RGB")
            overlay = np.array(pil_orig)
            for i, mask in enumerate(result.instance_masks):
                color = tuple(
                    int(c * 255)
                    for c in _hex_to_rgb(_WONG_COLORS[i % len(_WONG_COLORS)])
                )
                overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            ax_row[2].imshow(overlay)
        else:
            if hasattr(result, "label_map") and result.label_map is not None:
                ax_row[1].imshow(result.label_map, cmap="tab20")

    fig.suptitle(f"{site_name} — multi-scale segmentation", fontsize=10, fontfamily="DejaVu Sans")
    plt.tight_layout()

    out_base = output_dir / f"{site_name}_comparison"
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison figure: %s.png/.svg", out_base)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_site(
    site_name: str,
    altitude_groups: dict[float, list[dict]],
    segmenter_type: str,
    seg_config: Any,
    dino_config: Any,
    output_base: Path,
    force: bool,
    verbose: bool,
) -> None:
    """Run segmentation for all images in a site and produce outputs."""
    if segmenter_type == "sam":
        from dataclasses import replace
        from src.segmentation.segment import PatternSegmenter
        # Multiscale script has no classifier — always use self_attention
        if seg_config.attention_mode == "classifier":
            seg_config = replace(seg_config, attention_mode="self_attention")
        segmenter = PatternSegmenter(None, seg_config, dino_config)
        include_entropy = True
    else:
        from src.segmentation.clasp import ClaspSegmenter
        segmenter = ClaspSegmenter(seg_config)
        include_entropy = False

    results_by_altitude: dict[float, list[Any]] = {}

    for alt in sorted(altitude_groups.keys()):
        rows = altitude_groups[alt]
        alt_output_dir = output_base / site_name / f"{alt:.0f}m"
        alt_results: list[Any] = []

        for row in rows:
            image_path = Path(str(row.get("file_path", row.get("image_path", ""))))
            stem = image_path.stem
            overlay_path = alt_output_dir / f"{stem}_overlay.png"

            if overlay_path.exists() and not force:
                logger.info("Skipping %s (already segmented)", image_path.name)
                continue

            logger.info("Segmenting %s at %sm (%s)", image_path.name, alt, segmenter_type)
            try:
                if segmenter_type == "sam":
                    result = segmenter.segment(image_path, class_name="")
                else:
                    result = segmenter.segment(image_path)
                alt_results.append(result)
                _save_per_image_outputs(result, alt_output_dir, stem, segmenter_type)
            except Exception as exc:
                logger.warning("Failed to segment %s: %s", image_path.name, exc)

        results_by_altitude[alt] = alt_results

    representatives = select_representative_images(altitude_groups)
    rep_results = {
        alt: results_by_altitude[alt][0]
        for alt in sorted(results_by_altitude)
        if results_by_altitude[alt]
    }
    generate_comparison_figure(site_name, representatives, rep_results, output_base, segmenter_type)

    summary = compute_altitude_summary(results_by_altitude, include_entropy=include_entropy)
    summary_path = output_base / f"{site_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved summary: %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scale segmentation analysis")
    parser.add_argument("--config", required=True, help="Path to classifier_config.yaml")
    parser.add_argument(
        "--segmenter", choices=["sam", "clasp"], default="sam",
        help="Segmenter to use (default: sam)",
    )
    parser.add_argument(
        "--site", default="all",
        help="Site to process: big_pool | biofilm_pool | jbio | all (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import yaml
    from src.features.dino_embeddings import load_dino_config
    from src.segmentation.segment import load_segmentation_config

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    seg_config = load_segmentation_config(config_dict)
    dino_config = load_dino_config(config_dict.get("dino", {}))

    catalog_path = Path("data/metadata/image_catalog.csv")
    catalog = pd.read_csv(catalog_path)

    site_filter = args.site
    if site_filter == "jbio":
        site_filter = "biofilm_pool"

    groups = group_images_by_site_altitude(catalog)

    output_base = Path("outputs/segmentations_multiscale")
    output_base.mkdir(parents=True, exist_ok=True)

    sites_to_run = list(groups.keys()) if site_filter == "all" else [site_filter]
    for site_name in sites_to_run:
        if site_name not in groups:
            logger.warning("Site %r not found in catalog.", site_name)
            continue
        logger.info("=== Processing site: %s ===", site_name)
        run_site(
            site_name=site_name,
            altitude_groups=groups[site_name],
            segmenter_type=args.segmenter,
            seg_config=seg_config,
            dino_config=dino_config,
            output_base=output_base,
            force=args.force,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()

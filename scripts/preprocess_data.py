#!/usr/bin/env python3
"""Preprocessing script — Phase 2 of the desert patterns pipeline.

Tiles drone images, masks ground images, and standardizes all patches
to 518x518 for DINOv3 feature extraction.

Usage:
    python scripts/preprocess_data.py --config configs/data_config.yaml
    python scripts/preprocess_data.py --config configs/data_config.yaml --no-sam
    python scripts/preprocess_data.py --config configs/data_config.yaml --drone-only --verbose
    python scripts/preprocess_data.py --config configs/data_config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PreprocessingPipeline, load_preprocessing_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess images: tile drones, mask ground, standardize patches."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_config.yaml",
        help="Path to data configuration YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report but do not write output files.",
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Skip SAM masking; use horizon/crop fallbacks only.",
    )
    parser.add_argument(
        "--drone-only",
        action="store_true",
        help="Only process drone images (tiling).",
    )
    parser.add_argument(
        "--ground-only",
        action="store_true",
        help="Only process ground images (masking).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    logger.info("Loading config from %s", config_path)

    # Override SAM checkpoint if --no-sam
    config = load_preprocessing_config(config_path, PROJECT_ROOT)
    if args.no_sam:
        config.ground_masking.sam_checkpoint = None

    # Run pipeline
    pipeline = PreprocessingPipeline(config)
    manifest = pipeline.run(
        dry_run=args.dry_run,
        no_sam=args.no_sam,
        drone_only=args.drone_only,
        ground_only=args.ground_only,
    )

    # Log summary
    _log_summary(manifest)
    logger.info("Done.")


def _log_summary(manifest) -> None:
    """Log summary statistics about the preprocessing results."""
    logger.info("=== Preprocessing Summary ===")
    logger.info("Total patches: %d", len(manifest))

    if len(manifest) == 0:
        return

    if "source_type" in manifest.columns:
        logger.info("By source type:")
        for st, count in manifest["source_type"].value_counts().items():
            logger.info("  %s: %d", st, count)

    if "processing_method" in manifest.columns:
        logger.info("By processing method:")
        for pm, count in manifest["processing_method"].value_counts().items():
            logger.info("  %s: %d", pm, count)

    if "quality_flag" in manifest.columns:
        flagged = manifest[manifest["quality_flag"] != "good"]
        if len(flagged) > 0:
            logger.info("Non-good quality flags:")
            for qf, count in flagged["quality_flag"].value_counts().items():
                logger.info("  %s: %d", qf, count)


if __name__ == "__main__":
    main()

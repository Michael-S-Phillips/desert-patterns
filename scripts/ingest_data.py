#!/usr/bin/env python3
"""Data ingestion script — Phase 1 of the desert patterns pipeline.

Scans configured image directories, extracts metadata, computes quality metrics,
and produces the master image catalog CSV.

Usage:
    python scripts/ingest_data.py --config configs/data_config.yaml
    python scripts/ingest_data.py --config configs/data_config.yaml --dry-run
    python scripts/ingest_data.py --config configs/data_config.yaml --skip-quality --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_extractor import (
    MetadataExtractor,
    load_config,
    save_catalog,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest image data and build the master catalog."
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
        help="Scan and report but do not write the catalog CSV.",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip image quality assessment (faster).",
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
    config = load_config(config_path)

    # Run extraction
    extractor = MetadataExtractor(
        config=config,
        project_root=PROJECT_ROOT,
        skip_quality=args.skip_quality,
    )
    df = extractor.extract_all()

    # Log summary statistics
    _log_summary(df)

    # Save catalog
    if args.dry_run:
        logger.info("Dry run — catalog not saved.")
    else:
        output_path = PROJECT_ROOT / config.catalog_output
        save_catalog(df, output_path)

    logger.info("Done.")


def _log_summary(df) -> None:
    """Log summary statistics about the catalog."""
    logger.info("=== Catalog Summary ===")
    logger.info("Total images: %d", len(df))

    if "source_type" in df.columns:
        logger.info("By source type:")
        for st, count in df["source_type"].value_counts().items():
            logger.info("  %s: %d", st, count)

    if "site_name" in df.columns:
        logger.info("By site:")
        for site, count in df["site_name"].value_counts().items():
            logger.info("  %s: %d", site, count)

    if "altitude_group" in df.columns:
        logger.info("By altitude group:")
        for ag, count in df["altitude_group"].value_counts(dropna=False).items():
            label = ag if ag is not None else "N/A"
            logger.info("  %s: %d", label, count)

    if "temporal_phase" in df.columns:
        logger.info("By temporal phase:")
        for tp, count in df["temporal_phase"].value_counts().items():
            logger.info("  %s: %d", tp, count)

    if "quality_flag" in df.columns:
        logger.info("By quality flag:")
        for qf, count in df["quality_flag"].value_counts().items():
            logger.info("  %s: %d", qf, count)

    if "has_dng_pair" in df.columns:
        dng_count = df["has_dng_pair"].sum()
        logger.info("Images with DNG pair: %d", dng_count)


if __name__ == "__main__":
    main()

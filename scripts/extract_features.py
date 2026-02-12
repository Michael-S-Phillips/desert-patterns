#!/usr/bin/env python3
"""Feature extraction script — Phase 3 of the desert patterns pipeline.

Extracts DINOv3 embeddings, classical texture descriptors, and fuses them
into a single feature vector stored in HDF5.

Usage:
    python scripts/extract_features.py --config configs/feature_config.yaml
    python scripts/extract_features.py --config configs/feature_config.yaml --texture-only
    python scripts/extract_features.py --config configs/feature_config.yaml --dino-only --verbose
    python scripts/extract_features.py --config configs/feature_config.yaml --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_store import FeatureStore
from src.features.fusion import FusionConfig, fuse_features, load_fusion_config
from src.features.texture_descriptors import (
    TextureDescriptorExtractor,
    load_texture_config,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features: DINOv3 embeddings, texture descriptors, and fusion."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "feature_config.yaml",
        help="Path to feature configuration YAML file.",
    )
    parser.add_argument(
        "--dino-only",
        action="store_true",
        help="Only extract DINOv3 embeddings (skip texture and fusion).",
    )
    parser.add_argument(
        "--texture-only",
        action="store_true",
        help="Only extract texture descriptors (skip DINOv3 and fusion).",
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help="Skip fusion step (extract both feature types but don't fuse).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override DINOv3 batch size from config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract features even if they already exist in the store.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
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
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Resolve paths relative to project root
    manifest_path = PROJECT_ROOT / raw_config["manifest_path"]
    store_path = PROJECT_ROOT / raw_config["feature_store"]
    output_dir = PROJECT_ROOT / raw_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    logger.info("Loading preprocessing manifest from %s", manifest_path)
    manifest = pd.read_csv(manifest_path)
    # Column may be called "standardized_path" or "patch_path" depending on preprocessor version
    path_col = "standardized_path" if "standardized_path" in manifest.columns else "patch_path"
    image_paths = [Path(p) for p in manifest[path_col]]
    image_ids = manifest["patch_id"].tolist()
    logger.info("Found %d patches to process", len(image_paths))

    # Resolve relative paths against project root
    image_paths = [
        p if p.is_absolute() else PROJECT_ROOT / p for p in image_paths
    ]

    # Init feature store
    store = FeatureStore(store_path)

    # Save image IDs
    store.save_image_ids(image_ids)

    # --- DINOv3 extraction ---
    do_dino = not args.texture_only
    if do_dino and (args.force or not store.has_dino()):
        logger.info("=== Extracting DINOv3 embeddings ===")
        from src.features.dino_embeddings import DinoFeatureExtractor, load_dino_config

        dino_config = load_dino_config(raw_config.get("dino", {}))
        if args.batch_size is not None:
            dino_config.batch_size = args.batch_size

        extractor = DinoFeatureExtractor(dino_config)
        dino_embeddings = extractor.extract_batch(image_paths, dino_config.batch_size)
        store.save_dino(dino_embeddings)
        logger.info("Saved DINOv3 embeddings: %s", dino_embeddings.shape)
    elif do_dino:
        logger.info("DINOv3 embeddings already in store (use --force to re-extract)")

    # --- Texture extraction ---
    do_texture = not args.dino_only
    if do_texture and (args.force or not store.has_texture()):
        logger.info("=== Extracting texture descriptors ===")
        texture_config = load_texture_config(raw_config.get("texture", {}))
        tex_extractor = TextureDescriptorExtractor(texture_config)
        checkpoint = output_dir / "texture_checkpoint.npy"
        texture_features = tex_extractor.extract_batch(
            image_paths, checkpoint_path=checkpoint, checkpoint_interval=50,
        )
        store.save_texture(texture_features, tex_extractor.FEATURE_NAMES)
        logger.info("Saved texture features: %s", texture_features.shape)
    elif do_texture:
        logger.info("Texture features already in store (use --force to re-extract)")

    # --- Fusion ---
    do_fusion = not args.no_fusion and not args.dino_only and not args.texture_only
    if do_fusion and store.has_dino() and store.has_texture():
        if args.force or not store.has_fused():
            logger.info("=== Fusing features ===")
            fusion_config = load_fusion_config(raw_config.get("fusion", {}))
            dino_data = store.load_dino()
            texture_data = store.load_texture()
            result = fuse_features(dino_data, texture_data, fusion_config)
            store.save_fused(result.fused, result.dino_pca, result.texture_pca)
            logger.info(
                "Saved fused features: %s (dino_pca=%s, texture_pca=%s)",
                result.fused.shape,
                result.dino_pca.shape,
                result.texture_pca.shape,
            )
        else:
            logger.info("Fused features already in store (use --force to re-fuse)")
    elif do_fusion:
        logger.info("Cannot fuse: need both DINOv3 and texture features in store")

    # --- Validation ---
    issues = store.validate()
    if issues:
        logger.warning("Feature store validation issues:")
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("Feature store validation: all clean")

    # --- Summary ---
    _log_summary(store)
    logger.info("Done.")


def _log_summary(store: FeatureStore) -> None:
    """Log a summary of the feature store contents."""
    logger.info("=== Feature Store Summary ===")
    logger.info("Path: %s", store.store_path)
    logger.info("Images: %d", store.get_n_images())
    logger.info("Has DINOv3: %s", store.has_dino())
    logger.info("Has texture: %s", store.has_texture())
    logger.info("Has fused: %s", store.has_fused())


if __name__ == "__main__":
    main()

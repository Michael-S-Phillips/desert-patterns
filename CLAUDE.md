# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Unsupervised deep learning pipeline for discovering and characterizing patterned ground
types in Atacama Desert (Yungay basin) imagery. Supports microbial ecology research by
revealing natural pattern groupings without human-imposed categories.

This is NOT a supervised classification project. No training loop. No labels. Feature
extraction uses pretrained DINOv3 (inference only).

## Pipeline Architecture

The pipeline has five sequential stages:

1. **Ingest** — scan image directories, extract EXIF metadata (GPS, altitude, timestamp),
   compute GSD, assign temporal phases (pre/during/post rain), quality-flag images
2. **Preprocess** — tile drone images into overlapping 512px patches, mask ground in
   ground-level photos (SAM with horizon-detection and lower-crop fallbacks),
   standardize all to 518x518 for DINOv3
3. **Extract features** — DINOv3 ViT-B/16 [CLS] embeddings (768-d) + classical texture
   descriptors (~90 features: GLCM, Gabor, fractal dimension, lacunarity, LBP, global stats).
   Fuse via L2-normalize -> PCA (95% variance) -> weighted concat (default 70/30 DINOv3/texture)
4. **Cluster** — UMAP dimensionality reduction (cosine metric) -> HDBSCAN (EOM selection).
   Characterize clusters with texture profiles, feature importance (Kruskal-Wallis), temporal
   analysis (phase distributions, transition matrices, chi-squared tests)
5. **Visualize** — publication figures, Folium geospatial maps, interactive Gradio explorer

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Key Commands

```bash
# Run full pipeline stages sequentially
python scripts/ingest_data.py --config configs/data_config.yaml
python scripts/extract_features.py --config configs/feature_config.yaml
python scripts/run_clustering.py --config configs/clustering_config.yaml
python scripts/generate_figures.py --config configs/visualization_config.yaml

# Interactive explorer
python scripts/launch_explorer.py

# Tests
pytest tests/                        # all tests
pytest tests/test_metadata.py        # single test file
pytest tests/test_features.py -k "test_glcm"  # single test by name
```

## Source Layout

```
src/
├── data/           # ingest, metadata_extractor, tiling, ground_masking, preprocessing
├── features/       # dino_embeddings, texture_descriptors, feature_store (HDF5), fusion
├── clustering/     # dimensionality_reduction, cluster_discovery, cluster_characterization,
│                   # temporal_analysis, continuous_space
├── visualization/  # embedding_plots, cluster_galleries, geospatial_maps, temporal_plots,
│                   # publication_figures
├── inference/      # predict, batch_predict (assign new images to discovered clusters)
└── utils/          # exif, geo, io
```

## Data Layout

- Raw images in `data/kim_2023/` — drone (DJI, multiple altitudes) and ground (iPhone).
  **Never modify raw data.**
- Processed outputs go to `data/processed/` (tiles, masked ground patches)
- Master catalog: `data/metadata/image_catalog.csv` (single source of truth for downstream stages)
- Features cached in HDF5: `outputs/features/feature_store.h5`
  - Datasets: `/dino_cls`, `/texture`, `/image_ids`, `/dino_pca`, `/texture_pca`, `/fused`
- Cluster outputs in `outputs/clusters/` (assignments CSV, profiles JSON, quality metrics JSON)
- Figures in `outputs/figures/` (300 DPI PNG + SVG)

## Configuration

All config in YAML under `configs/`. Four config files correspond to the four pipeline scripts:
`data_config.yaml`, `feature_config.yaml`, `clustering_config.yaml`, `visualization_config.yaml`.

Key parameters with defaults:
- Tiling: 512px tiles, 25% overlap, reject tiles with >20% no-data
- DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m`, input 518x518, batch size 32
- GLCM: distances [1,3,5,10], 4 angles averaged over orientation
- Gabor: 4 frequencies x 6 orientations, mean + variance per filter
- UMAP: n_neighbors=30, min_dist=0.1, cosine metric, seed=42
- HDBSCAN: min_cluster_size=15, min_samples=5, cluster_selection_method=eom

## Conventions

- Type hints everywhere, dataclasses for config objects
- Docstrings on all public functions
- Log with Python `logging` module, never `print`
- All random operations seeded (seed=42) for reproducibility
- Colourblind-safe palettes only (Wong or Tol for discrete; viridis/cividis for continuous;
  never jet/rainbow/hot). Test with CVD simulation.
- Publication figures: 300 DPI PNG + SVG, Arial font, min 8pt, panels labeled (a)(b)(c)
- Statistical tests must report effect sizes and use multiple comparison correction

## Agent Files

See `.claude/agents/` for specialized agent instructions:
- `data-pipeline.md` — ingestion, tiling, masking, preprocessing
- `feature-extraction.md` — DINOv3 embeddings, texture descriptors, fusion
- `clustering-analysis.md` — UMAP, HDBSCAN, characterization, temporal analysis
- `visualization-export.md` — plots, maps, publication figures, Gradio app

## Implementation Reference

`SOFTWARE_PLAN.md` contains the full implementation specification with pseudocode for all
modules, detailed feature vector breakdowns, config file schemas, and the 10-phase
execution plan.

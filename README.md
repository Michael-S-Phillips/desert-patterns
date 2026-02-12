# Desert Patterns: Unsupervised Discovery of Patterned Ground Types in Atacama Desert Imagery

An unsupervised deep-learning pipeline for discovering and characterizing patterned ground types from heterogeneous desert imagery (drone and ground-level). Designed for the Yungay basin, Atacama Desert, this tool reveals natural pattern groupings from the data itself -- without human-imposed categories -- to support microbial ecology research.

There is no training loop and no labelled data. Feature extraction uses pretrained self-supervised models (inference only), and cluster discovery is fully unsupervised.

## Installation

Requires Python >= 3.10.

```bash
# Core pipeline (clustering, visualization, texture analysis)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Deep learning features (DINOv2 embeddings, SAM ground masking)
pip install -e ".[ml]"

# Interactive Gradio explorer
pip install -e ".[viz]"
```

## Pipeline Usage

The pipeline consists of six sequential stages. Each stage reads the output of the previous one and can be re-run independently.

### 1. Data Ingestion

Scans image directories, extracts EXIF/XMP metadata (GPS, altitude, timestamp, camera model), computes ground sampling distance, assigns temporal phases, and flags quality issues (blur, exposure).

```bash
python scripts/ingest_data.py --config configs/data_config.yaml
```

**Outputs:** `data/metadata/image_catalog.csv` (master catalog linking every image to its metadata).

### 2. Preprocessing

Tiles large drone images into overlapping 512 px patches, masks ground regions in ground-level photos (SAM with horizon-detection and lower-crop fallbacks), and standardizes all images to 518x518 px with per-image channel-wise normalization.

```bash
python scripts/preprocess_data.py --config configs/data_config.yaml
```

**Outputs:** `data/processed/` (tiles, masked ground patches, `preprocessing_manifest.csv`).

### 3. Feature Extraction

Extracts dual feature representations for every preprocessed image: DINOv2 embeddings (768-d) and classical texture descriptors (90-d). Fuses via PCA + weighted concatenation.

```bash
python scripts/extract_features.py --config configs/feature_config.yaml
```

**Outputs:** `outputs/features/feature_store.h5` (HDF5 with datasets: `/dino_cls`, `/texture`, `/image_ids`, `/dino_pca`, `/texture_pca`, `/fused`).

### 4. Clustering

Runs UMAP dimensionality reduction followed by HDBSCAN clustering across multiple feature ablations (DINOv2-only, texture-only, fused 70/30, fused 50/50). Computes quality metrics, bootstrap stability, cluster characterization, temporal analysis, and continuous pattern space.

```bash
python scripts/run_clustering.py --config configs/clustering_config.yaml

# Optional flags
python scripts/run_clustering.py --config configs/clustering_config.yaml \
    --ablation fused_default \       # run a single ablation
    --save-models \                  # persist fitted models for inference
    --scene-clusters "6,7" \         # feature curation: remove scene-content clusters
    --n-directions 2 \               # number of scene directions to project out
    --skip-temporal \                # skip temporal analysis
    --skip-continuous \              # skip continuous space analysis
    --skip-bootstrap \               # skip bootstrap stability
    --force --verbose
```

**Outputs per ablation:** `outputs/clusters/{name}_assignments.csv`, `_profiles.json`, `_quality_metrics.json`, `_feature_importance.csv`, `_temporal.json`, `_continuous_space.csv`.
**Saved models (with `--save-models`):** `outputs/models/{name}/` containing `umap_high_dim.joblib`, `umap_2d.joblib`, `umap_3d.joblib`, `hdbscan.joblib`, `pipeline_config.json`, and optionally `curation.joblib`.

### 5. Visualization

Generates publication-quality static figures and an interactive Gradio explorer.

```bash
# Static figures (300 DPI PNG + SVG)
python scripts/generate_figures.py --config configs/visualization_config.yaml

# Interactive explorer (Gradio, requires pip install -e ".[viz]")
python scripts/launch_explorer.py
```

**Outputs:** `outputs/figures/` (embedding plots, cluster galleries, temporal plots, silhouette diagrams).
**Gradio tabs:** Pattern Space, Cluster Browser, Temporal, Upload & Classify, Correlation Workbench.

### 6. Inference

Classifies new images into the discovered pattern clusters using persisted models.

```bash
# Single image
python scripts/run_inference.py --model-dir outputs/models/fused_default --image path/to/image.jpg

# Batch (directory)
python scripts/run_inference.py --model-dir outputs/models/fused_default --image-dir path/to/images/ --output results.csv
```

### Running Tests

```bash
pytest tests/                              # all tests
pytest tests/test_metadata.py              # single module
pytest tests/test_features.py -k "glcm"   # single test by keyword
```

## Configuration

All parameters are specified in YAML files under `configs/`. Each config file corresponds to a pipeline stage:

| Config file | Stage | Key parameters |
|---|---|---|
| `data_config.yaml` | Ingestion + preprocessing | Altitude group boundaries, tile size (512 px), overlap (25%), blur threshold, SAM checkpoint path |
| `feature_config.yaml` | Feature extraction | DINOv2 model name, GLCM distances, Gabor filter bank, LBP radius, PCA variance threshold (0.95), fusion weights (0.7/0.3) |
| `clustering_config.yaml` | Clustering | UMAP (n_neighbors=30, min_dist=0.1, cosine metric), HDBSCAN (min_cluster_size=15, min_samples=5, EOM selection), bootstrap iterations (100), ablation list, curation parameters |
| `visualization_config.yaml` | Visualization | DPI (300), font (Arial), export formats (PNG+SVG), figure width (7"), gallery grid size, Gradio server port |

---

## Methods

### Overview

The pipeline processes heterogeneous desert imagery (aerial drone at multiple altitudes and ground-level photographs) through six stages: ingestion, preprocessing, dual feature extraction, dimensionality reduction and clustering, visualization, and inference. The design is intentionally unsupervised: no human labels are used at any stage. Pattern categories emerge entirely from the structure of the feature space.

### Preprocessing

**Drone image tiling.** Large drone images (4000x3000 px) are divided into overlapping 512x512 px tiles using a regular grid with 25% overlap (stride = 384 px). Edge tiles are clamped to image boundaries. Each tile inherits the metadata of its parent image.

**Ground-level masking.** Ground-level photographs contain non-surface content (sky, horizon, equipment). A three-tier fallback chain isolates the ground surface: (1) the Segment Anything Model (SAM; Kirillov et al. 2023), ViT-B checkpoint, prompted with a point grid in the lower image region; (2) horizon detection via Canny edge detection and Hough line transform on the upper half; (3) lower 60% crop. The first method that succeeds is used.

**Standardization.** All tiles and masked ground patches are resized to 518x518 px (bicubic interpolation: INTER_AREA for downsampling, INTER_LANCZOS4 for upsampling) to match the DINOv2 input resolution. Per-image channel-wise normalization maps each channel to zero mean and unit variance, clipping at +/-3 standard deviations and rescaling to [0, 255]. Uniform-intensity images are mapped to a constant value of 128.

### Feature Extraction

Two complementary feature representations are extracted from every preprocessed image. The self-supervised embedding captures high-level semantic structure invariant to viewpoint, scale, and lighting; the classical texture descriptors provide physically interpretable features for scientific explanation.

**DINOv2 embeddings.** We use DINOv2 ViT-B/14 (Oquab et al. 2024), a Vision Transformer pretrained with self-supervision on 142 M images (LVD-142M dataset), loaded via HuggingFace Transformers. We extract the [CLS] token embedding, yielding a 768-dimensional feature vector per image. The [CLS] token aggregates global image information across all spatial positions via the self-attention mechanism and serves as a holistic image descriptor. Patch tokens (excluding the CLS and 4 register tokens) are optionally available for spatial analysis. Inference runs on MPS (Apple Silicon) or CPU; no GPU training is performed.

**Classical texture descriptors (90 features).** Each image is converted to 8-bit grayscale and processed through six feature families:

1. *Gray-Level Co-occurrence Matrix (GLCM; Haralick et al. 1973).* Computed at four pixel distances (1, 3, 5, 10) and four angles (0, 45, 90, 135 degrees), yielding a co-occurrence matrix at each (distance, angle) pair. Five properties are extracted: contrast, correlation, energy (angular second moment), homogeneity, and entropy (computed manually as negative sum of p log2 p over non-zero entries). Properties are averaged across the four angles for each distance, producing 5 properties x 4 distances = 20 features. The GLCM is computed with symmetric normalization at 256 grey levels.

2. *Gabor filter bank (Fogel & Sagi 1989).* A bank of 24 Gabor filters spans 4 spatial frequencies (0.05, 0.1, 0.2, 0.4 cycles/pixel) and 6 orientations (0 to 150 degrees in 30-degree steps). For each filter, the mean and variance of the response magnitude (sqrt(real^2 + imaginary^2)) are recorded, yielding 4 x 6 x 2 = 48 features. The image is normalized to [0, 1] before filtering.

3. *Fractal dimension (Mandelbrot 1982).* The box-counting dimension is computed on a Canny edge map (thresholds 50/150) of the Otsu-binarized image. The image is padded to the next power-of-two square, and box sizes increase as powers of 2. The fractal dimension is estimated by linear regression of log(occupied box count) vs. log(1/box size). Values typically range from 1.0 (simple boundaries) to 2.0 (space-filling). This yields 1 feature.

4. *Lacunarity (Plotnick et al. 1996).* Computed via a sliding-box algorithm at five box sizes (5, 10, 20, 40, 80 pixels) on the Otsu-binarized image. For each box size, lacunarity is defined as Lambda = var(mass) / mean(mass)^2 + 1, where mass is the count of foreground pixels in each box position. Box sums are computed efficiently using an integral image. This yields 5 features.

5. *Local Binary Patterns (LBP; Ojala et al. 2002).* Uniform LBP with P=8 sample points at radius R=1 produces P+2 = 10 histogram bins. The histogram is density-normalized. This yields 10 features.

6. *Global statistics.* Mean, standard deviation, skewness, and kurtosis of pixel intensities; Canny edge density (fraction of edge pixels); and dominant spatial frequency from the 2D FFT magnitude spectrum (location of peak energy, normalized by image dimensions). This yields 6 features.

### Feature Fusion

The 768-d DINOv2 embeddings and the 90-d texture descriptors are fused into a single feature vector per image via the following procedure:

1. **L2 normalization** of each feature set independently (row-wise).
2. **PCA** on each normalized set, retaining components explaining >= 95% of cumulative variance (fitted with seed = 42, full SVD solver). This reduces the DINOv2 features to typically 100-200 dimensions and the texture features to approximately 30-50 dimensions.
3. **Weighted concatenation:** fused = [w_dino * dino_pca | w_texture * texture_pca], with default weights w_dino = 0.7, w_texture = 0.3. The heavier DINOv2 weight reflects its stronger clustering performance; the texture weight adds interpretability.
4. **Final L2 normalization** of the concatenated vector.

The pipeline also supports single-modality analysis (DINOv2-only or texture-only) for ablation comparison.

### Feature Curation (Optional)

When initial (triage) clustering reveals clusters dominated by scene content rather than ground surface patterns (e.g., tape measures, sky), an optional orthogonal-projection curation step can remove scene-content directions from the DINOv2 embedding space before the final clustering.

1. Compute the centroid of scene-contaminated clusters and the centroid of pattern clusters from the triage stage.
2. Form difference vectors d_i = centroid(scene_i) - centroid(pattern) for each scene cluster.
3. Apply PCA to the difference matrix to extract the top K orthonormal scene directions V (default K = 3, clamped to the number of scene clusters; directions with eigenvalues below 1e-10 are discarded).
4. Project out: X_curated = X - (X V^T) V.

This zeroes the component of each feature vector along the identified scene directions while preserving orthogonal (pattern-relevant) components. The curated DINOv2 features replace the raw embeddings for the DINOv2-only and fused ablations; texture features are unaffected. Curation directions are persisted alongside the UMAP/HDBSCAN models so that new images at inference time receive the same projection.

### Dimensionality Reduction

We use UMAP (McInnes et al. 2018) for dimensionality reduction because it preserves both local neighbourhood structure and global topology better than t-SNE (van der Maaten & Hinton 2008), and supports projection of new data via a learned transform. Three UMAP reductions are computed:

1. **High-dimensional** (default 15 components): input to HDBSCAN. Using UMAP as a preprocessing step before density-based clustering mitigates the curse of dimensionality while preserving more manifold structure than PCA alone.
2. **2D** (2 components): for visualization in scatter plots.
3. **3D** (3 components): for interactive 3D exploration.

All three use n_neighbors = 30, min_dist = 0.1, and cosine metric on the L2-normalized feature vectors (random seed = 42). t-SNE at perplexities 15, 30, and 50 is available as an alternative visualization for verifying that cluster structure is not an artefact of UMAP geometry.

### Clustering

HDBSCAN (Campello et al. 2013; McInnes et al. 2017) operates on the 15-dimensional UMAP embeddings. We use HDBSCAN rather than centroid-based methods (e.g. k-means) because it:

- Does not require pre-specifying the number of clusters.
- Explicitly identifies noise points (label = -1) rather than forcing every sample into a cluster.
- Handles clusters of varying density through hierarchical density estimates.
- Provides soft cluster membership probabilities via the condensed tree.
- Supports new-point prediction via `approximate_predict` without refitting.

Parameters: min_cluster_size = 15, min_samples = 5, cluster_selection_method = "eom" (Excess of Mass; Campello et al. 2013). The EOM method extracts the most persistent clusters from the condensed tree, favouring cluster selections that maximize total excess of mass over the stability threshold -- this is preferable to the "leaf" method for datasets with clusters of varying density.

**Multi-ablation comparison.** Clustering is run on four feature configurations: DINOv2-only, texture-only, fused (0.7/0.3 default weighting), and fused (0.5/0.5 equal weighting). Cluster quality is compared across all four to select the best configuration.

### Cluster Validation

Four complementary validation metrics are computed, excluding noise points:

1. **Silhouette score** (Rousseeuw 1987): mean ratio of intra-cluster cohesion to nearest-cluster separation; range [-1, 1], higher is better. Per-cluster silhouette scores are also reported.
2. **Davies-Bouldin index** (Davies & Bouldin 1979): mean ratio of within-cluster scatter to between-cluster separation; lower is better.
3. **Calinski-Harabasz index** (Calinski & Harabasz 1974): ratio of between-cluster variance to within-cluster variance, weighted by degrees of freedom; higher is better.
4. **DBCV** (Moulavi et al. 2014): density-based cluster validation computed via `hdbscan.validity_index`; specifically designed for density-based clustering and accounts for arbitrary cluster shapes; range [-1, 1], higher is better.

**Bootstrap stability analysis.** 100 bootstrap resamples (with replacement) are drawn, reclustered with the same parameters, and compared to the full-data labels. Cluster robustness is assessed by the mean Adjusted Rand Index (ARI; Hubert & Arabie 1985) across bootstrap iterations, and per-cluster recovery rate (fraction of bootstraps in which >50% of a cluster's original members are assigned to the same bootstrap cluster).

### Cluster Characterization

Each discovered cluster is profiled along three axes:

1. **Texture feature profile:** Mean and standard deviation of all 90 texture features within the cluster. Z-scores relative to the global (non-noise) population identify the most distinguishing features.

2. **Feature importance:** The Kruskal-Wallis H-test (Kruskal & Wallis 1952) is computed per feature across all clusters (excluding noise). P-values are adjusted via Bonferroni correction. Effect size is reported as eta-squared: eta^2 = (H - k + 1) / (N - k), where k is the number of clusters and N the total non-noise sample count. Features are ranked by H-statistic.

3. **Metadata distribution:** For each cluster, the proportions of source type (drone/ground), altitude group (high/mid/low/ground), and temporal phase are computed.

Representative images (nearest to the cluster centroid in fused feature space) and boundary images (farthest from centroid) are identified for visual inspection.

### Temporal Analysis

Temporal dynamics are analysed across rain event phases (pre-rain, during-rain, post-rain). All temporal methods require a minimum of 5 samples per phase and return null results with logged warnings when data is insufficient.

1. **Phase distributions:** Cross-tabulation of cluster membership by temporal phase, normalized per cluster.

2. **Transition matrix:** For co-located images (matched by GPS proximity within 50 m across different temporal phases), cluster-to-cluster transition probabilities are computed as a row-normalized matrix.

3. **Embedding drift:** The centroid of each cluster is computed separately for each temporal phase in 2D UMAP space. The Euclidean distance between phase centroids measures how much a pattern type shifts in embedding space across the rain event.

4. **Statistical tests:** A chi-squared test on the cluster-by-phase contingency table tests whether cluster distributions differ significantly between phases. Cramer's V is reported as the effect size. Per-cluster pairwise proportion z-tests (with Bonferroni correction) identify which specific clusters change significantly between phase pairs.

### Continuous Pattern Space

As an alternative to discrete cluster assignments, the 2D UMAP embedding is treated as a continuous pattern space:

- **Kernel density estimation** (Gaussian KDE with Scott's bandwidth rule) maps the point distribution to a continuous density surface, enabling identification of pattern "hotspots" and visual comparison of density between temporal phases.
- **Measurement correlation:** External field measurements (e.g. microbial activity) are matched to the nearest image by GPS (Haversine distance). Spearman rank correlations are computed between each UMAP dimension and the measurement values independently. A random forest regressor (100 trees, seed = 42) trained on the 2D UMAP coordinates provides a combined R^2 for nonlinear association.

### Inference

Fitted UMAP and HDBSCAN models are serialized via joblib. New images are processed through the same preprocessing and feature extraction pipeline, then:

1. Features are projected into the high-dimensional UMAP space via `UMAP.transform()`.
2. Cluster assignment and membership probability are obtained via `hdbscan.approximate_predict()`.
3. Features are projected into 2D and 3D UMAP space for visualization.

If feature curation was applied during training, the saved orthogonal projection directions are automatically applied to new DINOv2 features before UMAP projection.

### Reproducibility

All stochastic operations are seeded (seed = 42): PCA, UMAP, HDBSCAN bootstrap resampling, and random forest regression. The pipeline is deterministic given a fixed configuration. All configuration is specified in YAML and tracked alongside code.

### Visualization Standards

All figures follow publication standards: 300 DPI raster (PNG) with vector (SVG) counterparts, Arial font (DejaVu Sans fallback) at minimum 8 pt, 7.0" width for double-column figures. Discrete colour palettes use the Wong (2011) 8-colour colourblind-safe palette; noise points are rendered in grey (#999999). Continuous colormaps use viridis or cividis (never jet, rainbow, or hot). The Matplotlib Agg backend is set automatically in headless environments.

---

## References

- Calinski, T. & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1), 1-27.
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, 160-172.
- Davies, D. L. & Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, 1(2), 224-227.
- Fogel, I. & Sagi, D. (1989). Gabor filters as texture discriminator. *Biological Cybernetics*, 61(2), 103-113.
- Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Trans. Systems, Man, and Cybernetics*, 3(6), 610-621.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.
- Kirillov, A. et al. (2023). Segment Anything. *ICCV 2023*.
- Kruskal, W. H. & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *JASA*, 47(260), 583-621.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for dimension reduction. *arXiv:1802.03426*.
- McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *JOSS*, 2(11), 205.
- Moulavi, D. et al. (2014). Density-based clustering validation. *SDM 2014*, 839-847.
- Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI*, 24(7), 971-987.
- Oquab, M. et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*.
- Plotnick, R. E., Gardner, R. H., & O'Neill, R. V. (1996). Lacunarity indices as measures of landscape texture. *Landscape Ecology*, 8(3), 201-211.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53-65.
- van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579-2605.
- Wong, B. (2011). Color blindness. *Nature Methods*, 8(6), 441.

## License

This project is for academic research purposes.

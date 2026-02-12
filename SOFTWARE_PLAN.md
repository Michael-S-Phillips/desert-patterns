# Software Plan: Unsupervised Desert Patterned Ground Discovery & Classification

## 1. Project Overview

**Goal:** Build an unsupervised deep learning pipeline that discovers and characterizes different types of patterned ground in desert terrain (Yungay basin, Atacama Desert) from heterogeneous image sources — without requiring human-labeled categories. The system should reveal natural pattern groupings from the data itself, then enable scientific correlation with microbial activity measurements.

**Scientific context:** The Yungay basin was imaged in 2017 before, during, and after a record rain event. A desert microbial ecologist seeks to determine whether different types of patterned ground can be associated with microbial activity. Rather than imposing a human taxonomy, we let the data reveal its own structure.

**Key design principles:**
- **Unsupervised first:** No human labels required for pattern discovery. The ecologist interprets clusters after they emerge.
- **Multi-scale invariant:** Must handle drone imagery at multiple altitudes and ground-level iPhone photos of the same terrain.
- **Scientifically interpretable:** Every discovered pattern group must be characterizable with physically meaningful texture descriptors.
- **Temporally aware:** Track how pattern distributions shift across pre-rain → during-rain → post-rain phases.

---

## 2. Project Structure

```
desert-patterns/
├── CLAUDE.md
├── .claude/
│   └── agents/
│       ├── data-pipeline.md
│       ├── feature-extraction.md
│       ├── clustering-analysis.md
│       └── visualization-export.md
├── pyproject.toml
├── README.md
├── configs/
│   ├── data_config.yaml
│   ├── feature_config.yaml
│   ├── clustering_config.yaml
│   └── visualization_config.yaml
├── data/
│   ├── raw/
│   │   ├── drone/
│   │   │   ├── high_altitude/
│   │   │   ├── mid_altitude/
│   │   │   └── low_altitude/
│   │   └── ground/
│   ├── processed/
│   │   ├── tiles/
│   │   └── ground_masked/
│   ├── metadata/
│   │   └── image_catalog.csv
│   └── splits/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── metadata_extractor.py
│   │   ├── tiling.py
│   │   ├── ground_masking.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── dino_embeddings.py
│   │   ├── texture_descriptors.py
│   │   ├── feature_store.py
│   │   └── fusion.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── dimensionality_reduction.py
│   │   ├── cluster_discovery.py
│   │   ├── cluster_characterization.py
│   │   ├── temporal_analysis.py
│   │   └── continuous_space.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── embedding_plots.py
│   │   ├── cluster_galleries.py
│   │   ├── geospatial_maps.py
│   │   ├── temporal_plots.py
│   │   └── publication_figures.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── batch_predict.py
│   └── utils/
│       ├── __init__.py
│       ├── exif.py
│       ├── geo.py
│       └── io.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_exploration.ipynb
│   ├── 03_clustering_analysis.ipynb
│   ├── 04_temporal_dynamics.ipynb
│   └── 05_microbial_correlation.ipynb
├── scripts/
│   ├── ingest_data.py
│   ├── extract_features.py
│   ├── run_clustering.py
│   ├── generate_figures.py
│   ├── predict.py
│   └── launch_explorer.py
├── outputs/
│   ├── features/
│   ├── clusters/
│   ├── figures/
│   └── reports/
└── tests/
    ├── test_metadata.py
    ├── test_tiling.py
    ├── test_features.py
    ├── test_clustering.py
    └── test_visualization.py
```

---

## 3. Phase-by-Phase Implementation Plan

### Phase 1: Data Ingestion & Exploration

#### 1a. Metadata Extraction (`src/data/metadata_extractor.py`)

Extract comprehensive metadata from every image:

```python
# Output schema: image_catalog.csv
{
    "image_id": str,             # unique hash-based ID
    "image_path": str,           # path to raw image
    "source_type": str,          # "drone" | "ground"
    "altitude_m": float | None,  # from EXIF/flight log, None for ground
    "altitude_group": str,       # "high" | "mid" | "low" | "ground"
    "lat": float | None,
    "lon": float | None,
    "timestamp": datetime,
    "temporal_phase": str,       # "pre_rain" | "during_rain" | "post_rain"
    "camera_model": str,
    "focal_length_mm": float,
    "image_width_px": int,
    "image_height_px": int,
    "gsd_cm_per_px": float | None,  # ground sampling distance (drone only)
    "quality_flag": str,         # "good" | "blur" | "overexposed" | "partial"
}
```

Implementation details:
- Use `exifread` and `Pillow` for EXIF extraction
- Parse DJI drone EXIF for altitude (RelativeAltitude tag), GPS, gimbal angle
- Estimate GSD: `gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)`
- Temporal phase assignment: based on timestamp ranges (configurable in `data_config.yaml` — the user provides the date boundaries for the rain event)
- Quality flagging: compute Laplacian variance for blur detection, histogram analysis for over/underexposure

#### 1b. Image Quality Filtering

- Compute blur score (Laplacian variance) — flag images below threshold
- Detect overexposed/underexposed frames via histogram clipping analysis
- Flag images with >50% sky/non-ground content (ground-level photos)
- Output: updated `image_catalog.csv` with quality columns; filtering is configurable but not destructive (raw images never deleted)

#### 1c. EDA Notebook (`notebooks/01_eda.ipynb`)

Auto-generated notebook with:
- Image count distributions by source_type, altitude_group, temporal_phase
- Geospatial scatter plot of image locations (folium map)
- Visual grid of representative images per group (random sample, 4x4 per group)
- Resolution/size distributions
- Quality score distributions
- Timestamp timeline showing coverage gaps

---

### Phase 2: Preprocessing Pipeline

#### 2a. Tiling for Drone Images (`src/data/tiling.py`)

Large drone images often contain multiple pattern types across the frame. Tile them into manageable patches.

```python
class TilingConfig:
    tile_size: int = 512         # pixels
    overlap_fraction: float = 0.25
    min_valid_fraction: float = 0.8  # reject tiles with >20% black/no-data
    output_format: str = "png"
```

- Tile each drone image into overlapping patches
- Each tile inherits parent image metadata + tile coordinates (row, col, pixel offset)
- Reject tiles that are mostly featureless (e.g., edge tiles with black borders)
- Compute per-tile statistics: mean brightness, contrast, edge density — useful later for filtering out uninformative tiles

#### 2b. Ground Region Masking (`src/data/ground_masking.py`)

Ground-level iPhone photos contain sky, horizon, equipment, people, etc. We need to isolate the ground surface.

Strategy (in order of preference):
1. **SAM (Segment Anything Model)** — prompt with points in the lower portion of the image to segment ground surface. Use the `sam2` package with the ViT-B checkpoint.
2. **Fallback: horizon detection** — detect the horizon line via edge detection + Hough transform on the upper half, then mask everything above it.
3. **Simple fallback: lower-half crop** — if both above fail, take the lower 60% of the image (most ground-level desert photos point slightly downward).

Output: for each ground image, save both the mask and the masked/cropped ground region.

#### 2c. Standardization (`src/data/preprocessing.py`)

- Resize all tiles and ground patches to a uniform size for feature extraction (e.g., 518×518 for DINOv2, which uses 14×14 patches on 518px input)
- Color normalization: per-image channel-wise standardization to handle lighting variation
- Optional white-balance correction using gray-world assumption
- Save processed images to `data/processed/` with a manifest CSV linking processed → raw paths

---

### Phase 3: Dual Feature Extraction

This is the core of the pipeline. For each processed image, extract two complementary feature vectors.

#### 3a. DINOv2 Embeddings (`src/features/dino_embeddings.py`)

DINOv2 is a self-supervised Vision Transformer pretrained on 142M images. It produces semantically rich features that are invariant to viewpoint, scale, and lighting — exactly the invariances needed for multi-scale desert imagery.

```python
class DinoFeatureExtractor:
    """
    Extract DINOv2 embeddings for pattern images.
    
    Uses the [CLS] token embedding as a global image descriptor.
    Optionally also extracts patch-level tokens for spatial analysis.
    """
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        # Load from torch.hub: facebookresearch/dinov2
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(device)
        self.transform = # DINOv2 standard transform: resize 518, normalize
    
    def extract_cls(self, image: PIL.Image) -> np.ndarray:
        """Global image embedding — 768-d vector for ViT-B/14."""
        ...
    
    def extract_patch_tokens(self, image: PIL.Image) -> np.ndarray:
        """Spatial feature map — (n_patches, 768) for region-level analysis."""
        ...
    
    def extract_batch(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Batch extraction with GPU acceleration. Returns (N, 768) array."""
        ...
```

Key details:
- Use `dinov2_vitb14` (ViT-Base with 14×14 patches) — good balance of quality and speed
- If GPU memory allows, optionally try `dinov2_vitl14` (ViT-Large, 1024-d) for richer features
- Extract [CLS] token as the primary global descriptor (768-d for ViT-B)
- Also save patch tokens for potential later use in spatial pattern analysis within images
- Process in batches with DataLoader for efficiency
- Save embeddings to `outputs/features/dino_embeddings.npy` with corresponding image IDs

#### 3b. Classical Texture Descriptors (`src/features/texture_descriptors.py`)

These provide physically interpretable features that can explain *why* clusters differ.

```python
class TextureDescriptorExtractor:
    """
    Extract interpretable texture features from grayscale images.
    
    Feature groups:
    1. GLCM (Gray-Level Co-occurrence Matrix) — spatial relationships between pixel intensities
    2. Gabor filters — multi-scale, multi-orientation frequency analysis
    3. Fractal dimension — scale complexity
    4. Lacunarity — gap/heterogeneity structure
    5. LBP (Local Binary Patterns) — local texture microstructure
    6. Global statistics — basic intensity/edge stats
    """
```

Feature vector breakdown (~80-100 features total):

**GLCM features (computed at multiple distances: 1, 3, 5, 10 pixels):**
- Contrast — intensity variation between neighboring pixels
- Correlation — linear dependency of gray levels
- Energy (Angular Second Moment) — uniformity
- Homogeneity — closeness to diagonal
- Entropy — randomness/disorder
- = 5 properties × 4 distances × 4 angles (averaged) = ~20 features

**Gabor filter bank responses:**
- Frequencies: 4 scales (0.05, 0.1, 0.2, 0.4 cycles/pixel)
- Orientations: 6 angles (0°, 30°, 60°, 90°, 120°, 150°)
- For each filter: mean and variance of response magnitude
- = 4 × 6 × 2 = 48 features

**Fractal dimension (box-counting method):**
- Computed on binarized edge map (Canny)
- Single scalar value ~1.0-2.0 (higher = more complex boundary structure)
- = 1 feature

**Lacunarity:**
- Sliding box algorithm at multiple scales
- Captures "gappiness" — how pattern density varies across scales
- Compute at 5 box sizes
- = 5 features

**LBP histogram:**
- Local Binary Pattern with radius=1,2,3 and P=8,16,24 points
- Histogram of LBP codes (use uniform patterns to reduce dimensionality)
- = ~10 features (uniform pattern bins)

**Global statistics:**
- Mean, std, skewness, kurtosis of intensity
- Edge density (Canny edge pixel fraction)
- Dominant frequency from 2D FFT
- = ~6 features

Libraries:
- `scikit-image` for GLCM (`graycomatrix`, `graycoprops`), LBP, Gabor
- Custom implementation for fractal dimension and lacunarity
- `scipy` for FFT analysis

#### 3c. Feature Storage & Fusion (`src/features/feature_store.py`, `src/features/fusion.py`)

```python
class FeatureStore:
    """
    Manages computed features with lazy loading and caching.
    
    Storage format: HDF5 file with datasets:
    - /dino_cls: (N, 768) float32
    - /texture: (N, ~90) float32
    - /image_ids: (N,) string
    - /metadata: group with per-image attributes
    """
    
    def get_fused_features(self, 
                           dino_weight: float = 0.7, 
                           texture_weight: float = 0.3,
                           normalize: bool = True) -> np.ndarray:
        """
        Fuse DINOv2 and texture features with configurable weighting.
        
        1. L2-normalize each feature set independently
        2. Apply PCA to each (retain 95% variance)
        3. Concatenate with weighting
        4. Final L2-normalization
        """
        ...
```

The fusion strategy matters. DINOv2 embeddings capture high-level semantic structure; texture descriptors capture fine-grained physical properties. They're complementary.

Fusion approach:
1. L2-normalize each feature set (DINOv2 768-d, texture ~90-d)
2. PCA on each independently — retain components explaining 95% of variance (DINOv2 typically reduces to ~100-200d, texture to ~30-50d)
3. Weighted concatenation: `fused = [dino_weight * dino_pca, texture_weight * texture_pca]`
4. Default weights: 0.7 DINOv2, 0.3 texture (configurable — DINOv2 is the primary representation, texture adds interpretability)
5. Final L2-normalization of fused vector

Also support using each feature set independently for ablation analysis.

---

### Phase 4: Clustering & Pattern Discovery

#### 4a. Dimensionality Reduction (`src/clustering/dimensionality_reduction.py`)

UMAP is preferred over t-SNE for this application because it better preserves global structure (important for continuous pattern space analysis) and is faster on medium-sized datasets.

```python
class DimensionalityReducer:
    def __init__(self, method: str = "umap", **kwargs):
        self.method = method
        # UMAP defaults tuned for pattern discovery:
        self.umap_params = {
            "n_neighbors": 30,       # balance local vs global structure
            "min_dist": 0.1,         # allow tight clusters
            "n_components": 2,       # for visualization
            "metric": "cosine",      # works well with normalized embeddings
            "random_state": 42,
        }
        self.umap_params.update(kwargs)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Reduce to 2D or 3D for visualization and clustering."""
        ...
    
    def fit_transform_high_dim(self, features: np.ndarray, n_components: int = 15) -> np.ndarray:
        """Reduce to intermediate dimensionality for clustering (higher than 2D)."""
        ...
```

Two reductions:
1. **High-dimensional reduction** (features → 10-20 dims) — input to HDBSCAN. Using UMAP for this preserves more structure than PCA alone.
2. **2D/3D reduction** (features → 2-3 dims) — for visualization. Always compute both 2D (plots) and 3D (interactive exploration).

Also compute t-SNE 2D as an alternative visualization (different perplexity values: 15, 30, 50) to verify that cluster structure is method-independent.

#### 4b. Cluster Discovery (`src/clustering/cluster_discovery.py`)

```python
class PatternClusterDiscovery:
    """
    Discover natural pattern groupings using HDBSCAN.
    
    HDBSCAN advantages over k-means:
    - No need to pre-specify number of clusters
    - Identifies noise/outlier images
    - Handles clusters of varying density
    - Provides soft cluster membership probabilities
    """
    
    def __init__(self, min_cluster_size: int = 15, min_samples: int = 5):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",  # Excess of Mass — better for varying densities
            prediction_data=True,            # enable soft clustering & new point prediction
        )
    
    def fit(self, features_reduced: np.ndarray) -> ClusterResult:
        """
        Returns:
        - cluster_labels: int array, -1 = noise
        - probabilities: float array, membership confidence
        - outlier_scores: float array, GLOSH outlier score per point
        - condensed_tree: for visualization and analysis
        """
        ...
    
    def evaluate_stability(self, features: np.ndarray, n_bootstrap: int = 100) -> dict:
        """
        Bootstrap stability analysis:
        - Resample data n times, recluster, measure adjusted Rand index
        - Report which clusters are consistently recovered
        """
        ...
```

Also compute and report:
- **Silhouette score** (overall and per-cluster)
- **Davies-Bouldin index**
- **Calinski-Harabasz index**
- **DBCV** (density-based cluster validation — native to HDBSCAN)
- Cluster sizes and noise fraction

Run clustering on multiple feature configurations:
1. DINOv2 only
2. Texture only
3. Fused (default 0.7/0.3)
4. Fused with different weight ratios

Compare cluster quality metrics across configurations to select the best.

#### 4c. Cluster Characterization (`src/clustering/cluster_characterization.py`)

This is where scientific interpretability comes in. For each discovered cluster, compute a statistical profile.

```python
class ClusterCharacterizer:
    """
    For each cluster, compute:
    1. Texture feature profile (mean ± std of each descriptor)
    2. Distinguishing features (features with highest between-cluster / within-cluster variance)
    3. Metadata profile (distribution of source_type, altitude_group, temporal_phase)
    4. Representative images (closest to cluster centroid)
    5. Boundary images (furthest from centroid, most ambiguous membership)
    6. Natural language description (auto-generated from distinguishing features)
    """
    
    def characterize(self, cluster_id: int) -> ClusterProfile:
        ...
    
    def generate_description(self, profile: ClusterProfile) -> str:
        """
        Auto-generate interpretable description from texture features.
        
        Example output:
        "Cluster 3 (n=142): Characterized by high fractal dimension (1.72 ± 0.08)
        and high GLCM entropy, consistent with fine-scale complex patterning. 
        Predominantly found in mid-altitude drone imagery (68%) during post-rain 
        phase (71%). Low lacunarity suggests relatively uniform pattern spacing."
        """
        ...
    
    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Rank texture features by discriminative power between clusters.
        Uses Kruskal-Wallis H-test per feature across clusters.
        Returns features sorted by H-statistic.
        """
        ...
```

#### 4d. Temporal Analysis (`src/clustering/temporal_analysis.py`)

Track how pattern distributions change across the rain event.

```python
class TemporalPatternAnalysis:
    """
    Analyze pattern dynamics across temporal phases.
    """
    
    def cluster_distribution_by_phase(self) -> pd.DataFrame:
        """Fraction of images in each cluster, per temporal phase."""
        ...
    
    def transition_matrix(self) -> np.ndarray:
        """
        For co-located images (same GPS region, different times):
        Compute transition probabilities between clusters from 
        pre→during, during→post, and pre→post.
        """
        ...
    
    def embedding_drift(self) -> dict:
        """
        Measure how the centroid of each cluster moves in embedding space
        across temporal phases. Large drift = pattern changed character.
        """
        ...
    
    def phase_statistical_tests(self) -> pd.DataFrame:
        """
        Chi-squared test: is cluster distribution significantly different
        between temporal phases?
        Per-cluster: is the proportion of each cluster significantly 
        different pre vs. post rain?
        """
        ...
```

#### 4e. Continuous Pattern Space (`src/clustering/continuous_space.py`)

As an alternative to discrete clusters, model patterns as positions in a continuous space.

```python
class ContinuousPatternSpace:
    """
    Instead of forcing discrete categories, represent each image
    as a point in a continuous 2D/3D pattern space.
    
    This enables:
    - Regression against microbial measurements (continuous → continuous)
    - Density estimation to find pattern "hotspots"
    - Gradient analysis: what direction in pattern space corresponds 
      to increasing microbial activity?
    """
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """Fit UMAP to produce the continuous pattern space coordinates."""
        ...
    
    def kernel_density(self, phase: str = None) -> KDEResult:
        """
        2D kernel density estimate in pattern space.
        If phase is specified, compute density for that phase only.
        Useful for comparing pattern distributions pre/post rain.
        """
        ...
    
    def correlate_with_measurements(self, 
                                     measurement_df: pd.DataFrame,
                                     measurement_col: str) -> CorrelationResult:
        """
        Given a DataFrame with columns [lat, lon, measurement_value],
        find the nearest image(s) to each measurement location,
        then compute correlation between pattern space coordinates 
        and measurement values.
        
        Methods:
        - Spearman rank correlation with UMAP dim 1, dim 2 independently
        - Random forest regression: UMAP coordinates → measurement value
        - Gaussian process regression for spatial interpolation
        """
        ...
```

---

### Phase 5: Visualization & Publication Outputs

#### 5a. Embedding Plots (`src/visualization/embedding_plots.py`)

- UMAP 2D scatter colored by cluster assignment (with noise points in gray)
- UMAP 2D scatter colored by temporal phase
- UMAP 2D scatter colored by altitude group
- UMAP 2D scatter colored by texture feature values (continuous colormap per feature)
- 3D interactive scatter (plotly) for exploration
- Side-by-side: DINOv2-only vs. texture-only vs. fused clustering results
- Cluster condensed tree dendrogram (from HDBSCAN)

#### 5b. Cluster Galleries (`src/visualization/cluster_galleries.py`)

For each cluster:
- Grid of 16-25 representative images (nearest to centroid)
- Grid of boundary/ambiguous images (lowest membership probability)
- Comparison strip: one row per cluster, 8 representative images, side by side
- Noise/outlier gallery

#### 5c. Geospatial Maps (`src/visualization/geospatial_maps.py`)

- Interactive folium map with image locations colored by cluster
- Heatmap of pattern density per cluster across the basin
- Temporal animation: cluster distribution map at pre → during → post (three panels or GIF)
- Overlay with microbial sampling locations (if coordinates provided)

#### 5d. Temporal Plots (`src/visualization/temporal_plots.py`)

- Stacked bar chart: cluster proportions by temporal phase
- Alluvial/Sankey diagram: pattern transitions between phases
- Embedding centroid drift vectors overlaid on UMAP
- Kernel density contour plots per phase (overlaid or side-by-side)

#### 5e. Publication Figure Generation (`src/visualization/publication_figures.py`)

```python
class PublicationFigureGenerator:
    """
    Generate camera-ready figures for peer-reviewed publication.
    
    Standards:
    - Colourblind-safe palettes (use 'colorspacious' for CVD simulation)
    - Default palette: custom discrete palette tested for deuteranopia
    - 300 DPI PNG + vector SVG for all figures
    - Font: Arial or Helvetica, minimum 8pt
    - Figure panels labeled (a), (b), (c)...
    - Consistent axis labels with units
    - Scale bars where applicable
    """
    
    PALETTE = [...]  # curated colourblind-safe colors
    
    def figure_1_method_overview(self) -> Figure:
        """Pipeline diagram: raw images → features → UMAP → clusters"""
    
    def figure_2_cluster_discovery(self) -> Figure:
        """
        Multi-panel:
        (a) UMAP scatter colored by cluster
        (b) Representative image grid per cluster
        (c) Silhouette plot
        """
    
    def figure_3_cluster_characterization(self) -> Figure:
        """
        (a) Heatmap of normalized texture features per cluster
        (b) Top distinguishing features (horizontal bar chart)
        (c) Radar/spider plot of feature profiles per cluster
        """
    
    def figure_4_temporal_dynamics(self) -> Figure:
        """
        (a) Stacked bar: cluster proportions by phase
        (b) UMAP density contours per phase (3 panels)
        (c) Transition Sankey diagram
        """
    
    def figure_5_geospatial(self) -> Figure:
        """
        (a) Basin map with cluster assignments
        (b) Spatial autocorrelation analysis
        (c) Overlay with microbial data (if available)
        """
```

---

### Phase 6: Inference & Interactive Exploration

#### 6a. New Image Prediction (`src/inference/predict.py`)

Once the pipeline is fitted (UMAP transform, HDBSCAN model, PCA transforms), new images can be projected into the existing pattern space.

```python
class PatternPredictor:
    """
    Assign new images to discovered clusters or locate them 
    in continuous pattern space.
    
    Uses HDBSCAN's approximate_predict for soft cluster assignment
    and the fitted UMAP transform for space coordinates.
    """
    
    def predict(self, image_path: str) -> PredictionResult:
        """
        Returns:
        - cluster_id: int (-1 if noise/novel)
        - cluster_probability: float
        - pattern_space_coords: (2,) or (3,) array
        - texture_profile: dict of texture descriptor values
        - nearest_cluster_images: list of similar images from training set
        """
        ...
```

#### 6b. Batch Prediction (`src/inference/batch_predict.py`)

Process all unannotated images through the fitted pipeline:
- Output CSV: `image_id, cluster_id, cluster_probability, umap_x, umap_y, temporal_phase, lat, lon`
- Output GeoJSON for GIS import

#### 6c. Interactive Explorer (`scripts/launch_explorer.py`)

Gradio app for the ecologist:

```
Tab 1: Pattern Space Explorer
  - Interactive UMAP scatter plot (plotly)
  - Click any point to see the image + texture profile + cluster info
  - Color by: cluster, temporal phase, altitude, any texture feature
  - Filter by: temporal phase, altitude group, cluster

Tab 2: Upload & Classify
  - Upload new image(s)
  - See: predicted cluster, confidence, pattern space position, texture profile
  - See: most similar images from the dataset
  - Grad-CAM-like visualization (optional): which image regions drive the DINOv2 embedding

Tab 3: Cluster Browser
  - Dropdown to select cluster
  - Shows: representative images, texture profile, temporal distribution, description

Tab 4: Temporal Comparison
  - Side-by-side density plots for selected temporal phases
  - Transition analysis for co-located images

Tab 5: Correlation Workbench (if microbial data provided)
  - Upload CSV with lat, lon, measurement columns
  - Auto-correlate with pattern space coordinates and cluster assignments
  - Generate correlation plots and statistical reports
```

---

## 4. Technology Stack

| Component | Tool | Purpose |
|---|---|---|
| Language | Python 3.11+ | |
| Deep Learning | PyTorch 2.x | DINOv2 inference, SAM for masking |
| Pretrained Models | torch.hub (facebookresearch/dinov2) | Feature extraction backbone |
| Image Processing | scikit-image, OpenCV, Pillow | Texture descriptors, tiling, preprocessing |
| Texture Analysis | scikit-image (GLCM, LBP, Gabor), scipy | Classical feature extraction |
| Dimensionality Reduction | umap-learn | UMAP embedding |
| Clustering | hdbscan | Density-based cluster discovery |
| Statistical Testing | scipy.stats, scikit-learn | Cluster validation, temporal tests |
| Data Management | h5py, pandas | Feature storage, metadata |
| Geospatial | folium, geopandas | Basin maps, spatial analysis |
| Visualization | matplotlib, seaborn, plotly | Publication figures, interactive plots |
| Interactive App | gradio | Ecologist-facing explorer |
| Config | PyYAML + dataclasses | All configs in YAML |
| Testing | pytest | Unit and integration tests |
| CLI | click or argparse | Script entry points |

**Notable: no annotation tooling, no training loop, no GPU training infrastructure needed.** The pipeline is inference-only on pretrained models + classical computation + clustering. This makes it dramatically simpler to run and reproduce.

---

## 5. Configuration Files

### `configs/data_config.yaml`

```yaml
raw_data_dir: "data/raw"
processed_dir: "data/processed"
metadata_output: "data/metadata/image_catalog.csv"

# Temporal phase boundaries (user must set these)
rain_event:
  pre_rain_end: "2017-06-01T00:00:00"
  during_rain_start: "2017-06-01T00:00:00"
  during_rain_end: "2017-06-15T00:00:00"
  post_rain_start: "2017-06-15T00:00:00"

# Altitude grouping thresholds (meters)
altitude_groups:
  high: [100, 999]
  mid: [30, 100]
  low: [5, 30]
  ground: [0, 5]

# Tiling
tiling:
  tile_size: 512
  overlap_fraction: 0.25
  min_valid_fraction: 0.8

# Ground masking
ground_masking:
  method: "sam"  # "sam" | "horizon" | "lower_crop"
  sam_checkpoint: "sam2_vit_b"
  crop_fraction: 0.6  # for lower_crop fallback

# Quality filtering
quality:
  blur_threshold: 100.0  # Laplacian variance
  exposure_clip_threshold: 0.05  # fraction of clipped pixels
  min_resolution: 256
```

### `configs/feature_config.yaml`

```yaml
output_dir: "outputs/features"
feature_store: "outputs/features/feature_store.h5"

dino:
  model_name: "dinov2_vitb14"
  input_size: 518
  batch_size: 32
  extract_patch_tokens: false  # set true for spatial analysis
  device: "cuda"  # "cuda" | "cpu" | "mps"

texture:
  glcm:
    distances: [1, 3, 5, 10]
    angles: [0, 0.785, 1.571, 2.356]  # 0, 45, 90, 135 degrees
    properties: ["contrast", "correlation", "energy", "homogeneity", "entropy"]
    levels: 256
  
  gabor:
    frequencies: [0.05, 0.1, 0.2, 0.4]
    orientations: [0, 30, 60, 90, 120, 150]  # degrees
    stats: ["mean", "variance"]
  
  fractal:
    method: "box_counting"
    threshold: "otsu"
  
  lacunarity:
    box_sizes: [5, 10, 20, 40, 80]
  
  lbp:
    radii: [1, 2, 3]
    method: "uniform"
  
  global:
    compute_fft_dominant_freq: true
    edge_detector: "canny"

fusion:
  dino_weight: 0.7
  texture_weight: 0.3
  pca_variance_threshold: 0.95
  normalize: true
```

### `configs/clustering_config.yaml`

```yaml
output_dir: "outputs/clusters"

dimensionality_reduction:
  # For clustering input
  high_dim:
    method: "umap"
    n_components: 15
    n_neighbors: 30
    min_dist: 0.1
    metric: "cosine"
  
  # For visualization
  viz_2d:
    method: "umap"
    n_components: 2
    n_neighbors: 30
    min_dist: 0.1
    metric: "cosine"
  
  viz_3d:
    method: "umap"
    n_components: 3
    n_neighbors: 30
    min_dist: 0.1
    metric: "cosine"
  
  # Alternative visualization for comparison
  tsne:
    perplexities: [15, 30, 50]
    n_components: 2

clustering:
  method: "hdbscan"
  min_cluster_size: 15
  min_samples: 5
  cluster_selection_method: "eom"
  
  # Run on these feature configurations for comparison
  feature_ablations:
    - name: "dino_only"
      features: "dino"
    - name: "texture_only"
      features: "texture"
    - name: "fused_default"
      features: "fused"
      dino_weight: 0.7
      texture_weight: 0.3
    - name: "fused_equal"
      features: "fused"
      dino_weight: 0.5
      texture_weight: 0.5

validation:
  bootstrap_n: 100
  compute_silhouette: true
  compute_davies_bouldin: true
  compute_calinski_harabasz: true
  compute_dbcv: true

random_seed: 42
```

### `configs/visualization_config.yaml`

```yaml
output_dir: "outputs/figures"

style:
  palette: "colorblind_safe"  # custom palette defined in code
  dpi: 300
  font_family: "Arial"
  font_size_base: 10
  export_formats: ["png", "svg"]
  figure_width_inches: 7  # single column journal width

cluster_gallery:
  images_per_cluster: 25
  grid_cols: 5
  thumbnail_size: 128

geospatial:
  basemap: "satellite"  # for folium
  marker_size: 8

temporal:
  phase_colors:
    pre_rain: "#2166ac"
    during_rain: "#67a9cf"
    post_rain: "#ef8a62"

gradio:
  server_port: 7860
  share: false
```

---

## 6. CLAUDE.md (Root Project Instructions)

```markdown
# Desert Patterned Ground Discovery

## Project Purpose
Unsupervised deep learning pipeline for discovering and characterizing patterned ground
types in Atacama Desert (Yungay basin) imagery. Supports microbial ecology research by
revealing natural pattern groupings without human-imposed categories.

## Architecture
This is NOT a supervised classification project. The pipeline is:
1. Ingest images → extract metadata → tile drone images → mask ground in ground-level photos
2. Extract dual features: DINOv2 embeddings (semantic) + classical texture descriptors (interpretable)
3. Fuse features → UMAP dimensionality reduction → HDBSCAN clustering
4. Characterize clusters with texture profiles → temporal analysis → geospatial mapping
5. Interactive Gradio explorer for the ecologist

No training loop. No labels. Feature extraction uses pretrained models (DINOv2) inference only.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Key Commands
```bash
python scripts/ingest_data.py --config configs/data_config.yaml
python scripts/extract_features.py --config configs/feature_config.yaml
python scripts/run_clustering.py --config configs/clustering_config.yaml
python scripts/generate_figures.py --config configs/visualization_config.yaml
python scripts/launch_explorer.py
pytest tests/
```

## Conventions
- All configs in YAML under `configs/`
- Raw data never modified; processed data in `data/processed/`
- Features cached in HDF5 at `outputs/features/feature_store.h5`
- Use type hints everywhere, dataclasses for configs
- Docstrings on all public functions
- Log with Python `logging` module, not print
- Colourblind-safe palettes for all visualizations
- All random operations seeded for reproducibility

## Agent Files
See `.claude/agents/` for specialized agent instructions:
- `data-pipeline.md` — ingestion, tiling, masking, preprocessing
- `feature-extraction.md` — DINOv2 embeddings, texture descriptors, fusion
- `clustering-analysis.md` — UMAP, HDBSCAN, characterization, temporal analysis
- `visualization-export.md` — plots, maps, publication figures, Gradio app
```

---

## 7. Agent Files

### Agent 1: `.claude/agents/data-pipeline.md`

```markdown
# Data Pipeline Agent

You are an expert data engineering agent for a desert terrain pattern
recognition project. This is an UNSUPERVISED pipeline — there are no labels
or annotation steps.

## Your Responsibilities

- Build the image ingestion pipeline: scan raw directories, extract EXIF metadata
  (GPS, altitude, timestamp, camera model, focal length) from every image
- Compute derived fields: ground sampling distance (GSD), altitude group, temporal
  phase assignment based on configurable date boundaries
- Implement image quality assessment: blur detection (Laplacian variance),
  exposure analysis (histogram clipping), minimum resolution filtering
- Build the drone image tiling module: slice large images into overlapping tiles
  (configurable size, overlap), reject uninformative edge tiles
- Build the ground-level image masking module: isolate ground surface from sky/horizon
  using SAM (Segment Anything Model) as primary method, with horizon detection and
  lower-crop as fallbacks
- Standardize all processed images to uniform size (518×518 for DINOv2 compatibility)
  with channel-wise normalization
- Generate the master image catalog CSV linking all processed images to their metadata
  and raw source paths

## Technical Context

- Images come from DJI drone (multiple altitudes: ~5m to ~200m+) and iPhone ground photos
- Temporal phases: pre-rain, during-rain, post-rain (Yungay basin, Atacama Desert, 2017)
- DJI drones store altitude in EXIF tag `RelativeAltitude` or XMP metadata
- GSD formula: `gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)`
- SAM usage: use `segment_anything` package, prompt with point grid in lower image region
- All configuration lives in `configs/data_config.yaml`

## Key Libraries

- `exifread`, `Pillow` for EXIF extraction
- `opencv-python` for image processing, blur detection
- `segment-anything` (SAM2) for ground masking
- `pandas` for metadata catalog
- `numpy` for image array operations

## Output Artifacts

- `data/metadata/image_catalog.csv` — master catalog of all images with full metadata
- `data/processed/tiles/` — tiled drone image patches
- `data/processed/ground_masked/` — masked ground-level image patches
- Console log with summary statistics (image counts per group, quality rejection rates)

## Quality Standards

- All preprocessing steps must be deterministic given a config (set random seeds)
- Handle corrupted/missing images gracefully with warnings in the log, never crashes
- Write unit tests for: EXIF parsing, GSD calculation, tiling geometry, quality metrics
- Validate that no raw files are modified — all outputs go to `data/processed/`
- The catalog CSV must be the single source of truth for all downstream pipeline steps
```

### Agent 2: `.claude/agents/feature-extraction.md`

```markdown
# Feature Extraction Agent

You are an expert agent for extracting deep learning embeddings and classical
texture features from desert terrain images. This is the core representation
layer of an unsupervised pattern discovery pipeline.

## Your Responsibilities

- Implement DINOv2 feature extraction using `torch.hub` (`facebookresearch/dinov2`)
- Extract [CLS] token embeddings (768-d for ViT-B/14) as global image descriptors
- Optionally extract patch-level tokens for future spatial analysis
- Implement a comprehensive classical texture descriptor extractor covering:
  - GLCM features (contrast, correlation, energy, homogeneity, entropy) at multiple
    distances and angles
  - Gabor filter bank responses (4 frequencies × 6 orientations, mean + variance)
  - Fractal dimension via box-counting on edge maps
  - Lacunarity at multiple box sizes
  - LBP (Local Binary Patterns) histograms with uniform patterns
  - Global statistics (intensity moments, edge density, FFT dominant frequency)
- Implement feature fusion: L2-normalize → PCA (95% variance) → weighted
  concatenation → final normalization
- Build the HDF5-based feature store for efficient caching and retrieval

## Technical Context

- DINOv2 was trained with self-supervision on 142M images — it produces features
  invariant to viewpoint, scale, and lighting. This is critical because our images
  range from 200m altitude nadir drone shots to handheld ground-level photos.
- DINOv2 ViT-B/14 expects 518×518 input (14×14 patch size, 37×37 grid)
- Texture descriptors serve a different purpose than DINOv2: they provide physically
  interpretable features that explain WHY clusters differ. Scientists need to say
  "this cluster has high fractal dimension" not just "the neural network says so."
- Feature fusion default: 70% DINOv2, 30% texture. DINOv2 is primary for clustering
  quality; texture is secondary for interpretability.

## Key Libraries

- `torch`, `torchvision` for DINOv2
- `scikit-image` for GLCM (`graycomatrix`, `graycoprops`), Gabor, LBP
- `scipy` for FFT, statistical computations
- `sklearn.decomposition.PCA` for dimensionality reduction
- `h5py` for feature storage
- `numpy` for all array operations

## Implementation Notes

- Batch DINOv2 extraction with DataLoader for GPU efficiency
- Compute texture features on grayscale images (convert RGB → gray first)
- For GLCM: quantize to 256 levels, compute at 4 distances × 4 angles, average
  over angles to reduce orientation sensitivity
- For fractal dimension: apply Otsu threshold → Canny edge detection → box counting
  on binary edge map
- Lacunarity: use sliding box algorithm, not gliding box (more efficient)
- LBP: use "uniform" method to reduce histogram bins to P+2 bins
- All features must be deterministic — avoid any randomness in extraction
- Cache aggressively: check if features exist in HDF5 before recomputing

## Output Artifacts

- `outputs/features/feature_store.h5` with datasets:
  - `/dino_cls`: (N, 768) float32
  - `/texture`: (N, ~90) float32
  - `/image_ids`: (N,) string
  - `/dino_pca`: PCA-reduced DINOv2 features
  - `/texture_pca`: PCA-reduced texture features
  - `/fused`: final fused feature vectors
- Feature extraction summary log: time per image, total features, PCA variance retained

## Quality Standards

- Unit tests for each texture feature on synthetic images (e.g., checkerboard should have
  high GLCM contrast, circle should have known fractal dimension ~1.0)
- Verify DINOv2 outputs are deterministic across runs
- Validate feature store integrity (no NaN, no inf, correct shapes)
- Profile memory usage — DINOv2 batch extraction should not OOM
```

### Agent 3: `.claude/agents/clustering-analysis.md`

```markdown
# Clustering & Analysis Agent

You are an expert agent for unsupervised pattern discovery and scientific
analysis of desert terrain imagery. You discover natural pattern groupings
from features and characterize them for ecological research.

## Your Responsibilities

- Implement UMAP dimensionality reduction for both clustering input (10-20d)
  and visualization (2d, 3d)
- Implement HDBSCAN clustering on reduced features, with soft cluster assignments
  and outlier detection
- Run clustering across multiple feature configurations (DINOv2 only, texture only,
  fused with different weights) and compare using cluster quality metrics
- Characterize each discovered cluster:
  - Statistical profile of texture features (mean ± std per feature)
  - Feature importance ranking (Kruskal-Wallis H-test per feature across clusters)
  - Metadata distribution (source type, altitude, temporal phase)
  - Representative and boundary images
  - Auto-generated natural language description
- Temporal analysis:
  - Cluster proportion changes across pre/during/post rain phases
  - Transition matrices for co-located images across time
  - Embedding centroid drift analysis
  - Chi-squared tests for statistical significance of phase differences
- Continuous pattern space analysis:
  - Kernel density estimation in UMAP space per temporal phase
  - Correlation framework for external measurements (microbial data)
- Bootstrap stability analysis: verify cluster robustness

## Technical Context

- HDBSCAN is preferred over k-means because: no need to specify k, identifies noise,
  handles varying cluster densities, provides membership probabilities
- UMAP with cosine metric works well for L2-normalized embedding vectors
- Use `cluster_selection_method="eom"` (Excess of Mass) for HDBSCAN — better for
  varying density clusters typical in natural data
- Temporal phase transitions can only be computed for images from overlapping geographic
  locations — use GPS coordinates to match images across time
- All statistical tests should report effect sizes, not just p-values

## Key Libraries

- `umap-learn` for UMAP
- `hdbscan` for clustering
- `scipy.stats` for statistical tests (chi2_contingency, kruskal, spearmanr)
- `sklearn.metrics` for silhouette_score, davies_bouldin_score, calinski_harabasz_score
- `numpy`, `pandas` for data manipulation

## Output Artifacts

- `outputs/clusters/cluster_assignments.csv` — image_id, cluster_id, probability, umap_x, umap_y
- `outputs/clusters/cluster_profiles.json` — per-cluster characterization
- `outputs/clusters/quality_metrics.json` — all validation scores per configuration
- `outputs/clusters/temporal_analysis.json` — phase distributions, transitions, tests
- `outputs/clusters/feature_importance.csv` — ranked features by discriminative power
- `outputs/clusters/continuous_space.csv` — pattern space coordinates for all images

## Quality Standards

- Set random seeds everywhere (UMAP, HDBSCAN, bootstrap)
- Report noise fraction — if >30% of images are noise, parameters need tuning
- Verify clusters are stable with bootstrap analysis (adjusted Rand > 0.7)
- All statistical tests must include multiple comparison correction (Bonferroni or FDR)
- Generate a summary report (markdown) with key findings and cluster descriptions
- Write tests for: UMAP output shape, HDBSCAN with known synthetic clusters,
  metric computation correctness
```

### Agent 4: `.claude/agents/visualization-export.md`

```markdown
# Visualization & Export Agent

You are an expert agent for scientific visualization and interactive tool
development. You create publication-quality figures and a Gradio-based
exploration interface for a desert microbial ecology research project.

## Your Responsibilities

- Generate publication-quality figures for all analysis results
- Build an interactive Gradio explorer application
- Create geospatial visualizations of pattern distributions
- Export results in formats suitable for GIS tools and scientific publications

## Publication Figure Standards

EVERY figure must meet these requirements:
- Colourblind-safe palette (test with deuteranopia, protanopia, tritanopia simulation)
- Default discrete palette: use a curated set tested for all CVD types
  (e.g., Wong palette or Tol palette)
- Continuous colormaps: viridis or cividis only (never jet, rainbow, or hot)
- Export as both 300 DPI PNG and vector SVG
- Font: Arial, minimum 8pt for labels, 10pt for axis labels
- Panel labels: (a), (b), (c) in bold, top-left of each panel
- Consistent axis labels with units
- Scale bars on all image-based figures
- Single-column width: 3.5 inches; double-column: 7 inches
- No chartjunk: minimal gridlines, no unnecessary borders or backgrounds

## Key Figures to Generate

1. **Method overview** — schematic of the pipeline (can be a clean diagram)
2. **Cluster discovery** — UMAP scatter + representative images + silhouette plot
3. **Cluster characterization** — texture feature heatmap + radar plots + feature importance
4. **Temporal dynamics** — stacked bars + density contours per phase + transition Sankey
5. **Geospatial mapping** — basin map with cluster assignments + microbial overlay
6. **Feature ablation** — comparison of DINOv2-only vs texture-only vs fused clustering

## Gradio Explorer Application

Build with the following tabs:
- **Pattern Space Explorer**: Interactive plotly scatter of UMAP space. Click points to see
  images. Color by cluster/phase/altitude/any texture feature. Filter by phase/altitude/cluster.
- **Upload & Classify**: Drag-and-drop new images. Show predicted cluster, confidence,
  nearest neighbors from dataset, texture profile.
- **Cluster Browser**: Dropdown per cluster → representative images, texture profile chart,
  temporal distribution, auto-generated description.
- **Temporal Comparison**: Side-by-side KDE plots for selected phases. Animation of pattern
  space density evolution.
- **Correlation Workbench**: Upload CSV of measurements with lat/lon columns.
  Auto-compute correlations with pattern space coordinates and cluster assignments.
  Scatter plots with regression lines and confidence bands.

## Geospatial Outputs

- Folium interactive maps with satellite basemap
- GeoJSON export of all image locations with cluster assignments (for QGIS import)
- KML export option for Google Earth visualization
- Heatmap layers per cluster showing spatial density

## Key Libraries

- `matplotlib` + `seaborn` for static figures
- `plotly` for interactive plots
- `gradio` for the explorer app
- `folium` for geospatial maps
- `geopandas` for GeoJSON generation
- `colorspacious` for CVD palette simulation

## Quality Standards

- Test all palettes for CVD accessibility before finalizing
- All figures must render correctly at final publication size (check legibility at 3.5")
- Gradio app must handle edge cases: no GPS data, single cluster, empty filters
- Interactive plots must be responsive with up to 10,000 points
- Write a figure generation script that reproduces ALL figures from cached results
  (no recomputation needed)
- Every figure must have a caption-ready description in the output report
```

---

## 8. Execution Order

1. **Project scaffolding** — directory structure, `pyproject.toml`, configs, CLAUDE.md, agent files
2. **Data ingestion** — metadata extraction, quality assessment, EDA notebook
3. **Preprocessing** — tiling, ground masking, standardization
4. **Feature extraction** — DINOv2 embeddings + texture descriptors + fusion
5. **Clustering** — UMAP + HDBSCAN on multiple feature configs, select best
6. **Characterization** — cluster profiles, feature importance, descriptions
7. **Temporal analysis** — phase distributions, transitions, statistical tests
8. **Visualization** — all publication figures + geospatial maps
9. **Gradio explorer** — interactive app for the ecologist
10. **Correlation framework** — ready for when microbial measurements are provided

Each phase produces standalone outputs. The ecologist gets useful results from Phase 5 onward — they can see what patterns the model discovered and start forming hypotheses immediately, even before temporal analysis and correlation work are complete.
```

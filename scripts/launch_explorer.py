#!/usr/bin/env python3
"""Interactive Gradio explorer for the desert patterns pipeline.

Five-tab interface for exploring clustering results interactively:
  Tab 1 — Pattern Space: interactive plotly UMAP with colour-by dropdown
  Tab 2 — Cluster Browser: representative gallery + top feature bar chart
  Tab 3 — Temporal: phase distribution + density contours
  Tab 4 — Upload & Classify: classify new images into discovered clusters
  Tab 5 — Correlation Workbench: correlate pattern space with measurements

Usage:
    pip install -e ".[viz]"
    python scripts/launch_explorer.py
    python scripts/launch_explorer.py --config configs/visualization_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch interactive pattern explorer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "visualization_config.yaml",
        help="Path to visualization configuration YAML file.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Ablation name to visualize.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import gradio with helpful error
    try:
        import gradio as gr
    except ImportError:
        logger.error(
            "Gradio is required for the interactive explorer. "
            "Install it with: pip install -e '.[viz]'"
        )
        sys.exit(1)

    import plotly.graph_objects as go

    # Load config
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    cluster_dir = PROJECT_ROOT / raw_config["cluster_dir"]
    manifest_path = PROJECT_ROOT / raw_config["manifest_path"]
    catalog_path = PROJECT_ROOT / raw_config["catalog_path"]
    ablation = args.ablation or raw_config.get("ablation_name", "fused_default")

    # Load data once at startup
    assignments_df = pd.read_csv(cluster_dir / f"{ablation}_assignments.csv")
    logger.info("Loaded %d assignments", len(assignments_df))

    profiles_path = cluster_dir / f"{ablation}_profiles.json"
    profiles = []
    if profiles_path.exists():
        with open(profiles_path) as f:
            profiles = json.load(f)

    temporal_path = cluster_dir / f"{ablation}_temporal.json"
    temporal_data = {}
    if temporal_path.exists():
        with open(temporal_path) as f:
            temporal_data = json.load(f)

    continuous_path = cluster_dir / f"{ablation}_continuous_space.csv"
    continuous_df = None
    if continuous_path.exists():
        continuous_df = pd.read_csv(continuous_path)

    importance_path = cluster_dir / f"{ablation}_feature_importance.csv"
    importance_df = None
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)

    # Metadata
    manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else None
    catalog_df = pd.read_csv(catalog_path) if catalog_path.exists() else None

    metadata_df = None
    patch_paths: dict[str, Path] = {}
    if manifest_df is not None:
        for _, row in manifest_df.iterrows():
            pid = row.get("patch_id", row.get("image_id"))
            out = row.get("patch_path", row.get("output_path"))
            if pid and out:
                patch_paths[str(pid)] = Path(out)

        if catalog_df is not None:
            metadata_df = manifest_df.merge(
                catalog_df,
                left_on="parent_image_id",
                right_on="image_id",
                how="left",
                suffixes=("_manifest", "_catalog"),
            )
            metadata_df["image_id"] = metadata_df["patch_id"]
        else:
            metadata_df = manifest_df.copy()
            if "patch_id" in metadata_df.columns:
                metadata_df["image_id"] = metadata_df["patch_id"]

    # Available colour-by options
    color_options = ["cluster_id"]
    if metadata_df is not None:
        for col in ("temporal_phase", "altitude_group", "source_type"):
            if col in metadata_df.columns:
                color_options.append(col)

    # Cluster IDs for browser
    cluster_ids = sorted(set(assignments_df["cluster_id"].values) - {-1})
    cluster_choices = [f"Cluster {c}" for c in cluster_ids]

    from src.visualization.embedding_plots import create_interactive_umap
    from src.visualization.style import get_cluster_colors, setup_matplotlib_style, FigureStyle

    style = FigureStyle()
    setup_matplotlib_style(style)

    # Try to load predictor for Tab 4
    predictor = None
    models_dir = PROJECT_ROOT / "outputs" / "models" / ablation
    if models_dir.is_dir() and (models_dir / "pipeline_config.json").exists():
        try:
            from src.inference.predict import PatternPredictor
            predictor = PatternPredictor(models_dir)
            logger.info("Predictor loaded from %s", models_dir)
        except Exception as e:
            logger.warning("Could not load predictor: %s", e)

    # ---- Build Gradio interface ----

    with gr.Blocks(title="Desert Pattern Explorer") as app:
        gr.Markdown("# Desert Pattern Explorer")
        gr.Markdown(f"Ablation: **{ablation}** — {len(assignments_df)} patches")

        # Tab 1: Pattern Space
        with gr.Tab("Pattern Space"):
            color_dropdown = gr.Dropdown(
                choices=color_options,
                value="cluster_id",
                label="Colour by",
            )
            umap_plot = gr.Plot(label="UMAP Embedding")

            def update_umap(color_col: str):
                return create_interactive_umap(assignments_df, metadata_df, color_col)

            color_dropdown.change(update_umap, inputs=color_dropdown, outputs=umap_plot)
            # Initial plot
            app.load(lambda: update_umap("cluster_id"), outputs=umap_plot)

        # Tab 2: Cluster Browser
        with gr.Tab("Cluster Browser"):
            cluster_dropdown = gr.Dropdown(
                choices=cluster_choices,
                value=cluster_choices[0] if cluster_choices else None,
                label="Select Cluster",
            )
            with gr.Row():
                gallery_output = gr.Gallery(label="Representative Images", columns=5, height=400)
                with gr.Column():
                    cluster_info = gr.Markdown()
                    feature_plot = gr.Plot(label="Top Features")

            def update_cluster(choice: str):
                if not choice:
                    return [], "", go.Figure()

                cid = int(choice.replace("Cluster ", ""))
                profile = next((p for p in profiles if p.get("cluster_id") == cid), None)

                # Gallery images
                images = []
                if profile:
                    for img_id in profile.get("representative_ids", [])[:25]:
                        path = patch_paths.get(img_id)
                        if path and path.exists():
                            images.append(str(path))

                # Info text
                info = f"### Cluster {cid}\n"
                if profile:
                    info += f"- **Size**: {profile.get('size', '?')}\n"
                    info += f"- **Description**: {profile.get('description', 'N/A')}\n"
                    dist = profile.get("distinguishing_features", [])
                    if dist:
                        info += "\n**Top distinguishing features:**\n"
                        for feat in dist[:5]:
                            info += f"- {feat.get('name', '?')}: z={feat.get('z_score', 0):.2f}\n"

                # Feature importance bar chart
                feat_fig = go.Figure()
                if importance_df is not None and not importance_df.empty:
                    # Show top 10 by effect size
                    top = importance_df.nlargest(10, "eta_squared") if "eta_squared" in importance_df.columns else importance_df.head(10)
                    name_col = "feature_name" if "feature_name" in top.columns else top.columns[0]
                    val_col = "eta_squared" if "eta_squared" in top.columns else top.columns[-1]
                    colors = get_cluster_colors(len(top))
                    feat_fig.add_trace(go.Bar(
                        x=top[val_col].values,
                        y=top[name_col].values,
                        orientation="h",
                        marker_color=colors[:len(top)],
                    ))
                    feat_fig.update_layout(
                        title="Feature Importance (eta²)",
                        xaxis_title="Effect size",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=150, r=20, t=40, b=40),
                        template="plotly_white",
                    )

                return images, info, feat_fig

            cluster_dropdown.change(
                update_cluster,
                inputs=cluster_dropdown,
                outputs=[gallery_output, cluster_info, feature_plot],
            )

        # Tab 3: Temporal
        with gr.Tab("Temporal"):
            with gr.Row():
                phase_plot = gr.Plot(label="Phase Distribution")
                density_plot = gr.Plot(label="Density Contours")

            def build_temporal_plots():
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Phase distribution
                phase_fig = go.Figure()
                dist = temporal_data.get("phase_distributions", {})
                if dist:
                    df = pd.DataFrame(dist).T
                    phases = df.columns.tolist()
                    colors = get_cluster_colors(len(phases))
                    for i, phase in enumerate(phases):
                        phase_fig.add_trace(go.Bar(
                            name=phase,
                            x=[str(c) for c in df.index],
                            y=df[phase].values.astype(float),
                            marker_color=colors[i],
                        ))
                    phase_fig.update_layout(
                        barmode="stack",
                        title="Phase Distribution by Cluster",
                        xaxis_title="Cluster",
                        yaxis_title="Fraction",
                        template="plotly_white",
                    )

                # Density as scatter (simplified for Gradio)
                density_fig = go.Figure()
                if continuous_df is not None and metadata_df is not None and "temporal_phase" in metadata_df.columns:
                    merged = continuous_df.merge(
                        metadata_df[["image_id", "temporal_phase"]].drop_duplicates(),
                        on="image_id", how="left",
                    )
                    phases_present = sorted(merged["temporal_phase"].dropna().unique())
                    colors = get_cluster_colors(len(phases_present))
                    for i, phase in enumerate(phases_present):
                        mask = merged["temporal_phase"] == phase
                        density_fig.add_trace(go.Scattergl(
                            x=merged.loc[mask, "pattern_x"],
                            y=merged.loc[mask, "pattern_y"],
                            mode="markers",
                            marker=dict(color=colors[i], size=5, opacity=0.6),
                            name=phase,
                        ))
                    density_fig.update_layout(
                        title="Pattern Space by Phase",
                        xaxis_title="Pattern dim 1",
                        yaxis_title="Pattern dim 2",
                        template="plotly_white",
                    )

                return phase_fig, density_fig

            app.load(build_temporal_plots, outputs=[phase_plot, density_plot])

        # Tab 4: Upload & Classify
        with gr.Tab("Upload & Classify"):
            if predictor is None:
                gr.Markdown(
                    "**Models not available.** Run clustering with `--save-models` first:\n\n"
                    "```\npython scripts/run_clustering.py --save-models\n```"
                )
            else:
                upload_image = gr.Image(type="filepath", label="Upload an image")
                with gr.Row():
                    classify_plot = gr.Plot(label="Position in Pattern Space")
                    with gr.Column():
                        classify_info = gr.Markdown()
                        neighbor_gallery = gr.Gallery(
                            label="Nearest Neighbors", columns=5, height=300
                        )

                # Pre-load curated DINO features for NN lookup
                _train_dino = None
                _curation_dirs = None

                def _load_train_features():
                    nonlocal _train_dino, _curation_dirs
                    if _train_dino is not None:
                        return
                    from src.features.feature_store import FeatureStore
                    store = FeatureStore(PROJECT_ROOT / raw_config["feature_store"])
                    raw_dino = store.load_dino()
                    all_ids = store.load_image_ids()

                    # Apply curation if present
                    curation_path = models_dir / "curation.joblib"
                    if curation_path.exists():
                        _curation_dirs = joblib.load(curation_path)
                        from src.features.feature_curation import project_out_directions
                        raw_dino = project_out_directions(raw_dino, _curation_dirs)

                    # Filter to only IDs in assignments
                    keep_ids = set(assignments_df["image_id"])
                    mask = [iid in keep_ids for iid in all_ids]
                    _train_dino = raw_dino[mask]

                def classify_uploaded(filepath: str | None):
                    if filepath is None:
                        return go.Figure(), "", []

                    import subprocess

                    _load_train_features()

                    # Extract DINO features in subprocess (avoids torch in main process)
                    # Returns raw 768-d features as JSON array
                    script = f"""\
import sys, json
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from src.data.preprocessing import standardize_image
from src.features.dino_embeddings import DinoFeatureExtractor

bgr = cv2.imread({filepath!r})
std = standardize_image(bgr, target_size=518)
rgb = cv2.cvtColor(std, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(rgb)

extractor = DinoFeatureExtractor()
features = extractor.extract_cls(pil_img)
json.dump(features.tolist(), sys.stdout)
"""
                    try:
                        proc = subprocess.run(
                            [sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=120,
                        )
                        if proc.returncode != 0:
                            err = proc.stderr.strip().split("\n")[-1] if proc.stderr else "unknown error"
                            return go.Figure(), f"**Error extracting features:** {err}", []
                        features = np.array(json.loads(proc.stdout))
                    except subprocess.TimeoutExpired:
                        return go.Figure(), "**Error:** Feature extraction timed out", []
                    except Exception as e:
                        return go.Figure(), f"**Error:** {e}", []

                    # Apply curation
                    if _curation_dirs is not None:
                        from src.features.feature_curation import project_out_directions
                        features = project_out_directions(
                            features.reshape(1, -1), _curation_dirs
                        )[0]

                    # Nearest-neighbor in curated DINO space
                    dists = np.linalg.norm(_train_dino - features.reshape(1, -1), axis=1)
                    nn_idx = np.argsort(dists)

                    # Assign cluster = majority vote of k nearest neighbors
                    k = 5
                    top_k_labels = assignments_df.iloc[nn_idx[:k]]["cluster_id"].values
                    from collections import Counter
                    vote = Counter(top_k_labels)
                    cluster_id = vote.most_common(1)[0][0]
                    confidence = vote.most_common(1)[0][1] / k

                    # Estimated UMAP position = mean of k-NN positions
                    top_k_coords = assignments_df.iloc[nn_idx[:k]][["umap_x", "umap_y"]].values
                    umap_2d = top_k_coords.mean(axis=0)

                    # Info text
                    info = (
                        f"### Prediction (k-NN, k={k})\n"
                        f"- **Cluster:** {cluster_id}\n"
                        f"- **Confidence:** {confidence:.1%} ({int(confidence*k)}/{k} neighbors agree)\n"
                        f"- **Estimated UMAP:** ({umap_2d[0]:.3f}, {umap_2d[1]:.3f})\n"
                        f"- **Distance to nearest:** {dists[nn_idx[0]]:.3f}\n"
                    )
                    profile = next(
                        (p for p in profiles if p.get("cluster_id") == cluster_id),
                        None,
                    )
                    if profile:
                        info += f"- **Description:** {profile.get('description', 'N/A')}\n"

                    # Build scatter with new point
                    fig = create_interactive_umap(assignments_df, metadata_df, "cluster_id")
                    fig.add_trace(go.Scatter(
                        x=[float(umap_2d[0])],
                        y=[float(umap_2d[1])],
                        mode="markers",
                        marker=dict(
                            symbol="star",
                            size=18,
                            color="red",
                            line=dict(width=2, color="black"),
                        ),
                        name="New image",
                        showlegend=True,
                    ))

                    # Nearest neighbor gallery
                    neighbor_images = []
                    for idx in nn_idx[:10]:
                        nid = assignments_df.iloc[idx]["image_id"]
                        path = patch_paths.get(nid)
                        if path and path.exists():
                            neighbor_images.append(str(path))

                    return fig, info, neighbor_images

                upload_image.change(
                    classify_uploaded,
                    inputs=upload_image,
                    outputs=[classify_plot, classify_info, neighbor_gallery],
                )

        # Tab 5: Correlation Workbench
        with gr.Tab("Correlation Workbench"):
            gr.Markdown("Upload a CSV with GPS coordinates and measurements "
                        "to correlate with pattern space.")
            measurement_file = gr.File(
                file_types=[".csv"], label="Measurement CSV"
            )
            measurement_col_dropdown = gr.Dropdown(
                choices=[], label="Measurement column", interactive=True,
            )
            corr_run_btn = gr.Button("Run Correlation")
            with gr.Row():
                corr_plot = gr.Plot(label="Measurement Overlay")
                corr_info = gr.Markdown()

            def update_columns(file_obj):
                if file_obj is None:
                    return gr.update(choices=[], value=None)
                try:
                    df = pd.read_csv(file_obj.name if hasattr(file_obj, "name") else file_obj)
                    skip = {"lat", "lon", "latitude", "longitude"}
                    cols = [c for c in df.columns if c.lower() not in skip]
                    return gr.update(choices=cols, value=cols[0] if cols else None)
                except Exception:
                    return gr.update(choices=[], value=None)

            measurement_file.change(
                update_columns,
                inputs=measurement_file,
                outputs=measurement_col_dropdown,
            )

            def run_correlation(file_obj, meas_col: str | None):
                if file_obj is None or not meas_col:
                    return go.Figure(), "Upload a CSV and select a column."

                try:
                    meas_df = pd.read_csv(
                        file_obj.name if hasattr(file_obj, "name") else file_obj
                    )
                except Exception as e:
                    return go.Figure(), f"**Error reading CSV:** {e}"

                if metadata_df is None:
                    return go.Figure(), "No image metadata available."

                from src.clustering.continuous_space import ContinuousPatternSpace

                cps = ContinuousPatternSpace(
                    embeddings_2d=assignments_df[["umap_x", "umap_y"]].values,
                    image_ids=assignments_df["image_id"].tolist(),
                    metadata_df=metadata_df,
                )
                corr = cps.correlate_with_measurements(meas_df, meas_col)

                info = (
                    f"### Correlation: {meas_col}\n"
                    f"- **Spearman dim 1:** r={corr.spearman_dim1[0]:.3f} "
                    f"(p={corr.spearman_dim1[1]:.4f})\n"
                    f"- **Spearman dim 2:** r={corr.spearman_dim2[0]:.3f} "
                    f"(p={corr.spearman_dim2[1]:.4f})\n"
                    f"- **RF R²:** {corr.combined_r2:.3f}\n"
                )

                # Build overlay plot
                fig = go.Figure()
                # Background: existing patches in gray
                fig.add_trace(go.Scattergl(
                    x=assignments_df["umap_x"],
                    y=assignments_df["umap_y"],
                    mode="markers",
                    marker=dict(color="#CCCCCC", size=4, opacity=0.4),
                    name="Patches",
                    showlegend=True,
                ))

                # Overlay measurement points matched to nearest image
                lat_col = "lat" if "lat" in meas_df.columns else "latitude"
                lon_col = "lon" if "lon" in meas_df.columns else "longitude"
                if lat_col in meas_df.columns and lon_col in meas_df.columns:
                    from src.clustering.continuous_space import _haversine_m

                    meta_gps = metadata_df.dropna(subset=["lat", "lon"])
                    id_to_idx = {
                        iid: i for i, iid in enumerate(assignments_df["image_id"])
                    }
                    mx, my, mvals = [], [], []
                    for _, mrow in meas_df.iterrows():
                        mlat, mlon = mrow[lat_col], mrow[lon_col]
                        if pd.isna(mlat) or pd.isna(mlon):
                            continue
                        best_dist, best_idx = float("inf"), -1
                        for _, irow in meta_gps.iterrows():
                            iid = irow.get("image_id", irow.get("patch_id"))
                            if iid not in id_to_idx:
                                continue
                            dist = _haversine_m(mlat, mlon, irow["lat"], irow["lon"])
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = id_to_idx[iid]
                        if best_idx >= 0:
                            mx.append(float(assignments_df.iloc[best_idx]["umap_x"]))
                            my.append(float(assignments_df.iloc[best_idx]["umap_y"]))
                            mvals.append(float(mrow[meas_col]))

                    if mx:
                        fig.add_trace(go.Scatter(
                            x=mx, y=my,
                            mode="markers",
                            marker=dict(
                                color=mvals,
                                colorscale="Viridis",
                                size=12,
                                colorbar=dict(title=meas_col),
                                line=dict(width=1, color="black"),
                            ),
                            text=[f"{v:.2f}" for v in mvals],
                            name=meas_col,
                        ))

                fig.update_layout(
                    title=f"Pattern Space + {meas_col}",
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    template="plotly_white",
                )
                return fig, info

            corr_run_btn.click(
                run_correlation,
                inputs=[measurement_file, measurement_col_dropdown],
                outputs=[corr_plot, corr_info],
            )

    # Launch
    gradio_config = raw_config.get("gradio", {})
    port = gradio_config.get("server_port", 7860)
    share = gradio_config.get("share", False)

    logger.info("Launching explorer on port %d (share=%s)", port, share)
    app.launch(server_port=port, share=share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

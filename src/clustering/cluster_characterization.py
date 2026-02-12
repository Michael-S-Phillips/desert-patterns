"""Cluster profiling, feature importance, and interpretation.

For each discovered cluster, compute a statistical profile including
texture means/stds, distinguishing features, metadata distributions,
representative and boundary images, and auto-generated descriptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kruskal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ClusterProfile:
    """Statistical profile for a single cluster."""

    cluster_id: int
    size: int
    texture_mean: np.ndarray  # (90,)
    texture_std: np.ndarray  # (90,)
    distinguishing_features: list[dict]  # top-10 [{name, value, z_score}]
    metadata_distribution: dict  # {field: {value: fraction}}
    representative_ids: list[str]
    boundary_ids: list[str]
    description: str


# ---------------------------------------------------------------------------
# Characterizer
# ---------------------------------------------------------------------------


class ClusterCharacterizer:
    """Compute statistical profiles for discovered clusters.

    Args:
        labels: Cluster labels ``(N,)`` from HDBSCAN (-1 = noise).
        features_texture: Texture feature matrix ``(N, 90)``.
        image_ids: Image/patch identifiers ``(N,)``.
        feature_names: Names for each texture feature column.
        metadata_df: DataFrame with columns including ``image_id`` plus
            any metadata fields (``source_type``, ``altitude_group``,
            ``temporal_phase``, etc.).
        features_fused: Optional fused features ``(N, D)`` for centroid
            distance computation.  Falls back to texture features if absent.
    """

    def __init__(
        self,
        labels: np.ndarray,
        features_texture: np.ndarray,
        image_ids: list[str],
        feature_names: list[str],
        metadata_df: pd.DataFrame,
        features_fused: np.ndarray | None = None,
    ) -> None:
        self.labels = labels
        self.features_texture = features_texture
        self.image_ids = image_ids
        self.feature_names = feature_names
        self.metadata_df = metadata_df
        self.features_fused = features_fused if features_fused is not None else features_texture

        self._cluster_ids = sorted(set(labels[labels >= 0]))
        self._n_clusters = len(self._cluster_ids)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def characterize(self, cluster_id: int) -> ClusterProfile:
        """Compute the statistical profile for a single cluster.

        Args:
            cluster_id: The cluster label to profile.

        Returns:
            ClusterProfile with texture stats, distinguishing features, etc.
        """
        mask = self.labels == cluster_id
        size = int(mask.sum())

        tex = self.features_texture[mask]
        tex_mean = tex.mean(axis=0)
        tex_std = tex.std(axis=0)

        # Global statistics for z-score computation
        all_nn = self.labels >= 0
        global_mean = self.features_texture[all_nn].mean(axis=0)
        global_std = self.features_texture[all_nn].std(axis=0)
        # Avoid division by zero
        global_std_safe = np.where(global_std == 0, 1.0, global_std)
        z_scores = (tex_mean - global_mean) / global_std_safe

        # Top-10 distinguishing features by absolute z-score
        top_indices = np.argsort(np.abs(z_scores))[::-1][:10]
        distinguishing = [
            {
                "name": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                "value": float(tex_mean[i]),
                "z_score": float(z_scores[i]),
            }
            for i in top_indices
        ]

        # Metadata distributions
        cluster_ids_list = [self.image_ids[i] for i in range(len(self.labels)) if mask[i]]
        meta_dist = self._compute_metadata_distribution(cluster_ids_list)

        # Representative and boundary IDs
        representative = self._find_representative_ids(cluster_id, n=25)
        boundary = self._find_boundary_ids(cluster_id, n=10)

        # Auto-description
        profile = ClusterProfile(
            cluster_id=cluster_id,
            size=size,
            texture_mean=tex_mean,
            texture_std=tex_std,
            distinguishing_features=distinguishing,
            metadata_distribution=meta_dist,
            representative_ids=representative,
            boundary_ids=boundary,
            description="",  # filled below
        )
        profile.description = self.generate_description(profile)

        return profile

    def characterize_all(self) -> list[ClusterProfile]:
        """Compute profiles for all clusters.

        Returns:
            List of ClusterProfile, one per cluster (excluding noise).
        """
        profiles = []
        for cid in self._cluster_ids:
            profiles.append(self.characterize(cid))
        return profiles

    def compute_feature_importance(self) -> pd.DataFrame:
        """Rank texture features by discriminative power between clusters.

        Uses Kruskal-Wallis H-test per feature across clusters (excluding
        noise), with Bonferroni correction.

        Returns:
            DataFrame with columns: ``feature``, ``h_statistic``, ``p_value``,
            ``p_adjusted``, ``effect_size`` (eta-squared).
        """
        non_noise = self.labels >= 0
        labels_nn = self.labels[non_noise]
        features_nn = self.features_texture[non_noise]

        unique_labels = sorted(set(labels_nn))
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            logger.warning("Feature importance requires ≥2 clusters, found %d", n_clusters)
            return pd.DataFrame(
                columns=["feature", "h_statistic", "p_value", "p_adjusted", "effect_size"]
            )

        n_features = features_nn.shape[1]
        n_total = len(labels_nn)
        results = []

        for fi in range(n_features):
            groups = [features_nn[labels_nn == cid, fi] for cid in unique_labels]
            # Skip features with zero variance everywhere
            if all(g.std() == 0 for g in groups):
                results.append({
                    "feature": self.feature_names[fi] if fi < len(self.feature_names)
                    else f"feature_{fi}",
                    "h_statistic": 0.0,
                    "p_value": 1.0,
                    "p_adjusted": 1.0,
                    "effect_size": 0.0,
                })
                continue

            h_stat, p_val = kruskal(*groups)

            # Eta-squared: (H - k + 1) / (n - k)
            eta_sq = (h_stat - n_clusters + 1) / (n_total - n_clusters)
            eta_sq = max(0.0, eta_sq)

            results.append({
                "feature": self.feature_names[fi] if fi < len(self.feature_names)
                else f"feature_{fi}",
                "h_statistic": float(h_stat),
                "p_value": float(p_val),
                "p_adjusted": 0.0,  # placeholder, filled below
                "effect_size": float(eta_sq),
            })

        df = pd.DataFrame(results)
        # Bonferroni correction
        df["p_adjusted"] = (df["p_value"] * n_features).clip(upper=1.0)
        # Sort by H-statistic descending
        df = df.sort_values("h_statistic", ascending=False).reset_index(drop=True)

        logger.info(
            "Feature importance: %d features tested, %d significant at p<0.05 (adjusted)",
            len(df),
            int((df["p_adjusted"] < 0.05).sum()),
        )

        return df

    def generate_description(self, profile: ClusterProfile) -> str:
        """Auto-generate a plain-text description from a cluster profile.

        Args:
            profile: The cluster profile to describe.

        Returns:
            Human-readable description string.
        """
        parts = [f"Cluster {profile.cluster_id} (n={profile.size}):"]

        # Top-3 distinguishing features
        top3 = profile.distinguishing_features[:3]
        if top3:
            feat_strs = []
            for f in top3:
                direction = "high" if f["z_score"] > 0 else "low"
                feat_strs.append(f"{direction} {f['name']} (z={f['z_score']:+.2f})")
            parts.append("Characterized by " + ", ".join(feat_strs) + ".")

        # Metadata distribution highlights
        for field_name, dist in profile.metadata_distribution.items():
            if not dist:
                continue
            # Report the dominant category
            top_cat = max(dist, key=dist.get)
            top_frac = dist[top_cat]
            if top_frac > 0.5:
                parts.append(
                    f"Predominantly {top_cat} {field_name} ({top_frac:.0%})."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_representative_ids(self, cluster_id: int, n: int = 25) -> list[str]:
        """Find images closest to the cluster centroid in fused feature space.

        Args:
            cluster_id: Cluster to examine.
            n: Number of representative IDs to return.

        Returns:
            List of image IDs closest to centroid.
        """
        mask = self.labels == cluster_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        cluster_features = self.features_fused[indices]
        centroid = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        n_return = min(n, len(indices))
        closest = np.argsort(distances)[:n_return]

        return [self.image_ids[indices[i]] for i in closest]

    def _find_boundary_ids(self, cluster_id: int, n: int = 10) -> list[str]:
        """Find the lowest-probability members of a cluster.

        These are the most ambiguous/boundary images.

        Args:
            cluster_id: Cluster to examine.
            n: Number of boundary IDs to return.

        Returns:
            List of image IDs with lowest membership probability.
        """
        from src.clustering.cluster_discovery import ClusterResult

        mask = self.labels == cluster_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        # Use distances from centroid as a fallback boundary measure
        cluster_features = self.features_fused[indices]
        centroid = cluster_features.mean(axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        n_return = min(n, len(indices))
        farthest = np.argsort(distances)[::-1][:n_return]

        return [self.image_ids[indices[i]] for i in farthest]

    def _compute_metadata_distribution(self, cluster_ids: list[str]) -> dict:
        """Compute distribution of metadata fields for a set of image IDs.

        Args:
            cluster_ids: Image IDs in this cluster.

        Returns:
            Dict mapping field name to {value: fraction}.
        """
        if self.metadata_df.empty or not cluster_ids:
            return {}

        # Find which column contains the image ID
        id_col = "image_id"
        if id_col not in self.metadata_df.columns:
            # Try patch_id or similar
            for candidate in ("patch_id", "id"):
                if candidate in self.metadata_df.columns:
                    id_col = candidate
                    break

        subset = self.metadata_df[self.metadata_df[id_col].isin(cluster_ids)]
        if subset.empty:
            return {}

        dist: dict[str, dict[str, float]] = {}
        metadata_fields = ["source_type", "altitude_group", "temporal_phase"]
        for field_name in metadata_fields:
            if field_name not in subset.columns:
                continue
            counts = subset[field_name].value_counts(normalize=True)
            dist[field_name] = {str(k): float(v) for k, v in counts.items()}

        return dist

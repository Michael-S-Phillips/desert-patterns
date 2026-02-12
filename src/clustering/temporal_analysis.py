"""Temporal analysis of pattern dynamics across rain phases.

Analyses cluster distributions, transition matrices, embedding drift,
and statistical tests for phase-dependent pattern changes.

All methods handle sparse temporal data gracefully (most ground images
have ``unknown`` phase), returning ``None`` with logged warnings when
insufficient data is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

_MIN_PHASE_SAMPLES = 5  # minimum samples per phase for statistical analysis

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TemporalResult:
    """Aggregated output of all temporal analyses."""

    phase_distributions: pd.DataFrame  # cluster × phase fractions
    transition_matrix: np.ndarray | None
    embedding_drift: dict | None
    test_results: pd.DataFrame  # chi-squared + per-cluster tests


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class TemporalPatternAnalysis:
    """Analyze pattern dynamics across temporal phases.

    Args:
        labels: Cluster labels ``(N,)`` from HDBSCAN (-1 = noise).
        image_ids: Image/patch identifiers ``(N,)``.
        metadata_df: DataFrame with at least ``image_id`` and
            ``temporal_phase`` columns; may also include ``lat``, ``lon``.
        features_2d: 2-D UMAP embeddings ``(N, 2)`` for drift analysis.
    """

    def __init__(
        self,
        labels: np.ndarray,
        image_ids: list[str],
        metadata_df: pd.DataFrame,
        features_2d: np.ndarray,
    ) -> None:
        self.labels = labels
        self.image_ids = image_ids
        self.metadata_df = metadata_df
        self.features_2d = features_2d

        # Build a lookup from image_id to index
        self._id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

        # Find the ID column in metadata
        self._id_col = "image_id"
        if self._id_col not in metadata_df.columns:
            for candidate in ("patch_id", "id"):
                if candidate in metadata_df.columns:
                    self._id_col = candidate
                    break

        # Merge labels into metadata for easier cross-tabs
        id_label_df = pd.DataFrame({
            "image_id": image_ids,
            "cluster": labels,
        })
        self._merged = metadata_df.merge(
            id_label_df,
            left_on=self._id_col,
            right_on="image_id",
            how="inner",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_distribution_by_phase(self) -> pd.DataFrame:
        """Compute cross-tabulation of cluster × temporal phase as fractions.

        Returns:
            DataFrame with clusters as rows, phases as columns; values
            are fractions summing to 1.0 per cluster.
        """
        if "temporal_phase" not in self._merged.columns:
            logger.warning("No temporal_phase column in metadata")
            return pd.DataFrame()

        # Exclude noise
        df = self._merged[self._merged["cluster"] >= 0]
        if df.empty:
            return pd.DataFrame()

        ct = pd.crosstab(df["cluster"], df["temporal_phase"], normalize="index")
        logger.info("Phase distribution computed: %d clusters × %d phases", *ct.shape)
        return ct

    def transition_matrix(self, distance_threshold_m: float = 50.0) -> np.ndarray | None:
        """Compute transition probabilities between clusters across phases.

        Matches images by GPS proximity across temporal phases and records
        the before → after cluster transition.

        Args:
            distance_threshold_m: Maximum distance (metres) to consider
                two images as co-located.

        Returns:
            Transition matrix as a row-normalized numpy array, or ``None``
            if fewer than 5 matched pairs are found.
        """
        if "temporal_phase" not in self._merged.columns:
            logger.warning("No temporal_phase column — cannot compute transition matrix")
            return None

        has_gps = (
            "lat" in self._merged.columns
            and "lon" in self._merged.columns
        )
        if not has_gps:
            logger.warning("No GPS coordinates — cannot compute transition matrix")
            return None

        df = self._merged[self._merged["cluster"] >= 0].copy()
        known_phases = ["pre_rain", "during_rain", "post_rain"]
        df = df[df["temporal_phase"].isin(known_phases)]

        if df.empty or df["temporal_phase"].nunique() < 2:
            logger.warning("Need ≥2 known temporal phases for transitions, found %d",
                           df["temporal_phase"].nunique())
            return None

        # Order phases
        phase_order = {p: i for i, p in enumerate(known_phases)}
        df = df.copy()
        df["phase_idx"] = df["temporal_phase"].map(phase_order)
        df = df.dropna(subset=["lat", "lon", "phase_idx"])

        # Find matched pairs: different phases, nearby GPS
        n_clusters = int(self.labels.max()) + 1 if self.labels.max() >= 0 else 0
        if n_clusters == 0:
            return None

        transitions: list[tuple[int, int]] = []
        phases = df["temporal_phase"].unique()

        for i, phase_a in enumerate(sorted(phases, key=lambda p: phase_order.get(p, 99))):
            for phase_b in sorted(phases, key=lambda p: phase_order.get(p, 99)):
                if phase_order.get(phase_a, 99) >= phase_order.get(phase_b, 99):
                    continue
                df_a = df[df["temporal_phase"] == phase_a]
                df_b = df[df["temporal_phase"] == phase_b]
                for _, row_a in df_a.iterrows():
                    for _, row_b in df_b.iterrows():
                        dist = _haversine_m(
                            row_a["lat"], row_a["lon"],
                            row_b["lat"], row_b["lon"],
                        )
                        if dist <= distance_threshold_m:
                            transitions.append((int(row_a["cluster"]), int(row_b["cluster"])))

        if len(transitions) < 5:
            logger.warning(
                "Only %d matched GPS pairs found (need ≥5) — skipping transition matrix",
                len(transitions),
            )
            return None

        # Build matrix
        matrix = np.zeros((n_clusters, n_clusters), dtype=np.float64)
        for c_from, c_to in transitions:
            if 0 <= c_from < n_clusters and 0 <= c_to < n_clusters:
                matrix[c_from, c_to] += 1

        # Row-normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        matrix = matrix / row_sums

        logger.info("Transition matrix: %d×%d from %d pairs", n_clusters, n_clusters,
                     len(transitions))
        return matrix

    def embedding_drift(self) -> dict | None:
        """Measure centroid shift per cluster between temporal phases.

        Returns:
            Dict mapping ``cluster_id`` → ``{phase_pair: drift_distance}``
            or ``None`` if phases are too sparse.
        """
        if "temporal_phase" not in self._merged.columns:
            return None

        df = self._merged[self._merged["cluster"] >= 0].copy()
        known_phases = ["pre_rain", "during_rain", "post_rain"]
        df = df[df["temporal_phase"].isin(known_phases)]

        phase_counts = df["temporal_phase"].value_counts()
        valid_phases = [p for p in known_phases if phase_counts.get(p, 0) >= _MIN_PHASE_SAMPLES]

        if len(valid_phases) < 2:
            logger.warning(
                "Embedding drift requires ≥2 phases with ≥%d samples; found %d valid phases",
                _MIN_PHASE_SAMPLES,
                len(valid_phases),
            )
            return None

        result: dict[int, dict[str, float]] = {}
        cluster_ids = sorted(set(df["cluster"]))

        for cid in cluster_ids:
            c_df = df[df["cluster"] == cid]
            drift_for_cluster: dict[str, float] = {}

            for i, phase_a in enumerate(valid_phases):
                for phase_b in valid_phases[i + 1:]:
                    ids_a = c_df[c_df["temporal_phase"] == phase_a][self._id_col].tolist()
                    ids_b = c_df[c_df["temporal_phase"] == phase_b][self._id_col].tolist()

                    idx_a = [self._id_to_idx[iid] for iid in ids_a if iid in self._id_to_idx]
                    idx_b = [self._id_to_idx[iid] for iid in ids_b if iid in self._id_to_idx]

                    if not idx_a or not idx_b:
                        continue

                    centroid_a = self.features_2d[idx_a].mean(axis=0)
                    centroid_b = self.features_2d[idx_b].mean(axis=0)
                    drift = float(np.linalg.norm(centroid_b - centroid_a))
                    pair_key = f"{phase_a}→{phase_b}"
                    drift_for_cluster[pair_key] = drift

            if drift_for_cluster:
                result[cid] = drift_for_cluster

        if not result:
            logger.warning("No clusters had sufficient data for embedding drift")
            return None

        logger.info("Embedding drift computed for %d clusters", len(result))
        return result

    def phase_statistical_tests(self) -> pd.DataFrame:
        """Run chi-squared test on cluster × phase contingency table.

        Also computes Cramer's V effect size and per-cluster proportion
        z-tests with Bonferroni correction.

        Returns:
            DataFrame with test results, or empty DataFrame if phases
            have fewer than ``_MIN_PHASE_SAMPLES`` samples.
        """
        if "temporal_phase" not in self._merged.columns:
            logger.warning("No temporal_phase column — skipping statistical tests")
            return pd.DataFrame()

        df = self._merged[self._merged["cluster"] >= 0].copy()
        known_phases = ["pre_rain", "during_rain", "post_rain"]
        df = df[df["temporal_phase"].isin(known_phases)]

        phase_counts = df["temporal_phase"].value_counts()
        valid_phases = [p for p in known_phases if phase_counts.get(p, 0) >= _MIN_PHASE_SAMPLES]

        if len(valid_phases) < 2:
            logger.warning(
                "Phase tests require ≥2 phases with ≥%d samples; found %d valid",
                _MIN_PHASE_SAMPLES,
                len(valid_phases),
            )
            return pd.DataFrame()

        df = df[df["temporal_phase"].isin(valid_phases)]

        # Contingency table
        ct = pd.crosstab(df["cluster"], df["temporal_phase"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            logger.warning("Contingency table too small for chi-squared: %s", ct.shape)
            return pd.DataFrame()

        chi2, p_val, dof, expected = chi2_contingency(ct)
        n_obs = ct.values.sum()
        k = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n_obs * k))) if n_obs > 0 and k > 0 else 0.0

        results = [{
            "test": "chi_squared_overall",
            "cluster": "all",
            "statistic": float(chi2),
            "p_value": float(p_val),
            "dof": int(dof),
            "effect_size": cramers_v,
            "effect_measure": "cramers_v",
        }]

        # Per-cluster proportion z-tests between phase pairs
        cluster_ids = sorted(ct.index)
        n_tests = 0
        pairwise_results: list[dict] = []

        for cid in cluster_ids:
            for i, phase_a in enumerate(valid_phases):
                for phase_b in valid_phases[i + 1:]:
                    if phase_a not in ct.columns or phase_b not in ct.columns:
                        continue
                    n_a = int(phase_counts.get(phase_a, 0))
                    n_b = int(phase_counts.get(phase_b, 0))
                    if n_a == 0 or n_b == 0:
                        continue

                    p_a = ct.loc[cid, phase_a] / n_a if cid in ct.index else 0
                    p_b = ct.loc[cid, phase_b] / n_b if cid in ct.index else 0
                    p_pool = (ct.loc[cid, phase_a] + ct.loc[cid, phase_b]) / (n_a + n_b)

                    # z-test
                    denom = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
                    if denom > 0:
                        z = (p_a - p_b) / denom
                        from scipy.stats import norm
                        p_z = float(2 * norm.sf(abs(z)))
                    else:
                        z = 0.0
                        p_z = 1.0

                    n_tests += 1
                    pairwise_results.append({
                        "test": "proportion_z",
                        "cluster": str(cid),
                        "phases": f"{phase_a}_vs_{phase_b}",
                        "statistic": float(z),
                        "p_value": float(p_z),
                        "p_adjusted": 0.0,  # filled below
                        "dof": 0,
                        "effect_size": float(abs(p_a - p_b)),
                        "effect_measure": "proportion_diff",
                    })

        # Bonferroni correction on pairwise tests
        for r in pairwise_results:
            r["p_adjusted"] = min(1.0, r["p_value"] * max(n_tests, 1))
        results.extend(pairwise_results)

        df_results = pd.DataFrame(results)
        logger.info(
            "Phase tests: chi2=%.2f (p=%.4f, V=%.3f), %d pairwise tests",
            chi2, p_val, cramers_v, n_tests,
        )
        return df_results

    def run_all(self) -> TemporalResult:
        """Orchestrate all temporal analyses.

        Returns:
            TemporalResult with all sub-analyses.
        """
        logger.info("=== Temporal Analysis ===")
        phase_dist = self.cluster_distribution_by_phase()
        trans = self.transition_matrix()
        drift = self.embedding_drift()
        tests = self.phase_statistical_tests()

        return TemporalResult(
            phase_distributions=phase_dist,
            transition_matrix=trans,
            embedding_drift=drift,
            test_results=tests,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two GPS points in metres."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

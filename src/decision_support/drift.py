"""Drift detection for monitoring input and prediction distribution changes."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift using PSI (Population Stability Index) and KS statistic."""

    def __init__(self, reference_data: Dict[str, Sequence[float]]):
        """Initialize with reference (baseline) data distributions.

        Args:
            reference_data: Dict mapping feature names to lists of reference values.
        """
        self.reference_data = {k: list(v) for k, v in reference_data.items()}
        self._reference_sorted = {k: sorted(v) for k, v in reference_data.items()}

    def compute_psi(
        self,
        feature_name: str,
        current_data: Sequence[float],
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index between reference and current distributions.

        Uses quantile-based binning from reference data so each bin has roughly
        equal reference counts, avoiding empty-bin artifacts.

        PSI < 0.1: no significant shift
        0.1 <= PSI < 0.25: moderate shift
        PSI >= 0.25: significant shift
        """
        if feature_name not in self.reference_data:
            raise KeyError(f"Feature '{feature_name}' not in reference data")
        if len(current_data) == 0:
            raise ValueError("Current data must not be empty")
        ref = self.reference_data[feature_name]
        if len(ref) == 0:
            raise ValueError("Reference data must not be empty")

        ref_sorted = sorted(ref)
        min_val = ref_sorted[0]
        max_val = ref_sorted[-1]
        if min_val == max_val:
            current_data_list = list(current_data)
            if all(abs(v - min_val) < 1e-12 for v in current_data_list):
                return 0.0
            return 1.0

        # Quantile-based bin edges from reference data
        n_bins = min(n_bins, len(ref))
        bin_edges = [min_val]
        for i in range(1, n_bins):
            idx = int(i * len(ref_sorted) / n_bins)
            idx = min(idx, len(ref_sorted) - 1)
            edge = ref_sorted[idx]
            if edge > bin_edges[-1]:
                bin_edges.append(edge)
        bin_edges.append(max_val + 1e-9)

        # Count in bins
        ref_counts = self._count_in_bins(ref, bin_edges)
        cur_counts = self._count_in_bins(list(current_data), bin_edges)

        # Normalize to proportions with Laplace smoothing
        ref_total = sum(ref_counts)
        cur_total = sum(cur_counts)
        if ref_total == 0 or cur_total == 0:
            return 1.0

        psi = 0.0
        for rc, cc in zip(ref_counts, cur_counts):
            rp = (rc + 0.5) / (ref_total + 0.5 * len(ref_counts))
            cp = (cc + 0.5) / (cur_total + 0.5 * len(cur_counts))
            psi += (cp - rp) * math.log(cp / rp)
        return max(0.0, psi)

    def compute_ks_statistic(
        self,
        feature_name: str,
        current_data: Sequence[float],
    ) -> float:
        """Compute Kolmogorov-Smirnov statistic between reference and current."""
        if feature_name not in self.reference_data:
            raise KeyError(f"Feature '{feature_name}' not in reference data")
        ref = self._reference_sorted[feature_name]
        cur = sorted(current_data)
        if len(cur) == 0:
            raise ValueError("Current data must not be empty")

        # Empirical CDFs
        all_vals = sorted(set(ref + cur))
        max_diff = 0.0
        n_ref = len(ref)
        n_cur = len(cur)
        for val in all_vals:
            ref_cdf = sum(1 for x in ref if x <= val) / n_ref
            cur_cdf = sum(1 for x in cur if x <= val) / n_cur
            diff = abs(ref_cdf - cur_cdf)
            if diff > max_diff:
                max_diff = diff
        return max_diff

    def detect_drift(
        self,
        current_data: Dict[str, Sequence[float]],
        psi_threshold: float = 0.2,
    ) -> Dict[str, bool]:
        """Detect drift for all features.

        Returns:
            Dict mapping feature names to boolean (True = drift detected).
        """
        results = {}
        for feature_name in self.reference_data:
            if feature_name not in current_data:
                logger.warning("Feature '%s' missing from current data, skipping", feature_name)
                results[feature_name] = False
                continue
            try:
                psi = self.compute_psi(feature_name, current_data[feature_name])
                results[feature_name] = psi >= psi_threshold
                if results[feature_name]:
                    logger.warning(
                        "Drift detected in '%s': PSI=%.4f (threshold=%.4f)",
                        feature_name,
                        psi,
                        psi_threshold,
                    )
            except (ValueError, KeyError) as e:
                logger.error("Error computing drift for '%s': %s", feature_name, e)
                results[feature_name] = False
        return results

    def _count_in_bins(self, data: Sequence[float], bin_edges: List[float]) -> List[int]:
        """Count data points in each bin defined by bin_edges."""
        n_bins = len(bin_edges) - 1
        counts = [0] * n_bins
        for val in data:
            for i in range(n_bins):
                if bin_edges[i] <= val < bin_edges[i + 1]:
                    counts[i] += 1
                    break
        return counts

    def get_psi_interpretation(self, psi: float) -> str:
        """Human-readable interpretation of a PSI value."""
        if psi < 0.1:
            return "no significant shift"
        if psi < 0.25:
            return "moderate shift"
        return "significant shift"

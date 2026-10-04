"""QTL (Quantitative Trait Locus) mapping via interval mapping.

Implements interval mapping for multi-locus QTL detection using
linear regression at marker positions and interpolated positions
between markers, with LOD score computation and confidence intervals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QTLResult:
    """Result for a single significant QTL."""

    chrom: str
    pos: float
    lod_score: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float


@dataclass
class QTLMappingResult:
    """Result of QTL mapping analysis."""

    significant_qtls: List[QTLResult]
    num_samples: int
    num_markers: int


class IntervalMapper:
    """Performs interval mapping for QTL detection.

    Uses linear regression to compute LOD scores at marker positions
    and at interpolated positions between markers. Identifies significant
    QTLs above a LOD threshold and computes confidence intervals.
    """

    def __init__(
        self,
        lod_threshold: float = 3.0,
        p_threshold: float = 0.05,
        interval_step: float = 1.0,
        ci_lod_drop: float = 1.5,
    ):
        """Initialize interval mapper.

        Args:
            lod_threshold: Minimum LOD score for significance.
            p_threshold: Maximum p-value for significance.
            interval_step: Step size (in cM or bp) for interval scanning.
            ci_lod_drop: LOD drop for confidence interval boundaries.
        """
        self.lod_threshold = lod_threshold
        self.p_threshold = p_threshold
        self.interval_step = interval_step
        self.ci_lod_drop = ci_lod_drop

    def map_qtl(
        self,
        markers: Sequence[Tuple[str, float, Sequence[float]]],
        phenotypes: Sequence[float],
    ) -> QTLMappingResult:
        """Run interval mapping QTL analysis.

        Args:
            markers: List of (chrom, pos, genotypes) tuples.
                Genotypes are dosages: 0=AA, 1=Aa, 2=aa.
            phenotypes: Phenotype values for each sample.

        Returns:
            QTLMappingResult with significant QTLs.
        """
        if not markers:
            raise ValueError("Marker data cannot be empty")
        if not phenotypes:
            raise ValueError("Phenotype data cannot be empty")

        n = len(phenotypes)
        for chrom, pos, genotypes in markers:
            if len(genotypes) != n:
                raise ValueError(
                    f"Marker ({chrom}:{pos}) has {len(genotypes)} genotypes "
                    f"but {n} phenotypes were provided — same length required"
                )

        # Group markers by chromosome
        chrom_markers: dict[str, List[Tuple[float, List[float]]]] = {}
        for chrom, pos, genotypes in markers:
            chrom_markers.setdefault(chrom, []).append((pos, list(genotypes)))

        # Sort markers by position within each chromosome
        for chrom in chrom_markers:
            chrom_markers[chrom].sort(key=lambda x: x[0])

        # Compute LOD scores at all positions (markers + intervals)
        all_qtls: List[QTLResult] = []

        for chrom, marker_list in chrom_markers.items():
            # Skip chromosome with only one marker (no intervals)
            if len(marker_list) < 1:
                continue

            # Compute LOD at marker positions and interpolated positions
            lod_scores = self._compute_lod_profile(chrom, marker_list, phenotypes)

            # Find peaks above threshold
            peaks = self._find_peaks(lod_scores, marker_list)

            for peak_pos, peak_lod, peak_effect in peaks:
                # Compute confidence interval
                ci_lower, ci_upper = self._compute_ci(peak_pos, lod_scores, marker_list, peak_lod)

                # Compute p-value from LOD score
                p_value = self._lod_to_pvalue(peak_lod, n)

                if p_value < self.p_threshold:
                    all_qtls.append(
                        QTLResult(
                            chrom=chrom,
                            pos=peak_pos,
                            lod_score=peak_lod,
                            p_value=p_value,
                            effect_size=peak_effect,
                            ci_lower=ci_lower,
                            ci_upper=ci_upper,
                        )
                    )

        # Sort by LOD score descending
        all_qtls.sort(key=lambda q: q.lod_score, reverse=True)

        return QTLMappingResult(
            significant_qtls=all_qtls,
            num_samples=n,
            num_markers=len(markers),
        )

    def _compute_lod_profile(
        self,
        chrom: str,
        marker_list: List[Tuple[float, List[float]]],
        phenotypes: Sequence[float],
    ) -> List[Tuple[float, float, float]]:
        """Compute LOD scores at marker and interpolated positions.

        Returns list of (position, lod_score, effect_size) tuples.
        """
        n = len(phenotypes)
        lod_scores: List[Tuple[float, float, float]] = []

        # Compute LOD at each marker position
        for pos, genotypes in marker_list:
            lod, effect = self._compute_lod_at_position(genotypes, phenotypes, n)
            lod_scores.append((pos, lod, effect))

        # Compute LOD at interpolated positions between markers
        for i in range(len(marker_list) - 1):
            pos1, geno1 = marker_list[i]
            pos2, geno2 = marker_list[i + 1]

            if pos2 <= pos1:
                continue

            # Scan at regular intervals between markers
            step = self.interval_step
            if step <= 0:
                step = (pos2 - pos1) / 10.0

            num_steps = max(1, int((pos2 - pos1) / step))
            actual_step = (pos2 - pos1) / num_steps

            for s in range(1, num_steps):
                interp_pos = pos1 + s * actual_step
                # Interpolate genotypes (weighted average)
                weight = s / num_steps
                interp_geno = [g1 * (1 - weight) + g2 * weight for g1, g2 in zip(geno1, geno2)]
                lod, effect = self._compute_lod_at_position(interp_geno, phenotypes, n)
                lod_scores.append((interp_pos, lod, effect))

        # Sort by position
        lod_scores.sort(key=lambda x: x[0])
        return lod_scores

    def _compute_lod_at_position(
        self,
        genotypes: Sequence[float],
        phenotypes: Sequence[float],
        n: int,
    ) -> Tuple[float, float]:
        """Compute LOD score at a single position using linear regression.

        LOD = (n/2) * log10(RSS0 / RSS1)

        Returns:
            (lod_score, effect_size)
        """
        # Check for no variation in genotypes
        geno_set = set(genotypes)
        if len(geno_set) <= 1:
            return 0.0, 0.0

        # Null model: phenotype ~ 1 (intercept only)
        mean_y = sum(phenotypes) / n
        rss0 = sum((y - mean_y) ** 2 for y in phenotypes)

        if rss0 == 0:
            return 0.0, 0.0

        # Alternative model: phenotype ~ genotype dosage
        mean_x = sum(genotypes) / n
        var_x = sum((x - mean_x) ** 2 for x in genotypes)

        if var_x == 0:
            return 0.0, 0.0

        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(genotypes, phenotypes))
        beta = cov_xy / var_x
        alpha = mean_y - beta * mean_x

        rss1 = sum((y - (alpha + beta * x)) ** 2 for x, y in zip(genotypes, phenotypes))

        if rss1 <= 0:
            # Perfect fit
            return float("inf"), beta

        # LOD score
        lod = (n / 2.0) * math.log10(rss0 / rss1)

        return max(0.0, lod), beta

    def _find_peaks(
        self,
        lod_scores: List[Tuple[float, float, float]],
        marker_list: List[Tuple[float, List[float]]],
    ) -> List[Tuple[float, float, float]]:
        """Find LOD peaks above threshold.

        Returns list of (position, lod_score, effect_size) for each peak.
        """
        if not lod_scores:
            return []

        peaks = []
        in_peak = False
        peak_start_idx = 0

        for i, (pos, lod, effect) in enumerate(lod_scores):
            if lod >= self.lod_threshold:
                if not in_peak:
                    in_peak = True
                    peak_start_idx = i
            else:
                if in_peak:
                    # Find the maximum within the peak region
                    peak_region = lod_scores[peak_start_idx:i]
                    if peak_region:
                        max_point = max(peak_region, key=lambda x: x[1])
                        peaks.append(max_point)
                    in_peak = False

        # Handle peak at the end
        if in_peak:
            peak_region = lod_scores[peak_start_idx:]
            if peak_region:
                max_point = max(peak_region, key=lambda x: x[1])
                peaks.append(max_point)

        return peaks

    def _compute_ci(
        self,
        peak_pos: float,
        lod_scores: List[Tuple[float, float, float]],
        marker_list: List[Tuple[float, List[float]]],
        peak_lod: float,
    ) -> Tuple[float, float]:
        """Compute confidence interval using LOD drop method.

        CI boundaries are where LOD drops by ci_lod_drop from peak.
        """
        ci_lod = peak_lod - self.ci_lod_drop

        # Find lower boundary
        ci_lower = marker_list[0][0]  # Default to first marker
        for pos, lod, _ in lod_scores:
            if pos >= peak_pos:
                break
            if lod >= ci_lod:
                ci_lower = pos

        # Find upper boundary
        ci_upper = marker_list[-1][0]  # Default to last marker
        for pos, lod, _ in reversed(lod_scores):
            if pos <= peak_pos:
                break
            if lod >= ci_lod:
                ci_upper = pos

        return ci_lower, ci_upper

    @staticmethod
    def _lod_to_pvalue(lod: float, n: int) -> float:
        """Convert LOD score to approximate p-value.

        Uses the approximation: p ≈ n * 10^(-LOD) for large n.
        """
        if lod <= 0:
            return 1.0
        # Approximate p-value from LOD score
        # For a chi-square with 1 df: LOD = chi2 / (2 * ln(10))
        chi2 = lod * 2.0 * math.log(10)
        # Approximate p-value from chi-square with 1 df
        # p = erfc(sqrt(chi2/2))
        p = math.erfc(math.sqrt(chi2 / 2.0))
        return min(1.0, max(0.0, p))


def run_qtl_mapping(
    markers: Sequence[Tuple[str, float, Sequence[float]]],
    phenotypes: Sequence[float],
    lod_threshold: float = 3.0,
    p_threshold: float = 0.05,
) -> QTLMappingResult:
    """Convenience function for QTL mapping.

    Args:
        markers: List of (chrom, pos, genotypes) tuples.
        phenotypes: Phenotype values for each sample.
        lod_threshold: Minimum LOD score for significance.
        p_threshold: Maximum p-value for significance.

    Returns:
        QTLMappingResult with significant QTLs.
    """
    mapper = IntervalMapper(lod_threshold=lod_threshold, p_threshold=p_threshold)
    return mapper.map_qtl(markers, phenotypes)

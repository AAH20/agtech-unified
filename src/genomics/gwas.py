"""Genome-Wide Association Study (GWAS) support.

Implements a simple linear regression-based GWAS for associating
genotypes with phenotypes, with p-values, effect sizes, and
confidence intervals.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SNPResult:
    """Result for a single SNP association test."""

    chrom: str
    pos: int
    ref: str
    alt: str
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    odds_ratio: float


@dataclass
class GWASResult:
    """Result of GWAS analysis."""

    significant_snps: List[SNPResult]
    manhattan_data: List[dict]
    num_samples: int


class GWASAnalyzer:
    """Performs genome-wide association studies.

    Uses linear regression to test association between each SNP
    and a continuous phenotype.
    """

    def __init__(self, significance_threshold: float = 1e-5):
        self.significance_threshold = significance_threshold

    def analyze(
        self,
        genotypes: Sequence[str],
        phenotypes: Sequence[float],
    ) -> GWASResult:
        """Run GWAS analysis.

        Args:
            genotypes: List of genotype strings (e.g., "A/A", "A/T", "T/T").
            phenotypes: List of phenotype values.

        Returns:
            GWASResult with significant SNPs and Manhattan plot data.
        """
        if not genotypes:
            raise ValueError("Genotype data cannot be empty")
        if not phenotypes:
            raise ValueError("Phenotype data cannot be empty")
        if len(genotypes) != len(phenotypes):
            raise ValueError(
                f"Genotype and phenotype data must have same length: "
                f"{len(genotypes)} vs {len(phenotypes)}"
            )

        n = len(genotypes)
        snp_results = []

        # Parse unique SNPs from genotype strings
        snp_info = self._parse_genotypes(genotypes)

        for chrom, pos, ref, alt, dosage in snp_info:
            # Linear regression: phenotype ~ dosage
            p_value, effect_size, ci_lower, ci_upper, odds_ratio = self._linear_regression(
                dosage, phenotypes
            )

            snp_results.append(
                SNPResult(
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    p_value=p_value,
                    effect_size=effect_size,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    odds_ratio=odds_ratio,
                )
            )

        significant = [s for s in snp_results if s.p_value < self.significance_threshold]

        manhattan_data = [
            {
                "chrom": s.chrom,
                "pos": s.pos,
                "p_value": s.p_value,
                "neg_log_p": -math.log10(max(s.p_value, 1e-300)),
            }
            for s in snp_results
        ]

        return GWASResult(
            significant_snps=significant,
            manhattan_data=manhattan_data,
            num_samples=n,
        )

    def _parse_genotypes(
        self,
        genotypes: Sequence[str],
    ) -> List[Tuple[str, int, str, str, List[float]]]:
        """Parse genotype strings into SNP info.

        Returns list of (chrom, pos, ref, alt, dosage) tuples.
        Dosage is the count of alt alleles (0, 1, or 2).
        """
        if not genotypes:
            return []

        # Parse first genotype to get SNP info
        first = genotypes[0]
        parts = first.split(":")
        if len(parts) >= 4:
            chrom, pos_str, ref, alt = parts[0], parts[1], parts[2], parts[3]
            pos = int(pos_str)
        else:
            # Default: single SNP on chr1
            chrom, pos, ref, alt = "1", 1, "A", "T"

        # Compute dosage for each sample
        dosage = []
        for gt in genotypes:
            gt = gt.upper().replace("|", "/")
            alleles = gt.split("/")
            if len(alleles) != 2:
                dosage.append(0.0)
                continue
            alt_count = sum(1 for a in alleles if a == alt.upper())
            dosage.append(float(alt_count))

        return [(chrom, pos, ref, alt, dosage)]

    def _linear_regression(
        self,
        x: Sequence[float],
        y: Sequence[float],
    ) -> Tuple[float, float, float, float, float]:
        """Simple linear regression y = beta * x + alpha.

        Returns:
            (p_value, effect_size, ci_lower, ci_upper, odds_ratio)
        """
        n = len(x)
        if n < 3:
            return 1.0, 0.0, 0.0, 0.0, 1.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Compute beta (slope)
        cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)

        if var_x == 0:
            return 1.0, 0.0, 0.0, 0.0, 1.0

        beta = cov_xy / var_x
        alpha = mean_y - beta * mean_x

        # Residuals
        residuals = [yi - (alpha + beta * xi) for xi, yi in zip(x, y)]
        rss = sum(r**2 for r in residuals)

        # Standard error of beta
        df = n - 2
        if df <= 0:
            return 1.0, beta, beta, beta, 1.0
        if rss == 0:
            # Perfect fit — infinitely significant
            return 0.0, beta, beta, beta, 1.0

        mse = rss / df
        se_beta = math.sqrt(mse / var_x)

        # t-statistic and p-value
        t_stat = beta / se_beta if se_beta > 0 else 0.0
        p_value = self._t_distribution_p_value(abs(t_stat), df)

        # Confidence interval for beta (95%)
        t_crit = self._t_critical_value(df, 0.975)
        ci_lower = beta - t_crit * se_beta
        ci_upper = beta + t_crit * se_beta

        # Odds ratio approximation: exp(beta)
        odds_ratio = math.exp(beta) if abs(beta) < 50 else float("inf")
        if odds_ratio == float("inf"):
            odds_ratio = 1e10

        return p_value, beta, ci_lower, ci_upper, odds_ratio

    @staticmethod
    def _t_distribution_p_value(t: float, df: int) -> float:
        """Approximate two-tailed p-value from t-statistic.

        Uses the regularized incomplete beta function approximation.
        """
        if df <= 0:
            return 1.0

        # Use normal approximation for large df
        if df > 30:
            # Two-tailed normal p-value
            return math.erfc(t / math.sqrt(2))

        # For small df, use a simple approximation
        # This is a rough approximation sufficient for GWAS screening
        x = df / (df + t * t)
        # Incomplete beta approximation
        p = _betai(df / 2.0, 0.5, x)
        return min(1.0, max(0.0, p))

    @staticmethod
    def _t_critical_value(df: int, confidence: float) -> float:
        """Approximate critical t-value.

        Uses a simple approximation for the t-distribution quantile.
        """
        if df <= 0:
            return 1.96

        # For large df, use normal approximation
        if df > 30:
            # Inverse normal CDF approximation
            p = confidence
            # Rational approximation
            if p < 0.5:
                t = -math.sqrt(-2 * math.log(1 - p))
            else:
                t = math.sqrt(-2 * math.log(1 - p))
            # Refine
            return t + (t**3 + t) / (4 * df)

        # For small df, use a lookup table approximation
        # Common critical values for 95% CI
        t_table = {
            1: 12.71,
            2: 4.30,
            3: 3.18,
            4: 2.78,
            5: 2.57,
            6: 2.45,
            7: 2.36,
            8: 2.31,
            9: 2.26,
            10: 2.23,
            11: 2.20,
            12: 2.18,
            13: 2.16,
            14: 2.14,
            15: 2.13,
            16: 2.12,
            17: 2.11,
            18: 2.10,
            19: 2.09,
            20: 2.09,
            21: 2.08,
            22: 2.07,
            23: 2.07,
            24: 2.06,
            25: 2.06,
            26: 2.06,
            27: 2.05,
            28: 2.05,
            29: 2.04,
            30: 2.04,
        }
        return t_table.get(df, 2.0)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function approximation.

    Uses continued fraction representation (Lentz's method).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Log of beta function
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    # Front factor
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)

    # Continued fraction
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    else:
        return 1.0 - front * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function.

    Uses Lentz's method.
    """
    max_iter = 200
    eps = 3e-7
    fpmin = 1e-30

    qab = a + b
    qap = a + 1
    qam = a - 1

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def run_gwas(
    genotypes: Sequence[str],
    phenotypes: Sequence[float],
    significance_threshold: float = 1e-5,
) -> GWASResult:
    """Convenience function for GWAS analysis.

    Args:
        genotypes: List of genotype strings.
        phenotypes: List of phenotype values.
        significance_threshold: P-value threshold for significance.

    Returns:
        GWASResult with significant SNPs and Manhattan plot data.
    """
    analyzer = GWASAnalyzer(significance_threshold=significance_threshold)
    return analyzer.analyze(genotypes, phenotypes)

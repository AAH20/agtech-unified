"""Confidence intervals for prediction scores.

Provides Wilson score intervals for binary classification scores,
bootstrap confidence intervals for arbitrary prediction scores,
and a convenience function to attach CIs to any prediction.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """A confidence interval for a prediction score."""

    lower: float
    upper: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Check if a value falls within the interval."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}] ({self.confidence:.0%} CI)"


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Calculate Wilson score confidence interval.

    The Wilson interval is a robust method for binomial proportions,
    especially for small samples or extreme proportions.

    Args:
        k: Number of successes.
        n: Total number of trials.
        confidence: Confidence level (default 0.95).

    Returns:
        ConfidenceInterval with lower and upper bounds.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k cannot exceed n")

    if n == 0:
        return ConfidenceInterval(lower=0.0, upper=0.0, confidence=confidence)

    p_hat = k / n

    # Z-score for the confidence level
    z = _z_score(confidence)

    z2 = z * z
    denominator = 1.0 + z2 / n

    # Center of the interval
    center = (p_hat + z2 / (2 * n)) / denominator

    # Margin
    margin = z * ((p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) ** 0.5) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return ConfidenceInterval(lower=lower, upper=upper, confidence=confidence)


def bootstrap_confidence_interval(
    data: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Calculate bootstrap confidence interval for a sample.

    Uses the percentile method: resample the data with replacement,
    compute the mean for each resample, and take the percentiles.

    Args:
        data: Sample data points.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95).

    Returns:
        ConfidenceInterval with lower and upper bounds.
    """
    if not data:
        raise ValueError("Data cannot be empty")

    n = len(data)
    if n == 1:
        val = float(data[0])
        return ConfidenceInterval(lower=val, upper=val, confidence=confidence)

    means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()

    alpha = 1.0 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return ConfidenceInterval(
        lower=means[lower_idx],
        upper=means[upper_idx],
        confidence=confidence,
    )


def add_confidence_interval(
    score: float,
    n: int,
    k: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Attach a Wilson confidence interval to a prediction score.

    Args:
        score: The point estimate (e.g., accuracy, precision).
        n: Total number of samples.
        k: Number of positive predictions.
        confidence: Confidence level (default 0.95).

    Returns:
        ConfidenceInterval with score stored as a dynamic attribute.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    ci = wilson_score_interval(k, n, confidence=confidence)
    # Attach the point estimate to the CI for convenience
    ci.score = score  # type: ignore[attr-defined]
    return ci


def _z_score(confidence: float) -> float:
    """Return the two-tailed z-score for a given confidence level.

    Uses the inverse of the standard normal CDF (rational approximation).

    Args:
        confidence: Confidence level between 0 and 1.

    Returns:
        Z-score (e.g., 1.96 for 95% confidence).
    """
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("Confidence must be between 0 and 1")

    # Rational approximation of the inverse normal CDF
    # Based on Abramowitz and Stegun formula 26.2.23
    alpha = 1.0 - confidence
    p = alpha / 2.0

    if p == 0.5:
        return 0.0

    # Use symmetry
    if p > 0.5:
        p = 1.0 - p

    t = (-2.0 * _log(p)) ** 0.5

    # Coefficients for the rational approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    return z


def _log(x: float) -> float:
    """Natural logarithm wrapper."""
    import math

    return math.log(x)

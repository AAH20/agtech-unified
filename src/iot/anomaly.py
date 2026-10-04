"""IoT anomaly detection: IQR, EWMA, and Welford's algorithm.

Three complementary approaches for detecting anomalies in sensor data:
- IQRDetector: Tukey's fences on a rolling window (robust to outliers)
- EWMADetector: Exponentially weighted moving average (adapts to level shifts)
- WelfordAccumulator: Incremental mean/stddev (numerically stable)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict


class IQRDetector:
    """IQR-based anomaly detection using Tukey's fences.

    Maintains a rolling window of recent values. A new value is anomalous
    when it falls outside [Q1 - k*IQR, Q3 + k*IQR].

    Args:
        window: Size of the rolling window (must be >= 2).
        k: IQR multiplier for fence width (must be positive).
    """

    def __init__(self, window: int = 20, k: float = 1.5) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if k <= 0:
            raise ValueError("k must be positive")
        self.window = window
        self.k = k
        self._values: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> "IQRDetector":
        """Add a value to the rolling window."""
        self._values.append(float(value))
        return self

    def is_anomalous(self, value: float) -> bool:
        """True when value falls outside the IQR fences."""
        if len(self._values) < 2:
            return False
        q1, q3 = self._quartiles()
        iqr = q3 - q1
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        return value < lower or value > upper

    def get_stats(self) -> Dict[str, float]:
        """Return current quartiles and fences."""
        if len(self._values) < 2:
            return {"q1": 0.0, "q3": 0.0, "iqr": 0.0, "lower_fence": 0.0, "upper_fence": 0.0}
        q1, q3 = self._quartiles()
        iqr = q3 - q1
        return {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": q1 - self.k * iqr,
            "upper_fence": q3 + self.k * iqr,
        }

    def _quartiles(self) -> tuple[float, float]:
        """Compute Q1 and Q3 using linear interpolation."""
        sorted_vals = sorted(self._values)
        q1 = self._percentile(sorted_vals, 0.25)
        q3 = self._percentile(sorted_vals, 0.75)
        return q1, q3

    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        """Linear interpolation percentile (same as numpy default)."""
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_vals[lo]
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class EWMADetector:
    """EWMA-based anomaly detection with exponential weighting.

    Tracks an exponentially weighted mean and variance. A new value is
    anomalous when it deviates from the EWMA mean by more than `threshold`
    standard deviations.

    Args:
        alpha: Smoothing factor in (0, 1]. Higher = faster adaptation.
        threshold: Number of standard deviations for anomaly (must be positive).
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 3.0) -> None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.alpha = alpha
        self.threshold = threshold
        self._mean: float = 0.0
        self._variance: float = 0.0
        self._initialized: bool = False

    def update(self, value: float) -> "EWMADetector":
        """Update the EWMA statistics with a new value."""
        value = float(value)
        if not self._initialized:
            self._mean = value
            self._variance = 0.0
            self._initialized = True
            return self
        diff = value - self._mean
        self._mean += self.alpha * diff
        self._variance = (1 - self.alpha) * (self._variance + self.alpha * diff * diff)
        return self

    def is_anomalous(self, value: float) -> bool:
        """True when value deviates beyond threshold standard deviations."""
        if not self._initialized:
            return False
        stddev = math.sqrt(self._variance)
        if stddev == 0.0:
            return value != self._mean
        z_score = abs(value - self._mean) / stddev
        return z_score > self.threshold

    def get_stats(self) -> Dict[str, float]:
        """Return current EWMA statistics."""
        return {
            "mean": self._mean,
            "stddev": math.sqrt(self._variance),
            "threshold": self.threshold,
        }


class WelfordAccumulator:
    """Welford's online algorithm for incremental mean and stddev.

    Numerically stable — avoids catastrophic cancellation that plagues
    the naive sum-of-squares approach. Supports merging accumulators.

    Usage:
        acc = WelfordAccumulator()
        for value in stream:
            acc.add(value)
        print(acc.mean, acc.stddev)
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    @property
    def count(self) -> int:
        """Number of values added."""
        return self._count

    @property
    def mean(self) -> float:
        """Running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """Population variance (0.0 if count == 0)."""
        if self._count == 0:
            return 0.0
        return self._m2 / self._count

    @property
    def stddev(self) -> float:
        """Population standard deviation."""
        return math.sqrt(self.variance)

    def add(self, value: float) -> "WelfordAccumulator":
        """Add a value and update statistics."""
        value = float(value)
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        return self

    def merge(self, other: "WelfordAccumulator") -> None:
        """Merge another WelfordAccumulator into this one."""
        if other._count == 0:
            return
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean
            self._m2 = other._m2
            return
        total = self._count + other._count
        delta = other._mean - self._mean
        self._mean = (self._count * self._mean + other._count * other._mean) / total
        self._m2 += other._m2 + delta * delta * self._count * other._count / total
        self._count = total

    def reset(self) -> None:
        """Clear all state."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

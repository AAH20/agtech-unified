"""Time-series forecasting for agricultural sensor data.

Provides simple exponential smoothing for soil moisture and temperature prediction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Sequence

logger = logging.getLogger(__name__)

__all__ = ["ExponentialSmoother", "ForecastResult"]


@dataclass
class ForecastResult:
    """Result of a time-series forecast."""

    values: List[float]
    lower_bound: float
    upper_bound: float
    alpha: float
    steps: int
    method: str = "exponential_smoothing"
    smoothed_values: List[float] = field(default_factory=list)


class ExponentialSmoother:
    """Simple exponential smoothing forecaster.

    Uses the formula: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    where s_t is the smoothed value at time t and x_t is the observation.

    Forecast for all future steps equals the last smoothed value.
    Confidence intervals widen with the forecast horizon.
    """

    def __init__(self, alpha: float = 0.3):
        """Initialize the smoother.

        Args:
            alpha: Smoothing factor in [0, 1]. Higher values weight recent
                observations more heavily.

        Raises:
            ValueError: If alpha is outside [0, 1].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.smoothed_values: List[float] = []

    def _validate_series(self, series: Sequence[float]) -> List[float]:
        """Validate and convert the input series."""
        if len(series) == 0:
            raise ValueError("Cannot forecast on empty series")
        values = []
        for v in series:
            if not isinstance(v, (int, float)):
                raise TypeError(f"Series values must be numeric, got {type(v).__name__}")
            values.append(float(v))
        return values

    def _smooth(self, values: List[float]) -> List[float]:
        """Apply exponential smoothing to the series."""
        if not values:
            return []
        smoothed = [values[0]]
        for i in range(1, len(values)):
            s = self.alpha * values[i] + (1.0 - self.alpha) * smoothed[i - 1]
            smoothed.append(s)
        return smoothed

    def _compute_std_residuals(self, values: List[float], smoothed: List[float]) -> float:
        """Compute standard deviation of residuals."""
        if len(values) < 2:
            return 0.0
        residuals = [v - s for v, s in zip(values, smoothed)]
        mean_res = sum(residuals) / len(residuals)
        variance = sum((r - mean_res) ** 2 for r in residuals) / (len(residuals) - 1)
        return math.sqrt(max(0.0, variance))

    def forecast(self, series: Sequence[float], steps: int = 1) -> ForecastResult:
        """Forecast future values using simple exponential smoothing.

        Args:
            series: Historical observations (chronological order).
            steps: Number of steps ahead to forecast.

        Returns:
            ForecastResult with predicted values and confidence intervals.

        Raises:
            ValueError: If series is empty or steps < 1.
            TypeError: If series contains non-numeric values.
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        values = self._validate_series(series)
        smoothed = self._smooth(values)
        self.smoothed_values = smoothed

        # Forecast value is the last smoothed value (flat forecast)
        forecast_value = smoothed[-1]

        # Compute confidence interval based on residual std and horizon
        std_residuals = self._compute_std_residuals(values, smoothed)

        # Confidence interval widens with sqrt(steps) for random walk assumption
        # Use 95% confidence (1.96 * std)
        z_score = 1.96
        margin = z_score * std_residuals * math.sqrt(steps)

        # For very short series, add a minimum margin based on data range
        if len(values) < 3:
            data_range = max(values) - min(values)
            margin = max(margin, 0.1 * data_range * math.sqrt(steps))

        lower = forecast_value - margin
        upper = forecast_value + margin

        forecast_values = [forecast_value] * steps

        logger.debug(
            "Forecast: alpha=%s, steps=%s, value=%.4f, CI=[%.4f, %.4f]",
            self.alpha,
            steps,
            forecast_value,
            lower,
            upper,
        )

        return ForecastResult(
            values=forecast_values,
            lower_bound=lower,
            upper_bound=upper,
            alpha=self.alpha,
            steps=steps,
            method="exponential_smoothing",
            smoothed_values=smoothed,
        )

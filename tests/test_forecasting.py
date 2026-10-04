"""Tests for time-series forecasting with exponential smoothing."""

import pytest

from src.decision_support.forecasting import ExponentialSmoother, ForecastResult


class TestExponentialSmoother:
    """Test simple exponential smoothing forecaster."""

    def test_basic_forecast(self):
        """Basic forecast returns a reasonable value."""
        series = [0.5, 0.55, 0.6, 0.58, 0.62, 0.65, 0.63, 0.67]
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast(series, steps=1)
        assert isinstance(result, ForecastResult)
        assert len(result.values) == 1
        assert 0.0 <= result.values[0] <= 1.0

    def test_forecast_multiple_steps(self):
        """Forecast multiple steps ahead."""
        series = [0.5, 0.55, 0.6, 0.58, 0.62, 0.65, 0.63, 0.67]
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast(series, steps=3)
        assert len(result.values) == 3
        # All values should be the same for simple exponential smoothing
        assert result.values[0] == result.values[1] == result.values[2]

    def test_constant_series(self):
        """Constant series should forecast the same constant."""
        series = [0.5, 0.5, 0.5, 0.5, 0.5]
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast(series, steps=1)
        assert abs(result.values[0] - 0.5) < 0.01

    def test_increasing_trend(self):
        """Increasing series should forecast higher than the mean."""
        series = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        smoother = ExponentialSmoother(alpha=0.5)
        result = smoother.forecast(series, steps=1)
        mean_val = sum(series) / len(series)
        assert result.values[0] > mean_val

    def test_decreasing_trend(self):
        """Decreasing series should forecast lower than the mean."""
        series = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        smoother = ExponentialSmoother(alpha=0.5)
        result = smoother.forecast(series, steps=1)
        mean_val = sum(series) / len(series)
        assert result.values[0] < mean_val

    def test_alpha_zero_uses_first_value(self):
        """Alpha=0 should use the initial value (first observation)."""
        series = [0.3, 0.5, 0.7, 0.9]
        smoother = ExponentialSmoother(alpha=0.0)
        result = smoother.forecast(series, steps=1)
        assert abs(result.values[0] - 0.3) < 0.01

    def test_alpha_one_uses_last_value(self):
        """Alpha=1 should use the last observation."""
        series = [0.3, 0.5, 0.7, 0.9]
        smoother = ExponentialSmoother(alpha=1.0)
        result = smoother.forecast(series, steps=1)
        assert abs(result.values[0] - 0.9) < 0.01

    def test_confidence_intervals(self):
        """Forecast includes confidence intervals."""
        series = [0.5, 0.55, 0.6, 0.58, 0.62, 0.65, 0.63, 0.67]
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast(series, steps=1)
        assert result.lower_bound <= result.values[0] <= result.upper_bound

    def test_confidence_interval_widens_with_steps(self):
        """Confidence intervals should widen for longer horizons."""
        series = [0.5, 0.55, 0.6, 0.58, 0.62, 0.65, 0.63, 0.67]
        smoother = ExponentialSmoother(alpha=0.3)
        result_1 = smoother.forecast(series, steps=1)
        result_5 = smoother.forecast(series, steps=5)
        width_1 = result_1.upper_bound - result_1.lower_bound
        width_5 = result_5.upper_bound - result_5.lower_bound
        assert width_5 >= width_1

    def test_empty_series_raises(self):
        """Empty series should raise ValueError."""
        smoother = ExponentialSmoother(alpha=0.3)
        with pytest.raises(ValueError, match="empty"):
            smoother.forecast([], steps=1)

    def test_single_value_series(self):
        """Single-value series should forecast that value."""
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast([0.42], steps=1)
        assert abs(result.values[0] - 0.42) < 0.01

    def test_invalid_alpha_raises(self):
        """Alpha outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            ExponentialSmoother(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha"):
            ExponentialSmoother(alpha=1.5)

    def test_non_numeric_raises(self):
        """Non-numeric values should raise TypeError."""
        smoother = ExponentialSmoother(alpha=0.3)
        with pytest.raises(TypeError):
            smoother.forecast(["a", "b", "c"], steps=1)

    def test_soil_moisture_forecast(self):
        """Forecast soil moisture values."""
        series = [0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08]
        smoother = ExponentialSmoother(alpha=0.4)
        result = smoother.forecast(series, steps=1)
        # Should forecast decreasing moisture
        assert result.values[0] < series[0]
        assert result.values[0] > 0.0

    def test_temperature_forecast(self):
        """Forecast temperature values."""
        series = [20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0]
        smoother = ExponentialSmoother(alpha=0.4)
        result = smoother.forecast(series, steps=1)
        # Should forecast increasing temperature
        assert result.values[0] > series[0]

    def test_forecast_result_metadata(self):
        """ForecastResult includes metadata."""
        series = [0.5, 0.55, 0.6, 0.58, 0.62]
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast(series, steps=1)
        assert result.alpha == 0.3
        assert result.steps == 1
        assert result.method == "exponential_smoothing"

    def test_smoothed_values_accessible(self):
        """Smoothed values are accessible after forecast."""
        series = [0.5, 0.55, 0.6, 0.58, 0.62]
        smoother = ExponentialSmoother(alpha=0.3)
        smoother.forecast(series, steps=1)
        assert len(smoother.smoothed_values) == len(series)

    def test_longer_series_more_stable(self):
        """Longer series should produce more stable forecasts."""
        short = [0.5, 0.8]
        long = [0.5, 0.55, 0.6, 0.58, 0.62, 0.65, 0.63, 0.67, 0.64, 0.66]
        smoother_short = ExponentialSmoother(alpha=0.5)
        smoother_long = ExponentialSmoother(alpha=0.5)
        result_short = smoother_short.forecast(short, steps=1)
        result_long = smoother_long.forecast(long, steps=1)
        # Longer series should have tighter confidence intervals
        width_short = result_short.upper_bound - result_short.lower_bound
        width_long = result_long.upper_bound - result_long.lower_bound
        assert width_long < width_short

    def test_forecast_with_gaps(self):
        """Forecast handles series with gaps (NaN-like behavior via zeros)."""
        series = [0.5, 0.0, 0.6, 0.0, 0.7, 0.0, 0.8]
        smoother = ExponentialSmoother(alpha=0.3)
        result = smoother.forecast(series, steps=1)
        # Should still produce a valid forecast
        assert 0.0 <= result.values[0] <= 1.0

    def test_different_alpha_values(self):
        """Different alpha values produce different forecasts."""
        series = [0.3, 0.5, 0.7, 0.9, 1.1]
        smoother_low = ExponentialSmoother(alpha=0.1)
        smoother_high = ExponentialSmoother(alpha=0.9)
        result_low = smoother_low.forecast(series, steps=1)
        result_high = smoother_high.forecast(series, steps=1)
        # Higher alpha should be more responsive to recent values
        assert result_high.values[0] > result_low.values[0]

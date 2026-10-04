"""Tests for IoT anomaly detection: IQR, EWMA, and Welford's algorithm.

TDD: these tests define the expected behavior for the new features.
"""

import math

import pytest

from src.iot.anomaly import EWMADetector, IQRDetector, WelfordAccumulator  # noqa: I001

# ===========================================================================
# IQR-based Anomaly Detection
# ===========================================================================


class TestIQRDetector:
    """IQR-based anomaly detection using Tukey's fences."""

    def test_normal_values_not_anomalous(self):
        """Values within the IQR fences are not anomalous."""
        detector = IQRDetector(window=20, k=1.5)
        for v in [10.0, 11.0, 10.5, 9.5, 10.2, 10.8, 9.8, 10.1]:
            detector.update(v)
        assert detector.is_anomalous(10.3) is False

    def test_outlier_above_upper_fence(self):
        """A value above Q3 + k*IQR is anomalous."""
        detector = IQRDetector(window=20, k=1.5)
        for v in [10.0, 11.0, 10.5, 9.5, 10.2, 10.8, 9.8, 10.1]:
            detector.update(v)
        assert detector.is_anomalous(100.0) is True

    def test_outlier_below_lower_fence(self):
        """A value below Q1 - k*IQR is anomalous."""
        detector = IQRDetector(window=20, k=1.5)
        for v in [10.0, 11.0, 10.5, 9.5, 10.2, 10.8, 9.8, 10.1]:
            detector.update(v)
        assert detector.is_anomalous(-100.0) is True

    def test_insufficient_data_not_anomalous(self):
        """With fewer than 2 values, nothing is anomalous."""
        detector = IQRDetector(window=20, k=1.5)
        detector.update(10.0)
        assert detector.is_anomalous(999.0) is False

    def test_zero_iqr_constant_values(self):
        """When all values are identical (IQR=0), any different value is anomalous."""
        detector = IQRDetector(window=20, k=1.5)
        for _ in range(10):
            detector.update(5.0)
        assert detector.is_anomalous(5.0) is False
        assert detector.is_anomalous(5.1) is True

    def test_rolling_window_discards_old(self):
        """Old values fall out of the rolling window."""
        detector = IQRDetector(window=5, k=1.5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            detector.update(v)
        # Window now has [1,2,3,4,5]; Q1=2, Q3=4, IQR=2
        # Upper fence = 4 + 1.5*2 = 7
        assert detector.is_anomalous(6.0) is False
        assert detector.is_anomalous(8.0) is True

    def test_k_parameter_controls_sensitivity(self):
        """Larger k makes the detector less sensitive."""
        data = [10.0, 11.0, 10.5, 9.5, 10.2, 10.8, 9.8, 10.1]
        detector_loose = IQRDetector(window=20, k=3.0)
        detector_tight = IQRDetector(window=20, k=1.0)
        for v in data:
            detector_loose.update(v)
            detector_tight.update(v)
        # Q1≈9.95, Q3≈10.58, IQR≈0.625
        # k=1.0 upper fence ≈ 11.2; k=3.0 upper fence ≈ 12.45
        # 12.0 is outside tight fence but inside loose fence
        assert detector_loose.is_anomalous(12.0) is False
        assert detector_tight.is_anomalous(12.0) is True

    def test_update_returns_self_for_chaining(self):
        """update() returns self to allow method chaining."""
        detector = IQRDetector(window=10)
        result = detector.update(5.0)
        assert result is detector

    def test_get_stats(self):
        """get_stats returns Q1, Q3, IQR, and fences."""
        detector = IQRDetector(window=20, k=1.5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            detector.update(v)
        stats = detector.get_stats()
        assert "q1" in stats
        assert "q3" in stats
        assert "iqr" in stats
        assert "lower_fence" in stats
        assert "upper_fence" in stats
        assert stats["q1"] == 2.0
        assert stats["q3"] == 4.0
        assert stats["iqr"] == 2.0

    def test_invalid_k_raises(self):
        """k must be positive."""
        with pytest.raises(ValueError, match="k must be positive"):
            IQRDetector(k=0)

    def test_invalid_window_raises(self):
        """Window must be at least 2."""
        with pytest.raises(ValueError, match="window must be at least 2"):
            IQRDetector(window=1)


# ===========================================================================
# EWMA-based Anomaly Detection
# ===========================================================================


class TestEWMADetector:
    """EWMA-based anomaly detection with exponential weighting."""

    def test_normal_values_not_anomalous(self):
        """Values near the EWMA mean are not anomalous."""
        detector = EWMADetector(alpha=0.5, threshold=3.0)
        for i in range(100):
            detector.update(10.0 + (i % 3 - 1) * 5.0)  # Noise ±5.0 around 10.0
        assert detector.is_anomalous(10.5) is False

    def test_large_deviation_is_anomalous(self):
        """A value far from the EWMA mean is anomalous."""
        detector = EWMADetector(alpha=0.3, threshold=3.0)
        for _ in range(50):
            detector.update(10.0)
        assert detector.is_anomalous(100.0) is True

    def test_negative_outlier_is_anomalous(self):
        """A value far below the EWMA mean is anomalous."""
        detector = EWMADetector(alpha=0.3, threshold=3.0)
        for _ in range(50):
            detector.update(10.0)
        assert detector.is_anomalous(-50.0) is True

    def test_adapts_to_new_level(self):
        """After a level shift, the detector adapts and stops flagging."""
        detector = EWMADetector(alpha=0.3, threshold=3.0)
        for _ in range(30):
            detector.update(10.0)
        # Shift to a new level
        for _ in range(30):
            detector.update(20.0)
        # Now 20.0 should be normal
        assert detector.is_anomalous(20.0) is False
        # But 10.0 should now be anomalous
        assert detector.is_anomalous(10.0) is True

    def test_alpha_controls_adaptation_speed(self):
        """Higher alpha adapts faster to level shifts."""
        detector_fast = EWMADetector(alpha=0.8, threshold=3.0)
        detector_slow = EWMADetector(alpha=0.1, threshold=3.0)
        for _ in range(30):
            detector_fast.update(10.0)
            detector_slow.update(10.0)
        # Shift to 20.0
        for _ in range(10):
            detector_fast.update(20.0)
            detector_slow.update(20.0)
        # Fast adapter should have adapted; slow one may still flag 20.0
        assert detector_fast.is_anomalous(20.0) is False

    def test_threshold_controls_sensitivity(self):
        """Higher threshold makes the detector less sensitive."""
        detector_low = EWMADetector(alpha=0.3, threshold=2.0)
        detector_high = EWMADetector(alpha=0.3, threshold=5.0)
        for i in range(50):
            detector_low.update(10.0 + (i % 3 - 1) * 2.0)
            detector_high.update(10.0 + (i % 3 - 1) * 2.0)
        # A moderately deviating value
        assert detector_low.is_anomalous(13.0) is True
        assert detector_high.is_anomalous(13.0) is False

    def test_insufficient_data_not_anomalous(self):
        """With no prior data, nothing is anomalous."""
        detector = EWMADetector(alpha=0.3, threshold=3.0)
        assert detector.is_anomalous(999.0) is False

    def test_update_returns_self_for_chaining(self):
        """update() returns self to allow method chaining."""
        detector = EWMADetector()
        result = detector.update(5.0)
        assert result is detector

    def test_get_stats(self):
        """get_stats returns mean, stddev, and threshold."""
        detector = EWMADetector(alpha=0.3, threshold=3.0)
        for _ in range(20):
            detector.update(10.0)
        stats = detector.get_stats()
        assert "mean" in stats
        assert "stddev" in stats
        assert "threshold" in stats
        assert abs(stats["mean"] - 10.0) < 0.1

    def test_invalid_alpha_raises(self):
        """Alpha must be in (0, 1]."""
        with pytest.raises(ValueError, match="alpha must be in"):
            EWMADetector(alpha=0)
        with pytest.raises(ValueError, match="alpha must be in"):
            EWMADetector(alpha=1.5)

    def test_invalid_threshold_raises(self):
        """Threshold must be positive."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            EWMADetector(threshold=0)


# ===========================================================================
# Welford's Algorithm for Incremental Mean/Stddev
# ===========================================================================


class TestWelfordAccumulator:
    """Welford's online algorithm for incremental mean and stddev."""

    def test_single_value(self):
        """A single value has zero stddev."""
        acc = WelfordAccumulator()
        acc.add(5.0)
        assert acc.count == 1
        assert acc.mean == 5.0
        assert acc.stddev == 0.0

    def test_mean_correct(self):
        """Mean is computed correctly for multiple values."""
        acc = WelfordAccumulator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            acc.add(v)
        assert acc.count == 5
        assert acc.mean == 3.0

    def test_stddev_correct(self):
        """Stddev matches the population stddev."""
        acc = WelfordAccumulator()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for v in values:
            acc.add(v)
        # Population stddev of these values
        expected_mean = sum(values) / len(values)
        expected_var = sum((v - expected_mean) ** 2 for v in values) / len(values)
        expected_stddev = math.sqrt(expected_var)
        assert abs(acc.mean - expected_mean) < 1e-10
        assert abs(acc.stddev - expected_stddev) < 1e-10

    def test_variance_correct(self):
        """Variance matches the population variance."""
        acc = WelfordAccumulator()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            acc.add(v)
        expected_mean = 3.0
        expected_var = sum((v - expected_mean) ** 2 for v in values) / len(values)
        assert abs(acc.variance - expected_var) < 1e-10

    def test_incremental_equals_batch(self):
        """Incremental computation matches batch computation."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        acc = WelfordAccumulator()
        for v in values:
            acc.add(v)
        batch_mean = sum(values) / len(values)
        batch_var = sum((v - batch_mean) ** 2 for v in values) / len(values)
        assert abs(acc.mean - batch_mean) < 1e-10
        assert abs(acc.variance - batch_var) < 1e-10

    def test_large_values_no_overflow(self):
        """Welford's algorithm handles large values without overflow."""
        acc = WelfordAccumulator()
        for _ in range(100):
            acc.add(1e15)
        assert acc.mean == 1e15
        assert acc.stddev == 0.0

    def test_merge_two_accumulators(self):
        """Merging two accumulators gives the same result as one."""
        values_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        values_b = [6.0, 7.0, 8.0, 9.0, 10.0]
        acc_a = WelfordAccumulator()
        acc_b = WelfordAccumulator()
        for v in values_a:
            acc_a.add(v)
        for v in values_b:
            acc_b.add(v)
        acc_a.merge(acc_b)
        all_values = values_a + values_b
        expected_mean = sum(all_values) / len(all_values)
        expected_var = sum((v - expected_mean) ** 2 for v in all_values) / len(all_values)
        assert acc_a.count == 10
        assert abs(acc_a.mean - expected_mean) < 1e-10
        assert abs(acc_a.variance - expected_var) < 1e-10

    def test_merge_empty_accumulator(self):
        """Merging an empty accumulator is a no-op."""
        acc = WelfordAccumulator()
        acc.add(5.0)
        acc.add(10.0)
        empty = WelfordAccumulator()
        acc.merge(empty)
        assert acc.count == 2
        assert acc.mean == 7.5

    def test_reset(self):
        """Reset clears all state."""
        acc = WelfordAccumulator()
        acc.add(5.0)
        acc.add(10.0)
        acc.reset()
        assert acc.count == 0
        assert acc.mean == 0.0
        assert acc.stddev == 0.0

    def test_add_returns_self_for_chaining(self):
        """add() returns self to allow method chaining."""
        acc = WelfordAccumulator()
        result = acc.add(5.0)
        assert result is acc

    def test_numerical_stability(self):
        """Welford's algorithm is numerically stable with offset data."""
        offset = 1e9
        acc = WelfordAccumulator()
        values = [offset + i for i in range(100)]
        for v in values:
            acc.add(v)
        expected_mean = offset + 49.5
        assert abs(acc.mean - expected_mean) < 1e-6

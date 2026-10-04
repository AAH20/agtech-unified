"""Tests for rate-of-change validation (IOT gap: no spike/drift detection).

Sensor readings that jump implausibly fast between consecutive samples
indicate wiring faults, EMI, or tampering. RateOfChangeValidator tracks
per-sensor history and flags spikes (instantaneous jump) and drift
(sustained one-directional movement).
"""

import pytest

from src.iot.data_validation import RateOfChangeValidator


class TestRateOfChangeValidator:
    """Spike and drift detection on consecutive readings."""

    def test_no_spike_on_stable_series(self):
        """A stable series produces no violations."""
        validator = RateOfChangeValidator(max_rate=10.0)
        violations = [validator.check("s1", 25.0, timestamp=t) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
        assert violations == [False, False, False, False, False]

    def test_spike_detected(self):
        """A sudden jump beyond max_rate is flagged."""
        validator = RateOfChangeValidator(max_rate=10.0)
        validator.check("s1", 25.0, timestamp=0.0)
        assert validator.check("s1", 50.0, timestamp=1.0) is True

    def test_no_spike_within_rate(self):
        """A change within the rate limit is not flagged."""
        validator = RateOfChangeValidator(max_rate=10.0)
        validator.check("s1", 25.0, timestamp=0.0)
        assert validator.check("s1", 30.0, timestamp=1.0) is False

    def test_first_reading_never_spikes(self):
        """The first reading for a sensor has no baseline."""
        validator = RateOfChangeValidator(max_rate=10.0)
        assert validator.check("s1", 999.0, timestamp=0.0) is False

    def test_per_sensor_isolation(self):
        """Rate tracking is independent per sensor."""
        validator = RateOfChangeValidator(max_rate=10.0)
        validator.check("s1", 25.0, timestamp=0.0)
        validator.check("s2", 100.0, timestamp=0.0)
        assert validator.check("s1", 26.0, timestamp=1.0) is False
        assert validator.check("s2", 200.0, timestamp=1.0) is True

    def test_negative_rate(self):
        """A sudden drop is also a spike (rate is absolute)."""
        validator = RateOfChangeValidator(max_rate=10.0)
        validator.check("s1", 50.0, timestamp=0.0)
        assert validator.check("s1", 25.0, timestamp=1.0) is True

    def test_drift_detected(self):
        """Sustained one-directional movement beyond drift threshold is flagged."""
        validator = RateOfChangeValidator(max_rate=100.0, drift_threshold=5.0)
        validator.check("s1", 0.0, timestamp=0.0)
        # Each step is within max_rate but cumulative drift exceeds threshold
        assert validator.check("s1", 2.0, timestamp=1.0) is False
        assert validator.check("s1", 4.0, timestamp=2.0) is False
        assert validator.check("s1", 6.0, timestamp=3.0) is True  # drift = 6 > 5

    def test_no_drift_on_oscillation(self):
        """Oscillating values do not accumulate drift."""
        validator = RateOfChangeValidator(max_rate=100.0, drift_threshold=5.0)
        validator.check("s1", 0.0, timestamp=0.0)
        assert validator.check("s1", 2.0, timestamp=1.0) is False
        assert validator.check("s1", 0.0, timestamp=2.0) is False
        assert validator.check("s1", 2.0, timestamp=3.0) is False

    def test_invalid_max_rate(self):
        """max_rate must be positive."""
        with pytest.raises(ValueError):
            RateOfChangeValidator(max_rate=0.0)
        with pytest.raises(ValueError):
            RateOfChangeValidator(max_rate=-1.0)

    def test_invalid_drift_threshold(self):
        """drift_threshold must be positive."""
        with pytest.raises(ValueError):
            RateOfChangeValidator(drift_threshold=0.0)

    def test_history_limit(self):
        """History is bounded to prevent unbounded memory growth."""
        validator = RateOfChangeValidator(max_rate=10.0, history=3)
        for i in range(10):
            validator.check("s1", float(i), timestamp=float(i))
        assert len(validator.get_history("s1")) == 3

    def test_get_history_empty(self):
        """No history for an unknown sensor returns an empty list."""
        validator = RateOfChangeValidator(max_rate=10.0)
        assert validator.get_history("unknown") == []

    def test_reset(self):
        """Reset clears all per-sensor state."""
        validator = RateOfChangeValidator(max_rate=10.0)
        validator.check("s1", 25.0, timestamp=0.0)
        validator.reset()
        assert validator.get_history("s1") == []
        assert validator.check("s1", 999.0, timestamp=1.0) is False

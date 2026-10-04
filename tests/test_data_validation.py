"""Test IoT data validation: schema validation, range checks, anomaly detection."""

import math

from src.iot.data_validation import (
    AnomalyDetector,
    RangeChecker,
    Schema,
    SchemaValidator,
)

SOIL_SCHEMA = Schema(
    fields={
        "sensor_id": {"type": str, "required": True},
        "moisture": {"type": (int, float), "required": True, "min": 0.0, "max": 100.0},
        "temperature": {"type": (int, float), "required": False, "min": -40.0, "max": 80.0},
        "unit": {"type": str, "required": False, "allowed": ["%", "C", "F"]},
    }
)


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------


def test_validate_valid_data():
    """A fully valid record passes with no errors."""
    validator = SchemaValidator()
    result = validator.validate(
        {"sensor_id": "s1", "moisture": 45.5, "temperature": 22.0, "unit": "%"},
        SOIL_SCHEMA,
    )

    assert result.valid is True
    assert result.errors == []


def test_validate_missing_required_field():
    """A missing required field produces an error."""
    validator = SchemaValidator()
    result = validator.validate({"moisture": 45.5}, SOIL_SCHEMA)

    assert result.valid is False
    assert any("sensor_id" in e for e in result.errors)


def test_validate_type_mismatch():
    """A value of the wrong type produces an error."""
    validator = SchemaValidator()
    result = validator.validate({"sensor_id": "s1", "moisture": "wet"}, SOIL_SCHEMA)

    assert result.valid is False
    assert any("moisture" in e for e in result.errors)


def test_validate_range_violation():
    """A value outside [min, max] produces an error."""
    validator = SchemaValidator()
    result = validator.validate({"sensor_id": "s1", "moisture": 150.0}, SOIL_SCHEMA)

    assert result.valid is False
    assert any("moisture" in e for e in result.errors)


def test_validate_allowed_values():
    """A value not in the allowed set produces an error."""
    validator = SchemaValidator()
    result = validator.validate(
        {"sensor_id": "s1", "moisture": 45.5, "unit": "kelvin"}, SOIL_SCHEMA
    )

    assert result.valid is False
    assert any("unit" in e for e in result.errors)


def test_validate_multiple_errors():
    """All violations are reported, not just the first."""
    validator = SchemaValidator()
    result = validator.validate({"moisture": -5.0, "unit": "kelvin"}, SOIL_SCHEMA)

    assert result.valid is False
    assert len(result.errors) >= 3  # missing sensor_id, moisture range, unit enum


# ----------------------------------------------------------------------
# Range checks
# ----------------------------------------------------------------------


def test_range_checker_within_bounds():
    """A value inside the range passes."""
    checker = RangeChecker(min_value=0.0, max_value=100.0)

    assert checker.check(50.0) is True
    assert checker.check(0.0) is True
    assert checker.check(100.0) is True


def test_range_checker_outside_bounds():
    """A value outside the range fails."""
    checker = RangeChecker(min_value=0.0, max_value=100.0)

    assert checker.check(-1.0) is False
    assert checker.check(100.1) is False


def test_range_checker_warn_threshold():
    """A value past the warn threshold is flagged but still valid."""
    checker = RangeChecker(min_value=0.0, max_value=100.0, warn_above=90.0)

    ok, warned = checker.check_with_warning(95.0)

    assert ok is True
    assert warned is True


# ----------------------------------------------------------------------
# Anomaly detection
# ----------------------------------------------------------------------


def test_anomaly_detector_flags_outlier():
    """A value far from the rolling mean is flagged as anomalous."""
    detector = AnomalyDetector(window=10, threshold=2.0)
    for value in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0]:
        detector.update(value)

    assert detector.is_anomalous(50.0) is True


def test_anomaly_detector_normal_stream_clean():
    """Values near the rolling mean are not flagged."""
    detector = AnomalyDetector(window=10, threshold=2.0)
    for value in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0]:
        detector.update(value)

    assert detector.is_anomalous(10.2) is False


def test_anomaly_detector_handles_constant_stream():
    """A zero-variance stream does not produce NaN or false positives."""
    detector = AnomalyDetector(window=5, threshold=2.0)
    for _ in range(5):
        detector.update(10.0)

    assert detector.is_anomalous(10.0) is False
    assert not math.isnan(detector.stddev)

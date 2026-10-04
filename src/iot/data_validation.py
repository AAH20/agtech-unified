"""IoT data validation: schema validation, range checks, anomaly detection.

Layers of defense for agricultural sensor data:
- SchemaValidator: structural validation (types, required fields, enums, bounds)
- RangeChecker: standalone numeric range checks with warning thresholds
- AnomalyDetector: rolling-window statistical anomaly detection
- CrossFieldValidator: conditional rules and field dependencies
- TemporalValidator: timestamp ordering, future/stale detection
- UnitConverter: unit conversion and consistency validation
- QualityScorer: data quality scoring (completeness, freshness, anomalies)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Outcome of a validation pass."""

    valid: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class Schema:
    """Declarative schema for sensor data records.

    Each field maps to a spec dict supporting:
    - type: expected Python type or tuple of types
    - required: whether the field must be present
    - min / max: numeric bounds (inclusive)
    - allowed: set of permitted values (enum)
    """

    fields: Dict[str, Dict[str, Any]]


class SchemaValidator:
    """Validates data records against a Schema."""

    def validate(self, data: Dict[str, Any], schema: Schema) -> ValidationResult:
        """Validate a record, collecting all errors (not fail-fast)."""
        errors: List[str] = []

        for name, spec in schema.fields.items():
            value = data.get(name)

            if value is None:
                if spec.get("required", False):
                    errors.append(f"Field '{name}' is required")
                continue

            expected = spec.get("type")
            if expected is not None and not isinstance(value, expected):
                errors.append(
                    f"Field '{name}' must be of type {expected}, got {type(value).__name__}"
                )
                continue  # range/enum checks are meaningless on wrong types

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                min_value = spec.get("min")
                max_value = spec.get("max")
                if min_value is not None and value < min_value:
                    errors.append(f"Field '{name}' value {value} below minimum {min_value}")
                if max_value is not None and value > max_value:
                    errors.append(f"Field '{name}' value {value} above maximum {max_value}")

            allowed = spec.get("allowed")
            if allowed is not None and value not in allowed:
                errors.append(f"Field '{name}' value {value!r} not in allowed set {allowed}")

        return ValidationResult(valid=not errors, errors=errors)


class RangeChecker:
    """Standalone numeric range check with optional warning threshold."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        warn_below: Optional[float] = None,
        warn_above: Optional[float] = None,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.warn_below = warn_below
        self.warn_above = warn_above

    def check(self, value: float) -> bool:
        """True when value is within [min_value, max_value] (inclusive)."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def check_with_warning(self, value: float) -> Tuple[bool, bool]:
        """Returns (in_range, warned).

        in_range is False outside hard bounds; warned is True when inside
        bounds but past a warn threshold.
        """
        in_range = self.check(value)
        warned = False
        if in_range:
            if self.warn_below is not None and value < self.warn_below:
                warned = True
            if self.warn_above is not None and value > self.warn_above:
                warned = True
        return in_range, warned


class AnomalyDetector:
    """Rolling-window statistical anomaly detection.

    Maintains a sliding window of recent values; a new value is anomalous
    when it deviates from the window mean by more than `threshold` standard
    deviations. Requires at least 2 samples before judging, and treats a
    zero-variance window as non-anomalous for any value equal to the mean.
    """

    def __init__(self, window: int = 20, threshold: float = 3.0) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.window = window
        self.threshold = threshold
        self._values: Deque[float] = deque(maxlen=window)
        self.mean: float = 0.0
        self.stddev: float = 0.0

    def update(self, value: float) -> None:
        """Add a value to the rolling window and refresh statistics."""
        self._values.append(float(value))
        n = len(self._values)
        self.mean = sum(self._values) / n
        if n < 2:
            self.stddev = 0.0
            return
        variance = sum((v - self.mean) ** 2 for v in self._values) / (n - 1)
        self.stddev = math.sqrt(variance)

    def is_anomalous(self, value: float) -> bool:
        """True when value deviates beyond threshold standard deviations."""
        if len(self._values) < 2:
            return False
        if self.stddev == 0.0:
            return value != self.mean
        z_score = abs(value - self.mean) / self.stddev
        return z_score > self.threshold


# ===========================================================================
# Cross-Field Validation (IOT-052)
# ===========================================================================


class CrossFieldValidator:
    """Cross-field validation for conditional rules and field dependencies.

    Supports:
    - Conditional rules: if field X equals value Y, then field Z must satisfy constraint
    - Field dependencies: if field A is present, field B must also be present
    """

    def __init__(self) -> None:
        self._rules: List[Dict[str, Any]] = []
        self._dependencies: List[Tuple[str, str]] = []

    def add_rule(
        self,
        condition: Dict[str, Any],
        constraint: Dict[str, Any],
    ) -> None:
        """Add a conditional rule.

        condition: {"field": "irrigation", "equals": "on"}
        constraint: {"field": "flow_rate", "min": 0.1}
        """
        self._rules.append({"condition": condition, "constraint": constraint})

    def add_dependency(self, field: str, requires: str) -> None:
        """Add a field dependency: if 'field' is present, 'requires' must also be present."""
        self._dependencies.append((field, requires))

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against all rules and dependencies.

        Returns {"valid": bool, "errors": List[str]}.
        """
        errors: List[str] = []

        # Check dependencies
        for dep_field, requires in self._dependencies:
            if dep_field in data and requires not in data:
                errors.append(f"Field '{dep_field}' requires field '{requires}'")

        # Check conditional rules
        for rule in self._rules:
            cond = rule["condition"]
            cons = rule["constraint"]

            cond_field = cond["field"]
            cond_value = cond.get("equals")

            # Check if condition is met
            if cond_field in data and data[cond_field] == cond_value:
                # Condition met, check constraint
                cons_field = cons["field"]
                if cons_field not in data:
                    errors.append(
                        f"Field '{cons_field}' is required when "
                        f"'{cond_field}' equals '{cond_value}'"
                    )
                else:
                    value = data[cons_field]
                    if "min" in cons and value < cons["min"]:
                        errors.append(
                            f"Field '{cons_field}' must be >= {cons['min']} "
                            f"when '{cond_field}' equals '{cond_value}'"
                        )
                    if "max" in cons and value > cons["max"]:
                        errors.append(
                            f"Field '{cons_field}' must be <= {cons['max']} "
                            f"when '{cond_field}' equals '{cond_value}'"
                        )

        return {"valid": len(errors) == 0, "errors": errors}


# ===========================================================================
# Temporal Validation (IOT-053)
# ===========================================================================


class TemporalValidator:
    """Temporal validation for timestamps.

    Validates:
    - Future timestamps (reject)
    - Stale data (flag)
    - Monotonic ordering (reject out-of-order)
    """

    def __init__(
        self,
        max_future_seconds: float = 60.0,
        stale_threshold_seconds: float = 300.0,
    ) -> None:
        self.max_future_seconds = max_future_seconds
        self.stale_threshold_seconds = stale_threshold_seconds

    def validate_timestamp(self, timestamp: float) -> Dict[str, Any]:
        """Validate a single timestamp.

        Returns {"valid": bool, "stale": bool, "errors": List[str]}.
        """
        now = time.time()
        errors: List[str] = []
        stale = False

        # Check future timestamp
        if timestamp > now + self.max_future_seconds:
            errors.append(
                f"Timestamp {timestamp} is in the future "
                f"(max allowed: {now + self.max_future_seconds})"
            )

        # Check stale data
        if now - timestamp > self.stale_threshold_seconds:
            stale = True

        return {"valid": len(errors) == 0, "stale": stale, "errors": errors}

    def validate_sequence(self, timestamps: List[float]) -> Dict[str, Any]:
        """Validate that timestamps are in monotonic order.

        Returns {"valid": bool, "errors": List[str]}.
        """
        errors: List[str] = []
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                errors.append(
                    f"Timestamp at index {i} ({timestamps[i]}) is before "
                    f"timestamp at index {i - 1} ({timestamps[i - 1]})"
                )
        return {"valid": len(errors) == 0, "errors": errors}


# ===========================================================================
# Unit Conversion (IOT-054)
# ===========================================================================


class UnitConverter:
    """Unit conversion and consistency validation.

    Supports temperature, length, weight, and pressure conversions.
    """

    # Conversion factors to SI base units
    _CONVERSIONS = {
        "celsius": {
            "to_si": lambda x: x + 273.15,
            "from_si": lambda x: x - 273.15,
            "si_unit": "kelvin",
        },
        "fahrenheit": {
            "to_si": lambda x: (x - 32) * 5 / 9 + 273.15,
            "from_si": lambda x: (x - 273.15) * 9 / 5 + 32,
            "si_unit": "kelvin",
        },
        "kelvin": {"to_si": lambda x: x, "from_si": lambda x: x, "si_unit": "kelvin"},
        "meter": {"to_si": lambda x: x, "from_si": lambda x: x, "si_unit": "meter"},
        "foot": {
            "to_si": lambda x: x * 0.3048,
            "from_si": lambda x: x / 0.3048,
            "si_unit": "meter",
        },
        "kilogram": {"to_si": lambda x: x, "from_si": lambda x: x, "si_unit": "kilogram"},
        "pound": {
            "to_si": lambda x: x * 0.453592,
            "from_si": lambda x: x / 0.453592,
            "si_unit": "kilogram",
        },
        "pascal": {"to_si": lambda x: x, "from_si": lambda x: x, "si_unit": "pascal"},
        "psi": {
            "to_si": lambda x: x * 6894.76,
            "from_si": lambda x: x / 6894.76,
            "si_unit": "pascal",
        },
    }

    _UNIT_TYPES = {
        "celsius": "temperature",
        "fahrenheit": "temperature",
        "kelvin": "temperature",
        "meter": "length",
        "foot": "length",
        "kilogram": "weight",
        "pound": "weight",
        "pascal": "pressure",
        "psi": "pressure",
    }

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert a value from one unit to another.

        Raises ValueError if units are incompatible or unknown.
        """
        if from_unit == to_unit:
            return value

        if from_unit not in self._CONVERSIONS:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_unit not in self._CONVERSIONS:
            raise ValueError(f"Unknown unit: {to_unit}")

        # Check compatibility
        if not self.is_compatible(from_unit, to_unit):
            raise ValueError(f"Incompatible units: {from_unit} and {to_unit}")

        # Convert to SI, then to target
        si_value = self._CONVERSIONS[from_unit]["to_si"](value)
        return self._CONVERSIONS[to_unit]["from_si"](si_value)

    def is_compatible(self, unit_a: str, unit_b: str) -> bool:
        """Check if two units are compatible (same physical quantity)."""
        if unit_a not in self._UNIT_TYPES or unit_b not in self._UNIT_TYPES:
            return False
        return self._UNIT_TYPES[unit_a] == self._UNIT_TYPES[unit_b]

    def to_si(self, value: float, unit: str, quantity: str) -> float:
        """Convert a value to SI units for a given quantity type."""
        if unit not in self._CONVERSIONS:
            raise ValueError(f"Unknown unit: {unit}")
        return self._CONVERSIONS[unit]["to_si"](value)


# ===========================================================================
# Quality Scoring (IOT-055)
# ===========================================================================


class QualityScorer:
    """Data quality scoring based on completeness, freshness, and anomalies."""

    def completeness(self, data: Dict[str, Any], required_fields: List[str]) -> float:
        """Compute completeness score (0.0 to 1.0).

        Score = (number of present required fields) / (total required fields).
        """
        if not required_fields:
            return 1.0
        present = sum(1 for f in required_fields if f in data and data[f] is not None)
        return present / len(required_fields)

    def freshness(self, timestamp: float, max_age: float) -> float:
        """Compute freshness score (0.0 to 1.0).

        Score decreases linearly from 1.0 (now) to 0.0 (max_age).
        """
        age = time.time() - timestamp
        if age <= 0:
            return 1.0
        if age >= max_age:
            return 0.0
        return 1.0 - (age / max_age)

    def overall(
        self,
        completeness: float,
        freshness: float,
        anomaly_rate: float,
    ) -> float:
        """Compute overall quality score (0.0 to 1.0).

        Weighted combination of completeness (40%), freshness (30%),
        and anomaly rate (30%, inverted).
        """
        completeness = max(0.0, min(1.0, completeness))
        freshness = max(0.0, min(1.0, freshness))
        anomaly_rate = max(0.0, min(1.0, anomaly_rate))

        return 0.4 * completeness + 0.3 * freshness + 0.3 * (1.0 - anomaly_rate)

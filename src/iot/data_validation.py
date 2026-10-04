"""IoT data validation: schema validation, range checks, anomaly detection.

Three layers of defense for agricultural sensor data:
- SchemaValidator: structural validation (types, required fields, enums, bounds)
- RangeChecker: standalone numeric range checks with warning thresholds
- AnomalyDetector: rolling-window statistical anomaly detection
"""

from __future__ import annotations

import math
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

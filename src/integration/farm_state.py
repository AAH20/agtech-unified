"""Shared FarmState model with validation, serialization, history, and diff.

Canonical data model for farm state across all modules. Replaces the
unvalidated FarmState in decision_support.recommender (GAP-011, GAP-016).

Enhanced with:
- JSON serialization/deserialization
- Time-series history with trend analysis
- State diff for change tracking
- State transition validation
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Validation ranges for each field
_RANGES = {
    "soil_moisture": (0.0, 1.0),
    "temperature": (-50.0, 60.0),
    "crop_height": (0.0, 10.0),
    "nutrient_level": (0.0, 1.0),
    "pest_pressure": (0.0, 1.0),
}


@dataclass
class FarmState:
    """Current state of the farm with validated fields.

    Ranges:
        soil_moisture:  0.0 to 1.0  (volumetric water content fraction)
        temperature:    -50 to 60    (Celsius)
        crop_height:    0.0 to 10.0  (meters)
        nutrient_level: 0.0 to 1.0  (NPK availability fraction)
        pest_pressure:  0.0 to 1.0  (infestation severity fraction)
    """

    soil_moisture: float
    temperature: float
    crop_height: float
    nutrient_level: float
    pest_pressure: float
    timestamp: float = field(default_factory=time.time)
    tenant_id: Optional[str] = None

    def __post_init__(self):
        for field_name, (lo, hi) in _RANGES.items():
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
            if not (lo <= value <= hi):
                raise ValueError(f"{field_name} must be in [{lo}, {hi}], got {value}")

    def is_valid(self) -> bool:
        """Return True if all fields are within their valid ranges."""
        try:
            self.__post_init__()
            return True
        except (ValueError, TypeError):
            return False

    SCHEMA_VERSION = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "soil_moisture": self.soil_moisture,
            "temperature": self.temperature,
            "crop_height": self.crop_height,
            "nutrient_level": self.nutrient_level,
            "pest_pressure": self.pest_pressure,
            "timestamp": self.timestamp,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FarmState":
        """Deserialize from dictionary."""
        return cls(
            soil_moisture=data["soil_moisture"],
            temperature=data["temperature"],
            crop_height=data["crop_height"],
            nutrient_level=data["nutrient_level"],
            pest_pressure=data["pest_pressure"],
            timestamp=data.get("timestamp", time.time()),
        )

    def update(self, **kwargs: Any) -> None:
        """Update individual fields in-place with validation.

        Args:
            **kwargs: Field names and their new values.

        Raises:
            ValueError: If an unknown field name is provided or a value is out of range.
        """
        for field_name, value in kwargs.items():
            if field_name not in _RANGES:
                raise ValueError(f"Unknown field: {field_name!r}. Valid fields: {sorted(_RANGES)}")
        # Validate all values before applying any
        for field_name, value in kwargs.items():
            lo, hi = _RANGES[field_name]
            if not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
            if not (lo <= value <= hi):
                raise ValueError(f"{field_name} must be in [{lo}, {hi}], got {value}")
        # Apply updates
        for field_name, value in kwargs.items():
            setattr(self, field_name, value)

    def patch(self, **kwargs: Any) -> None:
        """Alias for update()."""
        self.update(**kwargs)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "FarmState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class FarmStateDiff:
    """Diff between two FarmState snapshots."""

    changed_fields: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # Each value is (delta, new_value, percent_change)

    @classmethod
    def compute(cls, old: FarmState, new: FarmState) -> "FarmStateDiff":
        """Compute diff between two states."""
        changed = {}
        for field_name in _RANGES:
            old_val = getattr(old, field_name)
            new_val = getattr(new, field_name)
            if old_val != new_val:
                delta = new_val - old_val
                if old_val != 0:
                    pct = (abs(delta) / abs(old_val)) * 100.0
                else:
                    pct = 0.0 if new_val == 0 else 100.0
                changed[field_name] = (delta, new_val, pct)
        return cls(changed_fields=changed)

    def has_changes(self) -> bool:
        """Return True if any fields changed."""
        return len(self.changed_fields) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for field_name, (delta, new_val, pct) in self.changed_fields.items():
            old_val = new_val - delta
            result[field_name] = {"old": old_val, "new": new_val, "delta": delta, "percent": pct}
        return result


class FarmStateHistory:
    """Time-series history of FarmState snapshots."""

    def __init__(self, max_size: int = 1000):
        self._history: List[FarmState] = []
        self._max_size = max_size

    def append(self, state: FarmState) -> None:
        """Append a state snapshot."""
        self._history.append(state)
        if len(self._history) > self._max_size:
            self._history.pop(0)

    def get_all(self) -> List[FarmState]:
        """Return all snapshots."""
        return list(self._history)

    def get_by_range(self, start: float, end: float) -> List[FarmState]:
        """Return snapshots within a time range."""
        return [s for s in self._history if start <= s.timestamp <= end]

    def get_latest(self) -> Optional[FarmState]:
        """Return the most recent snapshot."""
        return self._history[-1] if self._history else None

    def get_trend(self, field_name: str) -> str:
        """Get trend direction for a field: 'increasing', 'decreasing', or 'stable'."""
        if len(self._history) < 2:
            return "stable"

        values = [getattr(s, field_name) for s in self._history]
        first = values[0]
        last = values[-1]
        threshold = abs(first) * 0.05 if first != 0 else 0.01

        if last - first > threshold:
            return "increasing"
        elif first - last > threshold:
            return "decreasing"
        return "stable"

    def get_rate_of_change(self, field_name: str) -> float:
        """Get rate of change for a field (units per second)."""
        if len(self._history) < 2:
            return 0.0

        values = [getattr(s, field_name) for s in self._history]
        times = [s.timestamp for s in self._history]
        total_change = values[-1] - values[0]
        total_time = times[-1] - times[0]
        if total_time == 0:
            return 0.0
        return total_change / total_time

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()


class StateTransitionValidator:
    """Validates state transitions based on agricultural rules."""

    # Thresholds for significant changes
    MOISTURE_CHANGE_THRESHOLD = 0.1
    CROP_HEIGHT_CHANGE_THRESHOLD = 0.1
    NUTRIENT_CHANGE_THRESHOLD = 0.2

    def validate(
        self,
        old: FarmState,
        new: FarmState,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate a state transition.

        Args:
            old: Previous state.
            new: New state.
            context: Optional context (e.g., {"irrigation_event": True}).

        Returns:
            True if the transition is valid.
        """
        context = context or {}
        violations = self.get_violations(old, new, context)
        return len(violations) == 0

    def get_violations(
        self,
        old: FarmState,
        new: FarmState,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Get list of transition violations.

        Args:
            old: Previous state.
            new: New state.
            context: Optional context.

        Returns:
            List of violation descriptions.
        """
        context = context or {}
        violations = []

        # Moisture increase requires irrigation
        moisture_delta = new.soil_moisture - old.soil_moisture
        if moisture_delta > self.MOISTURE_CHANGE_THRESHOLD:
            if not context.get("irrigation_event"):
                violations.append(
                    f"soil_moisture increased by {moisture_delta:.2f} without irrigation event"
                )

        # Crop height decrease requires harvest or stress
        height_delta = new.crop_height - old.crop_height
        if height_delta < -self.CROP_HEIGHT_CHANGE_THRESHOLD:
            if not context.get("harvest_event") and not context.get("stress_event"):
                violations.append(
                    f"crop_height decreased by {abs(height_delta):.2f} "
                    f"without harvest or stress event"
                )

        # Nutrient decrease requires fertilization or uptake
        nutrient_delta = new.nutrient_level - old.nutrient_level
        if nutrient_delta < -self.NUTRIENT_CHANGE_THRESHOLD:
            if not context.get("fertilization_event") and not context.get("uptake_event"):
                violations.append(
                    f"nutrient_level decreased by {abs(nutrient_delta):.2f} "
                    f"without fertilization or uptake event"
                )

        return violations

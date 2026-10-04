"""Shared FarmState model with validation.

Canonical data model for farm state across all modules. Replaces the
unvalidated FarmState in decision_support.recommender (GAP-011, GAP-016).
"""
from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self):
        for field_name, (lo, hi) in _RANGES.items():
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
            if not (lo <= value <= hi):
                raise ValueError(
                    f"{field_name} must be in [{lo}, {hi}], got {value}"
                )

    def is_valid(self) -> bool:
        """Return True if all fields are within their valid ranges."""
        try:
            self.__post_init__()
            return True
        except (ValueError, TypeError):
            return False

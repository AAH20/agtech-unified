"""Shared state model for unified digital twin.

Provides DigitalTwinState as the single source of truth that all
modules (simulator, water balance, nutrient cycling) read from and
write to, eliminating state duplication and inconsistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DigitalTwinState:
    """Unified state for the agricultural digital twin.

    All modules read from and write to this state, ensuring
    consistency across the water balance, nutrient cycling, and
    growth simulation subsystems.
    """

    # Core physical state
    soil_moisture: float = 0.3  # m³/m³
    temperature: float = 25.0  # Celsius
    crop_height: float = 0.0  # meters

    # Nutrient state (kg/ha)
    nitrogen: float = 0.0
    phosphorus: float = 0.0
    potassium: float = 0.0
    organic_matter: float = 0.0  # %
    pH: float = 6.5

    # Biotic stress factors (0-1)
    pest_pressure: float = 0.0
    disease_pressure: float = 0.0
    weed_pressure: float = 0.0

    # Metadata
    field_id: Optional[str] = None
    crop_type: Optional[str] = None
    growth_stage: Optional[str] = None
    day: int = 0

    # Additional properties for extensibility
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def nutrient_level(self) -> float:
        """Compute aggregate nutrient level (0-1) from N/P/K.

        Normalizes each nutrient against typical sufficiency ranges
        and returns the minimum (Liebig's law of the minimum).
        """
        n_norm = min(1.0, self.nitrogen / 100.0)
        p_norm = min(1.0, self.phosphorus / 20.0)
        k_norm = min(1.0, self.potassium / 80.0)
        return min(n_norm, p_norm, k_norm)

    def combined_stress_factor(self) -> float:
        """Compute combined stress factor from all stressors.

        Returns a value 0-1 where 1 = no stress, 0 = maximum stress.
        Combines water, temperature, nutrient, and biotic stresses.
        """
        # Water stress
        water_stress = 1.0
        if self.soil_moisture < 0.1:
            water_stress = 0.0
        elif self.soil_moisture < 0.3:
            water_stress = (self.soil_moisture - 0.1) / 0.2

        # Temperature stress (Gaussian around 25°C)
        import math

        temp_stress = math.exp(-((self.temperature - 25.0) ** 2) / 200.0)

        # Nutrient stress
        nutrient_stress = self.nutrient_level

        # Biotic stress
        biotic_stress = 1.0 - max(self.pest_pressure, self.disease_pressure, self.weed_pressure)

        return water_stress * temp_stress * nutrient_stress * biotic_stress

    def to_simulation_state(self):
        """Convert to SimulationState for backward compatibility."""
        from src.digital_twin.simulator import SimulationState

        return SimulationState(
            soil_moisture=self.soil_moisture,
            temperature=self.temperature,
            crop_height=self.crop_height,
            nutrient_level=self.nutrient_level,
            pest_pressure=self.pest_pressure,
            disease_pressure=self.disease_pressure,
            weed_pressure=self.weed_pressure,
        )

    @classmethod
    def from_simulation_state(cls, sim_state) -> DigitalTwinState:
        """Create from SimulationState."""
        return cls(
            soil_moisture=sim_state.soil_moisture,
            temperature=sim_state.temperature,
            crop_height=sim_state.crop_height,
            nitrogen=sim_state.nutrient_level * 100.0,
            phosphorus=sim_state.nutrient_level * 20.0,
            potassium=sim_state.nutrient_level * 80.0,
            pest_pressure=getattr(sim_state, "pest_pressure", 0.0),
            disease_pressure=getattr(sim_state, "disease_pressure", 0.0),
            weed_pressure=getattr(sim_state, "weed_pressure", 0.0),
        )

    def to_water_balance_state(self):
        """Convert to WaterBalanceState."""
        from src.digital_twin.water_balance import WaterBalanceState

        return WaterBalanceState(
            soil_moisture=self.soil_moisture,
        )

    @classmethod
    def from_water_balance_state(cls, wb_state) -> DigitalTwinState:
        """Create from WaterBalanceState."""
        return cls(
            soil_moisture=wb_state.soil_moisture,
        )

    def to_nutrient_state(self):
        """Convert to NutrientState."""
        from src.digital_twin.nutrient_cycling import NutrientState

        return NutrientState(
            nitrogen=self.nitrogen,
            phosphorus=self.phosphorus,
            potassium=self.potassium,
            organic_matter=self.organic_matter,
            pH=self.pH,
        )

    @classmethod
    def from_nutrient_state(cls, nut_state) -> DigitalTwinState:
        """Create from NutrientState."""
        return cls(
            nitrogen=nut_state.nitrogen,
            phosphorus=nut_state.phosphorus,
            potassium=nut_state.potassium,
            organic_matter=nut_state.organic_matter,
            pH=nut_state.pH,
        )

    def clone(self) -> DigitalTwinState:
        """Create an independent copy."""
        return DigitalTwinState.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "soil_moisture": self.soil_moisture,
            "temperature": self.temperature,
            "crop_height": self.crop_height,
            "nitrogen": self.nitrogen,
            "phosphorus": self.phosphorus,
            "potassium": self.potassium,
            "organic_matter": self.organic_matter,
            "pH": self.pH,
            "pest_pressure": self.pest_pressure,
            "disease_pressure": self.disease_pressure,
            "weed_pressure": self.weed_pressure,
            "field_id": self.field_id,
            "crop_type": self.crop_type,
            "growth_stage": self.growth_stage,
            "day": self.day,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DigitalTwinState:
        """Deserialize from dictionary."""
        return cls(
            soil_moisture=d.get("soil_moisture", 0.3),
            temperature=d.get("temperature", 25.0),
            crop_height=d.get("crop_height", 0.0),
            nitrogen=d.get("nitrogen", 0.0),
            phosphorus=d.get("phosphorus", 0.0),
            potassium=d.get("potassium", 0.0),
            organic_matter=d.get("organic_matter", 0.0),
            pH=d.get("pH", 6.5),
            pest_pressure=d.get("pest_pressure", 0.0),
            disease_pressure=d.get("disease_pressure", 0.0),
            weed_pressure=d.get("weed_pressure", 0.0),
            field_id=d.get("field_id"),
            crop_type=d.get("crop_type"),
            growth_stage=d.get("growth_stage"),
            day=d.get("day", 0),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class StateSnapshot:
    """Timestamped snapshot of digital twin state for persistence."""

    state: DigitalTwinState
    day: int
    label: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            import time

            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshot."""
        return {
            "state": self.state.to_dict(),
            "day": self.day,
            "label": self.label,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StateSnapshot:
        """Deserialize snapshot."""
        return cls(
            state=DigitalTwinState.from_dict(d["state"]),
            day=d["day"],
            label=d.get("label"),
            timestamp=d.get("timestamp"),
        )

"""Digital twin simulation engine for agriculture."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Current state of the agricultural system."""
    soil_moisture: float  # 0.0 to 1.0
    temperature: float   # Celsius
    crop_height: float   # meters
    nutrient_level: float  # 0.0 to 1.0


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    days_simulated: int
    final_state: SimulationState
    history: List[SimulationState]
    algorithm: str
    total_growth: float


class DigitalTwin:
    """Digital twin simulator using logistic growth model."""

    def __init__(self, algorithm: str = "logistic_growth"):
        self.algorithm = algorithm
        self.max_crop_height = 2.0  # meters
        self.optimal_temp = 25.0
        self.growth_rate = 0.1

    def simulate(self, initial_state: SimulationState, days: int) -> SimulationResult:
        """Simulate crop growth over time."""
        if days < 0:
            raise ValueError("Days must be non-negative")

        state = SimulationState(
            soil_moisture=initial_state.soil_moisture,
            temperature=initial_state.temperature,
            crop_height=initial_state.crop_height,
            nutrient_level=initial_state.nutrient_level,
        )

        history = []
        for _ in range(days):
            state = self._step(state)
            history.append(SimulationState(
                soil_moisture=state.soil_moisture,
                temperature=state.temperature,
                crop_height=state.crop_height,
                nutrient_level=state.nutrient_level,
            ))

        total_growth = state.crop_height - initial_state.crop_height

        return SimulationResult(
            days_simulated=days,
            final_state=state,
            history=history,
            algorithm=self.algorithm,
            total_growth=total_growth,
        )

    def _step(self, state: SimulationState) -> SimulationState:
        """Advance simulation by one day."""
        # Logistic growth: dh/dt = r * h * (1 - h/K) * stress_factor
        temp_factor = self._temp_factor(state.temperature)
        water_factor = self._water_factor(state.soil_moisture)
        nutrient_factor = state.nutrient_level

        stress = temp_factor * water_factor * nutrient_factor

        growth = self.growth_rate * state.crop_height * (1 - state.crop_height / self.max_crop_height) * stress
        new_height = min(self.max_crop_height, state.crop_height + growth)

        # Soil moisture depletes
        new_moisture = max(0.0, state.soil_moisture - 0.02 * stress)

        return SimulationState(
            soil_moisture=new_moisture,
            temperature=state.temperature,
            crop_height=new_height,
            nutrient_level=state.nutrient_level,
        )

    def _temp_factor(self, temp: float) -> float:
        """Temperature stress factor (0-1)."""
        if temp < 0 or temp > 45:
            return 0.0
        # Gaussian around optimal
        return math.exp(-((temp - self.optimal_temp) ** 2) / 200.0)

    def _water_factor(self, moisture: float) -> float:
        """Water stress factor (0-1)."""
        if moisture < 0.1:
            return 0.0
        return min(1.0, moisture / 0.5)

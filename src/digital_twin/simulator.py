"""Digital twin simulation engine for agriculture."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.integration.farm_state import FarmState

logger = logging.getLogger(__name__)


class CropType(Enum):
    """Supported crop types with specific parameters."""

    WHEAT = "wheat"
    CORN = "corn"
    RICE = "rice"
    SOYBEAN = "soybean"
    POTATO = "potato"
    TOMATO = "tomato"
    GENERIC = "generic"


_CROP_PROFILES: Dict[CropType, Dict[str, float]] = {
    CropType.WHEAT: {
        "max_crop_height": 1.5,
        "optimal_temp": 20.0,
        "growth_rate": 0.08,
        "crop_coefficient": 1.15,
    },
    CropType.CORN: {
        "max_crop_height": 3.0,
        "optimal_temp": 28.0,
        "growth_rate": 0.12,
        "crop_coefficient": 1.25,
    },
    CropType.RICE: {
        "max_crop_height": 1.2,
        "optimal_temp": 30.0,
        "growth_rate": 0.10,
        "crop_coefficient": 1.20,
    },
    CropType.SOYBEAN: {
        "max_crop_height": 1.0,
        "optimal_temp": 26.0,
        "growth_rate": 0.09,
        "crop_coefficient": 1.10,
    },
    CropType.POTATO: {
        "max_crop_height": 0.6,
        "optimal_temp": 18.0,
        "growth_rate": 0.11,
        "crop_coefficient": 1.05,
    },
    CropType.TOMATO: {
        "max_crop_height": 2.0,
        "optimal_temp": 25.0,
        "growth_rate": 0.10,
        "crop_coefficient": 1.15,
    },
    CropType.GENERIC: {
        "max_crop_height": 2.0,
        "optimal_temp": 25.0,
        "growth_rate": 0.10,
        "crop_coefficient": 1.0,
    },
}


@dataclass
class SimulationState:
    """Current state of the agricultural system."""

    soil_moisture: float
    temperature: float
    crop_height: float
    nutrient_level: float
    pest_pressure: float = 0.0
    disease_pressure: float = 0.0
    weed_pressure: float = 0.0


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

    def __init__(
        self,
        algorithm: str = "logistic_growth",
        crop_type: CropType = CropType.GENERIC,
    ):
        self.algorithm = algorithm
        self.crop_type = crop_type
        profile = _CROP_PROFILES.get(crop_type, _CROP_PROFILES[CropType.GENERIC])
        self.max_crop_height = profile["max_crop_height"]
        self.optimal_temp = profile["optimal_temp"]
        self.growth_rate = profile["growth_rate"]
        self.crop_coefficient = profile["crop_coefficient"]
        self._current_state: Optional[SimulationState] = None
        self._bus: Optional[EventBus] = None

    def ingest_sensor_data(self, state: FarmState) -> SimulationState:
        """Ingest a FarmState from IoT sensors and update the simulation state.

        Args:
            state: Validated FarmState from the integration layer.

        Returns:
            The updated SimulationState.

        Raises:
            ValueError: If the FarmState fails validation.
        """
        if not state.is_valid():
            raise ValueError("Invalid FarmState: sensor data failed validation")
        self._current_state = SimulationState(
            soil_moisture=state.soil_moisture,
            temperature=state.temperature,
            crop_height=state.crop_height,
            nutrient_level=state.nutrient_level,
            pest_pressure=getattr(state, "pest_pressure", 0.0),
            disease_pressure=getattr(state, "disease_pressure", 0.0),
            weed_pressure=getattr(state, "weed_pressure", 0.0),
        )
        if self._bus:
            self._bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED,
                    source="digital_twin.simulator",
                    payload={
                        "soil_moisture": state.soil_moisture,
                        "temperature": state.temperature,
                        "crop_height": state.crop_height,
                        "nutrient_level": state.nutrient_level,
                    },
                )
            )
        return self._current_state

    def get_current_state(self) -> Optional[SimulationState]:
        """Return the current simulation state, or None if not yet set."""
        return self._current_state

    def simulate_from_current(self, days: int) -> SimulationResult:
        """Simulate from the current ingested state.

        Args:
            days: Number of days to simulate.

        Returns:
            SimulationResult starting from the ingested state.

        Raises:
            ValueError: If no sensor data has been ingested yet.
        """
        if self._current_state is None:
            raise ValueError("No sensor data ingested; call ingest_sensor_data first")
        return self.simulate(self._current_state, days)

    def simulate(self, initial_state: SimulationState, days: int) -> SimulationResult:
        """Simulate crop growth over time."""
        if days < 0:
            raise ValueError("Days must be non-negative")

        state = SimulationState(
            soil_moisture=initial_state.soil_moisture,
            temperature=initial_state.temperature,
            crop_height=initial_state.crop_height,
            nutrient_level=initial_state.nutrient_level,
            pest_pressure=initial_state.pest_pressure,
            disease_pressure=initial_state.disease_pressure,
            weed_pressure=initial_state.weed_pressure,
        )

        history = []
        for _ in range(days):
            state = self._step(state)
            history.append(
                SimulationState(
                    soil_moisture=state.soil_moisture,
                    temperature=state.temperature,
                    crop_height=state.crop_height,
                    nutrient_level=state.nutrient_level,
                    pest_pressure=state.pest_pressure,
                    disease_pressure=state.disease_pressure,
                    weed_pressure=state.weed_pressure,
                )
            )

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
        biotic_factor = self._biotic_stress_factor(state)

        stress = temp_factor * water_factor * nutrient_factor * biotic_factor

        growth = (
            self.growth_rate
            * state.crop_height
            * (1 - state.crop_height / self.max_crop_height)
            * stress
        )
        new_height = min(self.max_crop_height, state.crop_height + growth)

        # Soil moisture depletes
        new_moisture = max(0.0, state.soil_moisture - 0.02 * stress)

        # Pest pressure evolves (logistic growth)
        new_pest = state.pest_pressure
        if new_pest > 0:
            new_pest = min(1.0, new_pest + 0.05 * new_pest * (1 - new_pest))

        return SimulationState(
            soil_moisture=new_moisture,
            temperature=state.temperature,
            crop_height=new_height,
            nutrient_level=state.nutrient_level,
            pest_pressure=new_pest,
            disease_pressure=state.disease_pressure,
            weed_pressure=state.weed_pressure,
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

    def _biotic_stress_factor(self, state: SimulationState) -> float:
        """Biotic stress factor from pest/disease/weed pressure."""
        return 1.0 - max(state.pest_pressure, state.disease_pressure, state.weed_pressure)

    def simulate_with_weather(
        self,
        initial_state: SimulationState,
        days: int,
        weather_data: List[Dict[str, Any]],
    ) -> SimulationResult:
        """Simulate with time-varying weather data.

        Args:
            initial_state: Starting simulation state.
            days: Number of days to simulate.
            weather_data: List of daily weather dicts with keys:
                'temp_max', 'temp_min', 'precipitation', 'humidity'.

        Returns:
            SimulationResult with weather-driven simulation.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")
        if len(weather_data) < days:
            raise ValueError("Weather data must cover all simulation days")

        state = SimulationState(
            soil_moisture=initial_state.soil_moisture,
            temperature=initial_state.temperature,
            crop_height=initial_state.crop_height,
            nutrient_level=initial_state.nutrient_level,
            pest_pressure=initial_state.pest_pressure,
            disease_pressure=initial_state.disease_pressure,
            weed_pressure=initial_state.weed_pressure,
        )

        history: List[SimulationState] = []
        for i in range(days):
            w = weather_data[i]
            # Update temperature from weather
            state.temperature = (w.get("temp_max", 25.0) + w.get("temp_min", 15.0)) / 2.0
            # Update moisture from precipitation
            precip = w.get("precipitation", 0.0)
            if precip > 0:
                state.soil_moisture = min(1.0, state.soil_moisture + precip / 500.0)
            state = self._step(state)
            history.append(
                SimulationState(
                    soil_moisture=state.soil_moisture,
                    temperature=state.temperature,
                    crop_height=state.crop_height,
                    nutrient_level=state.nutrient_level,
                    pest_pressure=state.pest_pressure,
                    disease_pressure=state.disease_pressure,
                    weed_pressure=state.weed_pressure,
                )
            )

        total_growth = state.crop_height - initial_state.crop_height
        return SimulationResult(
            days_simulated=days,
            final_state=state,
            history=history,
            algorithm=self.algorithm,
            total_growth=total_growth,
        )

    def stream_simulate(
        self,
        initial_state: SimulationState,
        days: int,
        callback: Optional[Callable[[int, SimulationState], None]] = None,
    ) -> Iterator[SimulationState]:
        """Stream simulation results day by day.

        Args:
            initial_state: Starting simulation state.
            days: Number of days to simulate.
            callback: Optional callback(day, state) called each day.

        Yields:
            SimulationState for each day.
        """
        state = SimulationState(
            soil_moisture=initial_state.soil_moisture,
            temperature=initial_state.temperature,
            crop_height=initial_state.crop_height,
            nutrient_level=initial_state.nutrient_level,
            pest_pressure=initial_state.pest_pressure,
            disease_pressure=initial_state.disease_pressure,
            weed_pressure=initial_state.weed_pressure,
        )

        for day in range(days):
            state = self._step(state)
            if callback:
                callback(day, state)
            yield SimulationState(
                soil_moisture=state.soil_moisture,
                temperature=state.temperature,
                crop_height=state.crop_height,
                nutrient_level=state.nutrient_level,
                pest_pressure=state.pest_pressure,
                disease_pressure=state.disease_pressure,
                weed_pressure=state.weed_pressure,
            )

    def get_phenological_stages(self) -> List[str]:
        """Return phenological stages for the current crop type."""
        stages = {
            CropType.WHEAT: [
                "emergence",
                "tillering",
                "stem_extension",
                "heading",
                "flowering",
                "grain_filling",
                "maturity",
            ],
            CropType.CORN: ["emergence", "V6", "V12", "VT", "R1", "R3", "R6", "maturity"],
            CropType.RICE: [
                "emergence",
                "tillering",
                "panicle_initiation",
                "booting",
                "heading",
                "flowering",
                "maturity",
            ],
            CropType.SOYBEAN: ["emergence", "V3", "V6", "R1", "R3", "R5", "R7", "maturity"],
            CropType.POTATO: [
                "emergence",
                "vegetative",
                "tuber_initiation",
                "tuber_bulking",
                "maturity",
            ],
            CropType.TOMATO: [
                "emergence",
                "vegetative",
                "flowering",
                "fruit_set",
                "fruit_development",
                "harvest",
            ],
            CropType.GENERIC: ["emergence", "vegetative", "reproductive", "maturity"],
        }
        return stages.get(self.crop_type, stages[CropType.GENERIC])

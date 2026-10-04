"""Water balance and irrigation model for agricultural digital twin.

Models soil water dynamics including precipitation, irrigation,
evapotranspiration, drainage, and runoff. Provides irrigation
scheduling and water stress assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WaterBalanceState:
    """Current water balance state of a field."""

    soil_moisture: float  # Current soil moisture (m³/m³)
    field_capacity: float = 0.30  # Field capacity (m³/m³)
    wilting_point: float = 0.10  # Permanent wilting point (m³/m³)
    root_depth: float = 0.5  # Root zone depth (m)
    precipitation: float = 0.0  # Today's precipitation (mm)
    irrigation: float = 0.0  # Today's irrigation (mm)
    evapotranspiration: float = 0.0  # Today's ET (mm)
    drainage: float = 0.0  # Today's drainage (mm)
    runoff: float = 0.0  # Today's runoff (mm)


@dataclass
class IrrigationSchedule:
    """Irrigation schedule recommendation."""

    should_irrigate: bool
    amount_mm: float
    reason: str
    priority: int  # 1 = highest, 5 = lowest


@dataclass
class WaterBalanceResult:
    """Result of water balance simulation."""

    days_simulated: int
    final_state: WaterBalanceState
    history: List[WaterBalanceState]
    total_precipitation: float
    total_irrigation: float
    total_evapotranspiration: float
    total_drainage: float
    total_runoff: float
    water_stress_days: int


class WaterBalanceModel:
    """Water balance model for agricultural fields.

    Simulates daily soil water dynamics using a bucket model approach.
    Tracks water inputs (precipitation, irrigation) and outputs
    (evapotranspiration, drainage, runoff).
    """

    def __init__(
        self,
        field_capacity: float = 0.30,
        wilting_point: float = 0.10,
        root_depth: float = 0.5,
        runoff_coefficient: float = 0.15,
        drainage_rate: float = 0.05,
    ):
        self.field_capacity = field_capacity
        self.wilting_point = wilting_point
        self.root_depth = root_depth
        self.runoff_coefficient = runoff_coefficient
        self.drainage_rate = drainage_rate
        self.max_available_water = (field_capacity - wilting_point) * root_depth * 1000  # mm

    def simulate(
        self,
        initial_state: WaterBalanceState,
        daily_weather: List[dict],
        irrigation_schedule: Optional[List[float]] = None,
    ) -> WaterBalanceResult:
        """Simulate water balance over multiple days.

        Args:
            initial_state: Starting water balance state.
            daily_weather: List of daily weather dicts with keys:
                'precipitation' (mm), 'evapotranspiration' (mm).
            irrigation_schedule: Optional list of irrigation amounts (mm) per day.

        Returns:
            WaterBalanceResult with full simulation history.
        """
        if irrigation_schedule is None:
            irrigation_schedule = [0.0] * len(daily_weather)

        if len(irrigation_schedule) != len(daily_weather):
            raise ValueError("Irrigation schedule length must match weather data length")

        state = WaterBalanceState(
            soil_moisture=initial_state.soil_moisture,
            field_capacity=initial_state.field_capacity,
            wilting_point=initial_state.wilting_point,
            root_depth=initial_state.root_depth,
        )

        history: List[WaterBalanceState] = []
        total_precip = 0.0
        total_irr = 0.0
        total_et = 0.0
        total_drain = 0.0
        total_runoff = 0.0
        stress_days = 0

        for i, weather in enumerate(daily_weather):
            precip = weather.get("precipitation", 0.0)
            et = weather.get("evapotranspiration", 0.0)
            irr = irrigation_schedule[i]

            state = self._step(state, precip, et, irr)
            history.append(
                WaterBalanceState(
                    soil_moisture=state.soil_moisture,
                    field_capacity=state.field_capacity,
                    wilting_point=state.wilting_point,
                    root_depth=state.root_depth,
                    precipitation=precip,
                    irrigation=irr,
                    evapotranspiration=et,
                    drainage=state.drainage,
                    runoff=state.runoff,
                )
            )

            total_precip += precip
            total_irr += irr
            total_et += et
            total_drain += state.drainage
            total_runoff += state.runoff

            if self.water_stress_factor(state.soil_moisture) < 0.5:
                stress_days += 1

        return WaterBalanceResult(
            days_simulated=len(daily_weather),
            final_state=state,
            history=history,
            total_precipitation=total_precip,
            total_irrigation=total_irr,
            total_evapotranspiration=total_et,
            total_drainage=total_drain,
            total_runoff=total_runoff,
            water_stress_days=stress_days,
        )

    def _step(
        self,
        state: WaterBalanceState,
        precipitation: float,
        evapotranspiration: float,
        irrigation: float,
    ) -> WaterBalanceState:
        """Advance water balance by one day."""
        # Convert mm to m³/m³ for soil moisture
        mm_to_m3m3 = 1.0 / (self.root_depth * 1000.0)

        # Water inputs
        total_input = precipitation + irrigation

        # Runoff: excess water that doesn't infiltrate
        runoff = total_input * self.runoff_coefficient if total_input > 5.0 else 0.0
        infiltrated = total_input - runoff

        # Update soil moisture
        new_moisture = state.soil_moisture + infiltrated * mm_to_m3m3

        # Drainage: water above field capacity drains
        drainage = 0.0
        if new_moisture > self.field_capacity:
            excess = (new_moisture - self.field_capacity) / mm_to_m3m3
            drainage = excess * self.drainage_rate
            new_moisture -= drainage * mm_to_m3m3

        # Evapotranspiration reduces soil moisture
        et_mm = min(
            evapotranspiration, (new_moisture - self.wilting_point) * self.root_depth * 1000.0
        )
        et_mm = max(0.0, et_mm)
        new_moisture -= et_mm * mm_to_m3m3

        # Clamp to physical bounds
        new_moisture = max(0.0, min(1.0, new_moisture))

        return WaterBalanceState(
            soil_moisture=new_moisture,
            field_capacity=state.field_capacity,
            wilting_point=state.wilting_point,
            root_depth=state.root_depth,
            precipitation=precipitation,
            irrigation=irrigation,
            evapotranspiration=et_mm,
            drainage=drainage,
            runoff=runoff,
        )

    def water_stress_factor(self, soil_moisture: float) -> float:
        """Calculate water stress factor (0-1).

        1.0 = no stress, 0.0 = severe stress.
        Linear interpolation between wilting point and field capacity.
        """
        if soil_moisture <= self.wilting_point:
            return 0.0
        if soil_moisture >= self.field_capacity:
            return 1.0
        return (soil_moisture - self.wilting_point) / (self.field_capacity - self.wilting_point)

    def irrigation_recommendation(
        self,
        current_moisture: float,
        forecast_et: float,
        days_ahead: int = 3,
    ) -> IrrigationSchedule:
        """Generate irrigation recommendation based on current conditions.

        Args:
            current_moisture: Current soil moisture (m³/m³).
            forecast_et: Forecasted evapotranspiration rate (mm/day).
            days_ahead: Number of days to look ahead.

        Returns:
            IrrigationSchedule with recommendation.
        """
        stress = self.water_stress_factor(current_moisture)
        water_deficit = (self.field_capacity - current_moisture) * self.root_depth * 1000.0
        future_et = forecast_et * days_ahead

        if stress >= 0.8:
            return IrrigationSchedule(
                should_irrigate=False,
                amount_mm=0.0,
                reason="Soil moisture adequate",
                priority=5,
            )

        if stress < 0.3 or water_deficit > future_et * 1.5:
            amount = min(water_deficit * 1.2, self.max_available_water * 0.8)
            return IrrigationSchedule(
                should_irrigate=True,
                amount_mm=round(amount, 1),
                reason="Severe water stress predicted",
                priority=1,
            )

        if stress < 0.5:
            amount = water_deficit * 0.8
            return IrrigationSchedule(
                should_irrigate=True,
                amount_mm=round(amount, 1),
                reason="Moderate water stress",
                priority=3,
            )

        return IrrigationSchedule(
            should_irrigate=False,
            amount_mm=0.0,
            reason="Monitor conditions",
            priority=4,
        )

    def calculate_water_use_efficiency(
        self,
        total_irrigation: float,
        total_et: float,
    ) -> float:
        """Calculate water use efficiency (WUE).

        WUE = ET / irrigation. Higher is better.
        Returns 0 if no irrigation was applied.
        """
        if total_irrigation <= 0:
            return 0.0
        return total_et / total_irrigation

"""Weed competition model for agricultural digital twin.

Models density-dependent resource preemption by weeds, including
light, water, and nutrient competition. Estimates yield loss from
weed-crop competition and simulates weed population dynamics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WeedState:
    """Current state of a weed population."""

    density: float = 0.0  # Weed density (plants/m²)
    biomass_g_m2: float = 0.0  # Weed biomass (g/m²)
    growth_rate: float = 0.05  # Intrinsic growth rate
    carrying_capacity: float = 200.0  # Maximum density (plants/m²)


@dataclass
class CompetitionResult:
    """Result of competition calculation."""

    yield_loss_fraction: float  # Fraction of yield lost (0-1)
    yield_loss_percent: float  # Percent yield lost (0-100)
    competition_index: float  # Combined competition index (0-1)
    light_factor: float  # Light competition factor (0-1)
    water_factor: float  # Water competition factor (0-1)
    nutrient_factor: float  # Nutrient competition factor (0-1)


@dataclass
class WeedSimulationResult:
    """Result of weed population simulation."""

    days_simulated: int
    final_state: WeedState
    history: List[WeedState]
    peak_density: float
    total_growth: float


class WeedCompetitionModel:
    """Weed competition model for agricultural digital twin.

    Models density-dependent resource preemption:
    - Light competition (canopy shading)
    - Water competition (soil moisture depletion)
    - Nutrient competition (N, P, K uptake)
    - Combined competition index
    - Yield loss estimation
    - Weed population dynamics (logistic growth)
    - Economic threshold calculation
    """

    def __init__(
        self,
        carrying_capacity: float = 200.0,
        growth_rate: float = 0.05,
        light_competition_coeff: float = 0.003,
        water_competition_coeff: float = 0.004,
        nutrient_competition_coeff: float = 0.002,
        crop_lai_factor: float = 0.5,
    ):
        self.carrying_capacity = carrying_capacity
        self.growth_rate = growth_rate
        self.light_competition_coeff = light_competition_coeff
        self.water_competition_coeff = water_competition_coeff
        self.nutrient_competition_coeff = nutrient_competition_coeff
        self.crop_lai_factor = crop_lai_factor

    def light_competition(self, weed_density: float, crop_lai: float) -> float:
        """Calculate light competition factor (0-1).

        Weeds shade the crop, reducing light availability.
        Higher crop LAI reduces weed light competition.

        Args:
            weed_density: Weed density (plants/m²).
            crop_lai: Crop leaf area index (m²/m²).

        Returns:
            Light competition factor (0-1).
        """
        if weed_density <= 0.0:
            return 0.0
        # Competition increases with weed density, decreases with crop LAI
        raw = self.light_competition_coeff * weed_density / (1.0 + self.crop_lai_factor * crop_lai)
        return min(1.0, raw)

    def water_competition(self, weed_density: float, soil_moisture: float) -> float:
        """Calculate water competition factor (0-1).

        Weeds compete for soil water. Competition is more severe
        in drier soil conditions.

        Args:
            weed_density: Weed density (plants/m²).
            soil_moisture: Soil moisture (m³/m³).

        Returns:
            Water competition factor (0-1).
        """
        if weed_density <= 0.0:
            return 0.0
        # Competition increases with weed severity and dryness
        moisture_factor = max(0.0, 1.0 - soil_moisture / 0.3)  # Normalized to field capacity
        raw = self.water_competition_coeff * weed_density * (0.5 + 0.5 * moisture_factor)
        return min(1.0, raw)

    def nutrient_competition(self, weed_density: float, soil_nitrogen: float) -> float:
        """Calculate nutrient competition factor (0-1).

        Weeds compete for soil nutrients (primarily nitrogen).
        Competition is more severe in nutrient-poor soil.

        Args:
            weed_density: Weed density (plants/m²).
            soil_nitrogen: Soil nitrogen content (kg/ha).

        Returns:
            Nutrient competition factor (0-1).
        """
        if weed_density <= 0.0:
            return 0.0
        # Competition increases with weed severity and N deficiency
        n_factor = max(0.0, 1.0 - soil_nitrogen / 150.0)  # Normalized to sufficient N
        raw = self.nutrient_competition_coeff * weed_density * (0.5 + 0.5 * n_factor)
        return min(1.0, raw)

    def competition_index(self, weed_density: float, crop_lai: float) -> float:
        """Calculate combined competition index (0-1).

        Weighted average of light, water, and nutrient competition.

        Args:
            weed_density: Weed density (plants/m²).
            crop_lai: Crop leaf area index (m²/m²).

        Returns:
            Combined competition index (0-1).
        """
        if weed_density <= 0.0:
            return 0.0
        # Use default soil conditions for combined index
        light = self.light_competition(weed_density, crop_lai)
        water = self.water_competition(weed_density, 0.2)
        nutrient = self.nutrient_competition(weed_density, 80.0)
        # Weighted average: light is most important
        return min(1.0, 0.5 * light + 0.3 * water + 0.2 * nutrient)

    def yield_loss(self, weed_density: float, crop_lai: float) -> CompetitionResult:
        """Estimate yield loss from weed competition.

        Args:
            weed_density: Weed density (plants/m²).
            crop_lai: Crop leaf area index (m²/m²).

        Returns:
            CompetitionResult with yield loss and competition factors.
        """
        if weed_density <= 0.0:
            return CompetitionResult(
                yield_loss_fraction=0.0,
                yield_loss_percent=0.0,
                competition_index=0.0,
                light_factor=0.0,
                water_factor=0.0,
                nutrient_factor=0.0,
            )

        light = self.light_competition(weed_density, crop_lai)
        water = self.water_competition(weed_density, 0.2)
        nutrient = self.nutrient_competition(weed_density, 80.0)
        index = min(1.0, 0.5 * light + 0.3 * water + 0.2 * nutrient)

        # Yield loss follows a sigmoid relationship with competition index
        # At low competition, loss is minimal; at high competition, loss approaches 1
        loss_fraction = index**1.5  # Non-linear: small competition = small loss

        return CompetitionResult(
            yield_loss_fraction=min(1.0, loss_fraction),
            yield_loss_percent=min(100.0, loss_fraction * 100.0),
            competition_index=index,
            light_factor=light,
            water_factor=water,
            nutrient_factor=nutrient,
        )

    def simulate(self, initial_state: WeedState, days: int) -> WeedSimulationResult:
        """Simulate weed population dynamics over time.

        Uses logistic growth: dN/dt = r * N * (1 - N/K)

        Args:
            initial_state: Starting weed state.
            days: Number of days to simulate.

        Returns:
            WeedSimulationResult with full simulation history.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")

        state = WeedState(
            density=initial_state.density,
            biomass_g_m2=initial_state.biomass_g_m2,
            growth_rate=initial_state.growth_rate,
            carrying_capacity=initial_state.carrying_capacity,
        )

        history: List[WeedState] = []
        peak_density = state.density

        for _ in range(days):
            # Logistic growth
            if state.density > 0.0 and state.carrying_capacity > 0.0:
                growth = (
                    state.growth_rate
                    * state.density
                    * (1.0 - state.density / state.carrying_capacity)
                )
                state.density += growth

            # Biomass increases with density
            state.biomass_g_m2 += state.density * 0.1  # Simplified biomass accumulation

            # Clamp to carrying capacity
            state.density = min(state.density, state.carrying_capacity)
            state.density = max(0.0, state.density)

            peak_density = max(peak_density, state.density)

            history.append(
                WeedState(
                    density=state.density,
                    biomass_g_m2=state.biomass_g_m2,
                    growth_rate=state.growth_rate,
                    carrying_capacity=state.carrying_capacity,
                )
            )

        return WeedSimulationResult(
            days_simulated=days,
            final_state=state,
            history=history,
            peak_density=peak_density,
            total_growth=state.density - initial_state.density,
        )

    def economic_threshold(self, crop_value: float, control_cost: float) -> float:
        """Calculate economic threshold weed density.

        The economic threshold is the weed density at which the cost
        of control equals the value of yield loss prevented.

        Args:
            crop_value: Crop value ($/ha).
            control_cost: Weed control cost ($/ha).

        Returns:
            Economic threshold density (plants/m²).
        """
        if crop_value <= 0.0 or control_cost <= 0.0:
            return 0.0
        # Simplified: threshold = control_cost / (crop_value * loss_per_weed)
        # Assume each weed causes ~0.1% yield loss at economic threshold
        loss_per_weed = 0.001
        threshold = control_cost / (crop_value * loss_per_weed)
        return max(0.0, threshold)

    def critical_period(self, crop_type: str) -> Tuple[int, int]:
        """Get the critical period of weed competition for a crop.

        The critical period is the growth stage during which weed
        competition causes the greatest yield loss.

        Args:
            crop_type: Crop type (e.g., "corn", "wheat", "soybean").

        Returns:
            Tuple of (start_day, end_day) relative to planting.
        """
        # Critical periods from literature (days after planting)
        periods: Dict[str, Tuple[int, int]] = {
            "corn": (14, 42),
            "wheat": (20, 60),
            "soybean": (21, 49),
            "rice": (20, 50),
            "potato": (28, 56),
            "tomato": (21, 49),
        }
        return periods.get(crop_type.lower(), (14, 42))

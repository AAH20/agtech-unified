"""Pest population dynamics model for agricultural digital twin.

Models pest population growth using logistic dynamics with temperature
dependence, carrying capacity limits, pesticide intervention, degree-day
driven development, and predator-prey interactions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class PestState:
    """Current state of a pest population."""

    population: float = 0.0  # Normalized population (0-1)
    carrying_capacity: float = 1.0  # Maximum sustainable population (0-1)
    temperature: float = 25.0  # Current temperature (°C)
    pesticide_efficacy: float = 0.0  # Last pesticide efficacy applied (0-1)
    predator_population: float = 0.0  # Normalized predator population (0-1)
    degree_days_accumulated: float = 0.0  # Accumulated degree-days


@dataclass
class PestSimulationResult:
    """Result of pest population simulation."""

    days_simulated: int
    final_state: PestState
    history: List[PestState]
    peak_population: float
    total_growth: float


class PestModel:
    """Pest population dynamics model.

    Simulates pest population growth using logistic dynamics with:
    - Temperature-dependent growth rate
    - Carrying capacity limits
    - Pesticide intervention effects
    - Extinction threshold
    - Degree-day driven development
    - Temperature-dependent development rates
    - Predator-prey interactions
    """

    def __init__(
        self,
        growth_rate: float = 0.3,
        optimal_temp: float = 25.0,
        temp_tolerance: float = 15.0,
        extinction_threshold: float = 0.001,
        base_temp: float = 10.0,
        max_temp: float = 35.0,
        degree_day_threshold: float = 150.0,
        predator_efficiency: float = 0.0,
        predator_growth_rate: float = 0.05,
        predator_death_rate: float = 0.1,
    ):
        self.growth_rate = growth_rate
        self.optimal_temp = optimal_temp
        self.temp_tolerance = temp_tolerance
        self.extinction_threshold = extinction_threshold
        self.base_temp = base_temp
        self.max_temp = max_temp
        self.degree_day_threshold = degree_day_threshold
        self.predator_efficiency = predator_efficiency
        self.predator_growth_rate = predator_growth_rate
        self.predator_death_rate = predator_death_rate

    def simulate(
        self,
        initial_state: PestState,
        days: int,
        pesticide_efficacy: float = 0.0,
    ) -> PestSimulationResult:
        """Simulate pest population over time.

        Args:
            initial_state: Starting pest state.
            days: Number of days to simulate.
            pesticide_efficacy: Pesticide efficacy applied daily (0-1).

        Returns:
            PestSimulationResult with full simulation history.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")

        state = PestState(
            population=initial_state.population,
            carrying_capacity=initial_state.carrying_capacity,
            temperature=initial_state.temperature,
            predator_population=initial_state.predator_population,
            degree_days_accumulated=initial_state.degree_days_accumulated,
        )

        history: List[PestState] = []
        peak_pop = state.population

        for _ in range(days):
            state = self._step(state, pesticide_efficacy)
            history.append(
                PestState(
                    population=state.population,
                    carrying_capacity=state.carrying_capacity,
                    temperature=state.temperature,
                    predator_population=state.predator_population,
                    degree_days_accumulated=state.degree_days_accumulated,
                )
            )
            peak_pop = max(peak_pop, state.population)

        return PestSimulationResult(
            days_simulated=days,
            final_state=state,
            history=history,
            peak_population=peak_pop,
            total_growth=state.population - initial_state.population,
        )

    def _step(self, state: PestState, pesticide_efficacy: float) -> PestState:
        """Advance pest population by one day."""
        temp_factor = self.temperature_factor(state.temperature)

        # Logistic growth: dN/dt = r * N * (1 - N/K) * temp_factor
        growth = (
            self.growth_rate
            * state.population
            * (1.0 - state.population / max(state.carrying_capacity, 1e-9))
            * temp_factor
        )
        new_pop = state.population + growth

        # Apply pesticide effect
        if pesticide_efficacy > 0.0:
            new_pop *= 1.0 - pesticide_efficacy

        # Apply predation effect
        if self.predator_efficiency > 0.0 and state.predator_population > 0.0:
            predation = self.predator_efficiency * state.population * state.predator_population
            new_pop -= predation

        # Extinction threshold
        if new_pop < self.extinction_threshold:
            new_pop = 0.0

        # Clamp to carrying capacity and non-negative
        new_pop = max(0.0, min(new_pop, state.carrying_capacity))

        # Update predator population
        new_predators = self._update_predators(state, new_pop)

        # Accumulate degree-days
        dd = self.accumulate_degree_days(state.temperature, 1)
        new_dd = state.degree_days_accumulated + dd

        return PestState(
            population=new_pop,
            carrying_capacity=state.carrying_capacity,
            temperature=state.temperature,
            predator_population=new_predators,
            degree_days_accumulated=new_dd,
        )

    def _update_predators(self, state: PestState, new_pest_pop: float) -> float:
        """Update predator population based on prey availability.

        Predator dynamics: dP/dt = (efficiency * prey - death_rate) * P
        """
        if self.predator_growth_rate <= 0.0:
            return state.predator_population

        # Predator growth proportional to prey consumption
        growth = self.predator_growth_rate * new_pest_pop * state.predator_population
        # Natural death
        death = self.predator_death_rate * state.predator_population
        new_pred = state.predator_population + growth - death
        return max(0.0, min(1.0, new_pred))

    def temperature_factor(self, temperature: float) -> float:
        """Calculate temperature effect on pest growth (0-1).

        Gaussian response centered at optimal_temp.
        """
        if temperature < 0.0 or temperature > 45.0:
            return 0.0
        return math.exp(-((temperature - self.optimal_temp) ** 2) / (2.0 * self.temp_tolerance**2))

    def accumulate_degree_days(self, temperature: float, days: int) -> float:
        """Accumulate degree-days over a period.

        Degree-days = max(0, T - T_base) * days

        Args:
            temperature: Mean temperature (°C).
            days: Number of days.

        Returns:
            Accumulated degree-days.
        """
        if temperature <= self.base_temp:
            return 0.0
        return (temperature - self.base_temp) * days

    def development_rate(self, temperature: float) -> float:
        """Calculate temperature-dependent development rate (0-1).

        Uses a thermal response curve: zero below base_temp and above max_temp,
        peaking at optimal_temp. Asymmetric curve with steeper decline above
        optimal (insects develop faster when warm but die quickly when too hot).

        Args:
            temperature: Current temperature (°C).

        Returns:
            Development rate as a fraction (0-1).
        """
        if temperature <= self.base_temp or temperature >= self.max_temp:
            return 0.0

        if temperature <= self.optimal_temp:
            # Linear increase from base to optimal
            return (temperature - self.base_temp) / (self.optimal_temp - self.base_temp)
        else:
            # Linear decrease from optimal to max
            return (self.max_temp - temperature) / (self.max_temp - self.optimal_temp)

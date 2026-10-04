"""Disease progression SEIR model for agricultural digital twin.

Models plant disease spread using SEIR (Susceptible-Exposed-Infectious-Recovered)
compartmental dynamics with environmental stress factors, vaccination,
infection latency, sporulation, and host-pathogen interactions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DiseaseState:
    """Current state of disease progression (SEIR compartments).

    All values are normalized proportions (0-1) summing to 1.0.
    """

    susceptible: float = 1.0
    exposed: float = 0.0
    infectious: float = 0.0
    recovered: float = 0.0
    spore_load: float = 0.0  # Environmental spore load (0-1)
    lesion_count: float = 0.0  # Normalized lesion count on infected hosts (0-1)

    @property
    def total_population(self) -> float:
        """Total population across all compartments."""
        return self.susceptible + self.exposed + self.infectious + self.recovered


@dataclass
class DiseaseSimulationResult:
    """Result of disease progression simulation."""

    days_simulated: int
    final_state: DiseaseState
    history: List[DiseaseState]
    peak_infection: float
    total_infected: float


class DiseaseModel:
    """SEIR disease progression model for plant populations.

    Simulates disease spread with:
    - Temperature-dependent transmission
    - Humidity-dependent transmission
    - Vaccination (moves susceptible to recovered)
    - Basic reproduction number (R0) calculation
    - Infection latency (exposed compartment)
    - Sporulation (pathogen reproduction driving new infections)
    - Environmental triggers (leaf wetness, temperature window, humidity)
    - Host-pathogen dynamics (resistance, virulence, immunity waning)
    """

    def __init__(
        self,
        infection_rate: float = 0.5,  # beta: contact rate * transmission prob
        recovery_rate: float = 0.1,  # gamma: 1 / infectious period
        incubation_rate: float = 0.3,  # sigma: 1 / incubation period
        optimal_temp: float = 25.0,
        temp_tolerance: float = 10.0,
        optimal_humidity: float = 70.0,
        humidity_tolerance: float = 30.0,
        mean_latency_period: float | None = None,
        sporulation_rate: float = 0.0,
        sporulation_infection_efficiency: float = 0.0,
        leaf_wetness_threshold: float = 6.0,
        min_temp_for_infection: float = 0.0,
        max_temp_for_infection: float = 45.0,
        humidity_threshold: float = 0.0,
        host_resistance_default: float = 0.0,
        pathogen_virulence: float = 0.0,
        immunity_waning_rate: float = 0.0,
    ):
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.incubation_rate = (
            1.0 / mean_latency_period if mean_latency_period is not None else incubation_rate
        )
        self.optimal_temp = optimal_temp
        self.temp_tolerance = temp_tolerance
        self.optimal_humidity = optimal_humidity
        self.humidity_tolerance = humidity_tolerance
        self.mean_latency_period = mean_latency_period
        self.sporulation_rate = sporulation_rate
        self.sporulation_infection_efficiency = sporulation_infection_efficiency
        self.leaf_wetness_threshold = leaf_wetness_threshold
        self.min_temp_for_infection = min_temp_for_infection
        self.max_temp_for_infection = max_temp_for_infection
        self.humidity_threshold = humidity_threshold
        self.host_resistance_default = host_resistance_default
        self.pathogen_virulence = pathogen_virulence
        self.immunity_waning_rate = immunity_waning_rate

    def simulate(
        self,
        initial_state: DiseaseState,
        days: int,
        vaccination_rate: float = 0.0,
        temperature: float = 25.0,
        humidity: float = 70.0,
        leaf_wetness_hours: float | None = None,
        host_resistance: float | None = None,
    ) -> DiseaseSimulationResult:
        """Simulate disease progression over time.

        Args:
            initial_state: Starting SEIR state.
            days: Number of days to simulate.
            vaccination_rate: Daily vaccination rate (fraction of susceptible).
            temperature: Current temperature (°C).
            humidity: Current relative humidity (%).
            leaf_wetness_hours: Daily leaf wetness duration (hours).
            host_resistance: Host resistance level (0-1), overrides default.

        Returns:
            DiseaseSimulationResult with full simulation history.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")
        state = DiseaseState(
            susceptible=initial_state.susceptible,
            exposed=initial_state.exposed,
            infectious=initial_state.infectious,
            recovered=initial_state.recovered,
            spore_load=initial_state.spore_load,
            lesion_count=initial_state.lesion_count,
        )

        history: List[DiseaseState] = []
        peak_infection = state.infectious

        temp_factor = self.temperature_stress_factor(temperature)
        humidity_factor = self.humidity_stress_factor(humidity)
        env_factor = temp_factor * humidity_factor

        # Apply environmental triggers
        env_factor *= self._environmental_trigger_factor(temperature, humidity, leaf_wetness_hours)

        # Host resistance reduces effective infection rate
        resistance = (
            host_resistance if host_resistance is not None else self.host_resistance_default
        )
        effective_infection_rate = self.infection_rate * (1.0 - resistance)

        # Pathogen virulence amplifies transmission
        effective_infection_rate *= 1.0 + self.pathogen_virulence

        for _ in range(days):
            state = self._step(state, vaccination_rate, env_factor, effective_infection_rate)
            history.append(
                DiseaseState(
                    susceptible=state.susceptible,
                    exposed=state.exposed,
                    infectious=state.infectious,
                    recovered=state.recovered,
                    spore_load=state.spore_load,
                    lesion_count=state.lesion_count,
                )
            )
            peak_infection = max(peak_infection, state.infectious)

        total_infected = state.infectious + state.recovered

        return DiseaseSimulationResult(
            days_simulated=days,
            final_state=state,
            history=history,
            peak_infection=peak_infection,
            total_infected=total_infected,
        )

    def _step(
        self,
        state: DiseaseState,
        vaccination_rate: float,
        env_factor: float,
        effective_infection_rate: float,
    ) -> DiseaseState:
        """Advance disease progression by one day."""
        # Force of infection from contact transmission
        lambda_t = effective_infection_rate * env_factor * state.infectious

        # Force of infection from sporulation
        if self.sporulation_rate > 0.0 and self.sporulation_infection_efficiency > 0.0:
            spore_infection = (
                self.sporulation_rate * self.sporulation_infection_efficiency * state.infectious
            )
            lambda_t += spore_infection

        # SEIR transitions
        new_exposed = lambda_t * state.susceptible
        new_infectious = self.incubation_rate * state.exposed
        new_recovered = self.recovery_rate * state.infectious

        # Vaccination: move susceptible to recovered
        vaccinated = vaccination_rate * state.susceptible

        # Immunity waning: recovered lose immunity
        waned = self.immunity_waning_rate * state.recovered

        # Update compartments
        new_s = state.susceptible - new_exposed - vaccinated + waned
        new_e = state.exposed + new_exposed - new_infectious
        new_i = state.infectious + new_infectious - new_recovered
        new_r = state.recovered + new_recovered + vaccinated - waned

        # Update spore load
        new_spore_load = state.spore_load
        if self.sporulation_rate > 0.0:
            new_spore_load += self.sporulation_rate * state.infectious
        new_spore_load = max(0.0, min(1.0, new_spore_load))

        # Update lesion count
        new_lesion_count = state.lesion_count
        if state.infectious > 0.0:
            new_lesion_count = max(0.0, min(1.0, state.lesion_count + 0.1 * state.infectious))

        # Clamp to non-negative
        new_s = max(0.0, new_s)
        new_e = max(0.0, new_e)
        new_i = max(0.0, new_i)
        new_r = max(0.0, new_r)

        return DiseaseState(
            susceptible=new_s,
            exposed=new_e,
            infectious=new_i,
            recovered=new_r,
            spore_load=new_spore_load,
            lesion_count=new_lesion_count,
        )

    def _environmental_trigger_factor(
        self,
        temperature: float,
        humidity: float,
        leaf_wetness_hours: float | None,
    ) -> float:
        """Calculate environmental trigger factor (0-1).

        Combines temperature window, humidity threshold, and leaf wetness
        into a single multiplier for disease transmission.
        """
        factor = 1.0

        # Temperature window trigger
        if temperature < self.min_temp_for_infection or temperature > self.max_temp_for_infection:
            factor *= 0.1  # Strongly reduced but not zero

        # Humidity threshold trigger
        if humidity < self.humidity_threshold:
            factor *= 0.5

        # Leaf wetness trigger
        if leaf_wetness_hours is not None and self.leaf_wetness_threshold > 0.0:
            if leaf_wetness_hours < self.leaf_wetness_threshold:
                factor *= 0.2  # Insufficient wetness for infection

        return factor

    def temperature_stress_factor(self, temperature: float) -> float:
        """Calculate temperature effect on disease transmission (0-1).

        Gaussian response centered at optimal_temp.
        """
        if temperature < 0.0 or temperature > 45.0:
            return 0.0
        return math.exp(-((temperature - self.optimal_temp) ** 2) / (2.0 * self.temp_tolerance**2))

    def humidity_stress_factor(self, humidity: float) -> float:
        """Calculate humidity effect on disease transmission (0-1).

        Gaussian response centered at optimal_humidity.
        """
        if humidity < 0.0 or humidity > 100.0:
            return 0.0
        return math.exp(
            -((humidity - self.optimal_humidity) ** 2) / (2.0 * self.humidity_tolerance**2)
        )

    def compute_r0(self) -> float:
        """Compute basic reproduction number (R0).

        R0 = infection_rate / recovery_rate (for simple SIR).
        For SEIR, R0 = infection_rate / recovery_rate (same in limit).
        """
        if self.recovery_rate <= 0.0:
            return float("inf")
        return self.infection_rate / self.recovery_rate

"""Nutrient cycling model for agricultural digital twin.

Models nitrogen (N), phosphorus (P), and potassium (K) dynamics
in the soil-plant system. Includes mineralization, immobilization,
plant uptake, leaching, and fertilizer application.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NutrientState:
    """Current nutrient state of a field."""

    nitrogen: float  # kg/ha (plant-available N)
    phosphorus: float  # kg/ha (plant-available P)
    potassium: float  # kg/ha (plant-available K)
    organic_matter: float  # % soil organic matter
    pH: float = 6.5  # Soil pH
    crop_uptake_n: float = 0.0  # Today's N uptake (kg/ha)
    crop_uptake_p: float = 0.0  # Today's P uptake (kg/ha)
    crop_uptake_k: float = 0.0  # Today's K uptake (kg/ha)
    leaching_n: float = 0.0  # Today's N leaching (kg/ha)
    mineralization_n: float = 0.0  # Today's N mineralization (kg/ha)
    immobilization_n: float = 0.0  # Today's N immobilization (kg/ha)


@dataclass
class FertilizerApplication:
    """Fertilizer application event."""

    nitrogen: float = 0.0  # kg/ha N
    phosphorus: float = 0.0  # kg/ha P
    potassium: float = 0.0  # kg/ha K
    day: int = 0  # Day of application


@dataclass
class NutrientBudget:
    """Nutrient budget summary."""

    total_n_applied: float
    total_p_applied: float
    total_k_applied: float
    total_n_uptake: float
    total_p_uptake: float
    total_k_uptake: float
    total_n_leached: float
    total_n_mineralized: float
    total_n_immobilized: float
    n_use_efficiency: float
    p_use_efficiency: float
    k_use_efficiency: float


@dataclass
class NutrientCyclingResult:
    """Result of nutrient cycling simulation."""

    days_simulated: int
    final_state: NutrientState
    history: List[NutrientState]
    budget: NutrientBudget


class NutrientCyclingModel:
    """Nutrient cycling model for agricultural fields.

    Simulates daily nutrient dynamics including:
    - Mineralization/immobilization of organic N
    - Plant nutrient uptake
    - Leaching losses
    - Fertilizer application effects
    """

    def __init__(
        self,
        mineralization_rate: float = 0.002,  # fraction per day
        immobilization_rate: float = 0.001,
        n_leaching_rate: float = 0.01,
        p_leaching_rate: float = 0.001,
        k_leaching_rate: float = 0.005,
        max_uptake_rate: float = 5.0,  # kg/ha/day
        optimal_ph: float = 6.5,
    ):
        self.mineralization_rate = mineralization_rate
        self.immobilization_rate = immobilization_rate
        self.n_leaching_rate = n_leaching_rate
        self.p_leaching_rate = p_leaching_rate
        self.k_leaching_rate = k_leaching_rate
        self.max_uptake_rate = max_uptake_rate
        self.optimal_ph = optimal_ph

    def simulate(
        self,
        initial_state: NutrientState,
        days: int,
        crop_demand: Optional[Dict[str, float]] = None,
        fertilizer_schedule: Optional[List[FertilizerApplication]] = None,
    ) -> NutrientCyclingResult:
        """Simulate nutrient cycling over time.

        Args:
            initial_state: Starting nutrient state.
            days: Number of days to simulate.
            crop_demand: Daily crop nutrient demand dict with keys
                'n', 'p', 'k' (kg/ha/day). Defaults to zero demand.
            fertilizer_schedule: List of fertilizer applications.

        Returns:
            NutrientCyclingResult with full simulation history.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")

        if crop_demand is None:
            crop_demand = {"n": 0.0, "p": 0.0, "k": 0.0}

        if fertilizer_schedule is None:
            fertilizer_schedule = []

        # Build fertilizer lookup by day
        fert_by_day: Dict[int, FertilizerApplication] = {}
        for fert in fertilizer_schedule:
            fert_by_day[fert.day] = fert

        state = NutrientState(
            nitrogen=initial_state.nitrogen,
            phosphorus=initial_state.phosphorus,
            potassium=initial_state.potassium,
            organic_matter=initial_state.organic_matter,
            pH=initial_state.pH,
        )

        history: List[NutrientState] = []
        total_n_uptake = 0.0
        total_p_uptake = 0.0
        total_k_uptake = 0.0
        total_n_leached = 0.0
        total_n_mineralized = 0.0
        total_n_immobilized = 0.0
        total_n_applied = 0.0
        total_p_applied = 0.0
        total_k_applied = 0.0

        for day in range(days):
            # Apply fertilizer if scheduled
            if day in fert_by_day:
                fert = fert_by_day[day]
                state.nitrogen += fert.nitrogen
                state.phosphorus += fert.phosphorus
                state.potassium += fert.potassium
                total_n_applied += fert.nitrogen
                total_p_applied += fert.phosphorus
                total_k_applied += fert.potassium

            # Mineralization: organic N → plant-available N
            mineralized = state.organic_matter * self.mineralization_rate * 10.0
            state.nitrogen += mineralized
            total_n_mineralized += mineralized

            # Immobilization: plant-available N → organic N
            immobilized = min(state.nitrogen * self.immobilization_rate, state.nitrogen * 0.1)
            state.nitrogen -= immobilized
            total_n_immobilized += immobilized

            # pH factor affects nutrient availability
            ph_factor = self._ph_factor(state.pH)

            # Plant uptake
            n_demand = crop_demand.get("n", 0.0) * ph_factor
            p_demand = crop_demand.get("p", 0.0) * ph_factor
            k_demand = crop_demand.get("k", 0.0) * ph_factor

            n_uptake = min(n_demand, state.nitrogen, self.max_uptake_rate)
            p_uptake = min(p_demand, state.phosphorus, self.max_uptake_rate * 0.2)
            k_uptake = min(k_demand, state.potassium, self.max_uptake_rate * 0.5)

            state.nitrogen -= n_uptake
            state.phosphorus -= p_uptake
            state.potassium -= k_uptake
            total_n_uptake += n_uptake
            total_p_uptake += p_uptake
            total_k_uptake += k_uptake

            # Leaching losses
            n_leached = state.nitrogen * self.n_leaching_rate
            p_leached = state.phosphorus * self.p_leaching_rate
            k_leached = state.potassium * self.k_leaching_rate

            state.nitrogen -= n_leached
            state.phosphorus -= p_leached
            state.potassium -= k_leached
            total_n_leached += n_leached

            # Clamp to non-negative
            state.nitrogen = max(0.0, state.nitrogen)
            state.phosphorus = max(0.0, state.phosphorus)
            state.potassium = max(0.0, state.potassium)

            # Record daily state
            history.append(
                NutrientState(
                    nitrogen=state.nitrogen,
                    phosphorus=state.phosphorus,
                    potassium=state.potassium,
                    organic_matter=state.organic_matter,
                    pH=state.pH,
                    crop_uptake_n=n_uptake,
                    crop_uptake_p=p_uptake,
                    crop_uptake_k=k_uptake,
                    leaching_n=n_leached,
                    mineralization_n=mineralized,
                    immobilization_n=immobilized,
                )
            )

        n_ue = (total_n_uptake / total_n_applied) if total_n_applied > 0 else 0.0
        p_ue = (total_p_uptake / total_p_applied) if total_p_applied > 0 else 0.0
        k_ue = (total_k_uptake / total_k_applied) if total_k_applied > 0 else 0.0

        budget = NutrientBudget(
            total_n_applied=total_n_applied,
            total_p_applied=total_p_applied,
            total_k_applied=total_k_applied,
            total_n_uptake=total_n_uptake,
            total_p_uptake=total_p_uptake,
            total_k_uptake=total_k_uptake,
            total_n_leached=total_n_leached,
            total_n_mineralized=total_n_mineralized,
            total_n_immobilized=total_n_immobilized,
            n_use_efficiency=n_ue,
            p_use_efficiency=p_ue,
            k_use_efficiency=k_ue,
        )

        return NutrientCyclingResult(
            days_simulated=days,
            final_state=state,
            history=history,
            budget=budget,
        )

    def _ph_factor(self, ph: float) -> float:
        """Calculate pH effect on nutrient availability (0-1).

        Optimal at self.optimal_ph, decreases as pH deviates.
        """
        deviation = abs(ph - self.optimal_ph)
        return max(0.0, math.exp(-deviation / 2.0))

    def nitrification_rate(self, temperature: float, soil_moisture: float) -> float:
        """Compute nitrification rate (NH4 → NO3).

        Temperature and moisture dependent. Optimal at 25-30°C and
        moderate moisture.

        Args:
            temperature: Soil temperature (°C).
            soil_moisture: Soil moisture (m³/m³).

        Returns:
            Nitrification rate (0-1).
        """
        # Temperature factor (Q10 = 2)
        if temperature < 5.0 or temperature > 45.0:
            temp_factor = 0.0
        else:
            temp_factor = 2.0 ** ((temperature - 25.0) / 10.0)
            temp_factor = min(1.0, temp_factor)

        # Moisture factor (optimal at 0.3 m³/m³)
        if soil_moisture < 0.05:
            moisture_factor = 0.0
        elif soil_moisture > 0.5:
            moisture_factor = 0.3  # Reduced under waterlogged conditions
        else:
            moisture_factor = 1.0 - abs(soil_moisture - 0.3) / 0.3

        return max(0.0, min(1.0, temp_factor * moisture_factor))

    def denitrification_rate(self, soil_moisture: float, temperature: float) -> float:
        """Compute denitrification rate (NO3 → N2/N2O).

        Occurs under anaerobic (waterlogged) conditions.

        Args:
            soil_moisture: Soil moisture (m³/m³).
            temperature: Soil temperature (°C).

        Returns:
            Denitrification rate (0-1).
        """
        # Denitrification requires high moisture (>0.4 m³/m³)
        if soil_moisture < 0.4:
            return 0.0

        # Moisture factor
        moisture_factor = min(1.0, (soil_moisture - 0.4) / 0.3)

        # Temperature factor
        if temperature < 5.0:
            temp_factor = 0.0
        elif temperature > 35.0:
            temp_factor = 0.5
        else:
            temp_factor = 1.0

        return max(0.0, min(1.0, moisture_factor * temp_factor))

    def split_n_pools(self, state: NutrientState) -> tuple:
        """Split total nitrogen into NH4 and NO3 pools.

        Args:
            state: Current nutrient state.

        Returns:
            Tuple of (nh4, no3) in kg/ha.
        """
        # Simplified: 20% NH4, 80% NO3 at typical conditions
        nh4 = state.nitrogen * 0.2
        no3 = state.nitrogen * 0.8
        return nh4, no3

    def nutrient_deficiency(self, state: NutrientState) -> Dict[str, bool]:
        """Check for nutrient deficiencies.

        Returns dict with 'n', 'p', 'k' keys, True if deficient.
        """
        return {
            "n": state.nitrogen < 20.0,
            "p": state.phosphorus < 5.0,
            "k": state.potassium < 30.0,
        }

    def recommend_fertilizer(
        self,
        state: NutrientState,
        target_n: float = 100.0,
        target_p: float = 20.0,
        target_k: float = 80.0,
    ) -> Dict[str, float]:
        """Recommend fertilizer application rates.

        Args:
            state: Current nutrient state.
            target_n: Target N level (kg/ha).
            target_p: Target P level (kg/ha).
            target_k: Target K level (kg/ha).

        Returns:
            Dict with recommended N, P, K application rates.
        """
        n_deficit = max(0.0, target_n - state.nitrogen)
        p_deficit = max(0.0, target_p - state.phosphorus)
        k_deficit = max(0.0, target_k - state.potassium)

        return {
            "n": round(n_deficit, 1),
            "p": round(p_deficit, 1),
            "k": round(k_deficit, 1),
        }

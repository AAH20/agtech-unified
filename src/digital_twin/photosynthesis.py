"""Photosynthesis model for agricultural digital twin.

Models C3 and C4 photosynthetic pathways, radiation use efficiency (RUE),
light response curves, temperature response, CO2 response, and the
Farquhar-von Caemmerer-Berry biochemical model of leaf photosynthesis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class C3C4Pathway(Enum):
    """Photosynthetic pathway types."""

    C3 = "c3"
    C4 = "c4"

    def default_rue(self) -> float:
        """Default radiation use efficiency (g/MJ PAR)."""
        return 2.5 if self is C3C4Pathway.C3 else 3.5

    def quantum_efficiency(self) -> float:
        """Maximum quantum efficiency (mol CO2 / mol photons)."""
        return 0.06 if self is C3C4Pathway.C3 else 0.04

    def co2_compensation_point(self) -> float:
        """CO2 compensation point (umol/mol)."""
        return 50.0 if self is C3C4Pathway.C3 else 5.0

    def optimal_temperature(self) -> float:
        """Optimal temperature for photosynthesis (°C)."""
        return 25.0 if self is C3C4Pathway.C3 else 35.0

    def min_temperature(self) -> float:
        """Minimum temperature for photosynthesis (°C)."""
        return 0.0 if self is C3C4Pathway.C3 else 10.0

    def max_temperature(self) -> float:
        """Maximum temperature for photosynthesis (°C)."""
        return 40.0 if self is C3C4Pathway.C3 else 45.0


@dataclass
class PhotosynthesisState:
    """Current state of the photosynthesis model."""

    lai: float = 3.0  # Leaf area index (m²/m²)
    biomass_g_m2: float = 0.0  # Accumulated biomass (g/m²)
    temperature: float = 25.0  # Current temperature (°C)
    par_mj_m2_day: float = 20.0  # Daily PAR (MJ/m²/day)
    water_stress_factor: float = 1.0  # Water stress factor (0-1)


@dataclass
class RUEResult:
    """Result of RUE-based biomass calculation."""

    biomass_g_m2: float  # Biomass produced (g/m²)
    apar_mj_m2: float  # Absorbed PAR (MJ/m²)
    rue_effective: float  # Effective RUE (g/MJ)


@dataclass
class FarquharResult:
    """Result of Farquhar biochemical model."""

    gross_photosynthesis: float  # Gross photosynthesis (umol CO2/m²/s)
    net_photosynthesis: float  # Net photosynthesis (umol CO2/m²/s)
    aj: float  # Light-limited rate (umol CO2/m²/s)
    ac: float  # Rubisco-limited rate (umol CO2/m²/s)
    limiting_process: str  # "light" or "rubisco"


@dataclass
class PhotosynthesisResult:
    """Result of daily photosynthesis simulation."""

    days_simulated: int
    final_state: PhotosynthesisState
    history: List[PhotosynthesisState]
    total_biomass_g_m2: float
    mean_daily_biomass_g_m2: float


class PhotosynthesisModel:
    """Photosynthesis model for agricultural digital twin.

    Implements:
    - C3/C4 pathway-specific parameters
    - Radiation use efficiency (RUE) biomass accumulation
    - Light response curve (rectangular hyperbola)
    - Temperature response (asymmetric Gaussian)
    - CO2 response factor
    - Farquhar-von Caemmerer-Berry biochemical model
    - Water stress effects
    - Daily simulation with LAI-dependent light interception
    """

    def __init__(
        self,
        extinction_coefficient: float = 0.65,
        dark_respiration_fraction: float = 0.015,
        co2_base_ppm: float = 400.0,
    ):
        self.extinction_coefficient = extinction_coefficient
        self.dark_respiration_fraction = dark_respiration_fraction
        self.co2_base_ppm = co2_base_ppm

    def rue_biomass(
        self,
        par_mj_m2_day: float,
        rue: float,
        light_interception: float,
        water_stress_factor: float = 1.0,
    ) -> RUEResult:
        """Calculate biomass from radiation use efficiency.

        Args:
            par_mj_m2_day: Daily photosynthetically active radiation (MJ/m²/day).
            rue: Radiation use efficiency (g/MJ PAR).
            light_interception: Fraction of light intercepted (0-1).
            water_stress_factor: Water stress factor (0-1, 1=no stress).

        Returns:
            RUEResult with biomass and absorbed PAR.
        """
        apar = par_mj_m2_day * light_interception
        rue_effective = rue * water_stress_factor
        biomass = apar * rue_effective
        return RUEResult(
            biomass_g_m2=biomass,
            apar_mj_m2=apar,
            rue_effective=rue_effective,
        )

    def quantum_efficiency(self, pathway: C3C4Pathway) -> float:
        """Get maximum quantum efficiency for a pathway.

        Args:
            pathway: C3 or C4 photosynthetic pathway.

        Returns:
            Maximum quantum efficiency (mol CO2 / mol photons).
        """
        return pathway.quantum_efficiency()

    def co2_compensation_point(self, pathway: C3C4Pathway) -> float:
        """Get CO2 compensation point for a pathway.

        Args:
            pathway: C3 or C4 photosynthetic pathway.

        Returns:
            CO2 compensation point (umol/mol).
        """
        return pathway.co2_compensation_point()

    def co2_response_factor(
        self, co2_ppm: float, pathway: C3C4Pathway, base_co2: float = 400.0
    ) -> float:
        """Calculate CO2 response factor (relative to base CO2).

        C3 plants show stronger response to elevated CO2 due to
        photorespiration suppression. C4 plants are less responsive
        because their CO2 concentrating mechanism already saturates Rubisco.

        Args:
            co2_ppm: Current CO2 concentration (ppm).
            pathway: C3 or C4 photosynthetic pathway.
            base_co2: Baseline CO2 concentration (ppm).

        Returns:
            Response factor (1.0 = no change from base).
        """
        if base_co2 <= 0:
            return 1.0
        # C3: ~30% increase at 2x CO2; C4: ~10% increase
        if pathway is C3C4Pathway.C3:
            return 1.0 + 0.30 * (co2_ppm - base_co2) / base_co2
        else:
            return 1.0 + 0.10 * (co2_ppm - base_co2) / base_co2

    def optimal_temperature(self, pathway: C3C4Pathway) -> float:
        """Get optimal temperature for photosynthesis.

        Args:
            pathway: C3 or C4 photosynthetic pathway.

        Returns:
            Optimal temperature (°C).
        """
        return pathway.optimal_temperature()

    def min_temperature(self, pathway: C3C4Pathway) -> float:
        """Get minimum temperature for photosynthesis.

        Args:
            pathway: C3 or C4 photosynthetic pathway.

        Returns:
            Minimum temperature (°C).
        """
        return pathway.min_temperature()

    def max_temperature(self, pathway: C3C4Pathway) -> float:
        """Get maximum temperature for photosynthesis.

        Args:
            pathway: C3 or C4 photosynthetic pathway.

        Returns:
            Maximum temperature (°C).
        """
        return pathway.max_temperature()

    def temperature_response(self, temperature: float, pathway: C3C4Pathway) -> float:
        """Calculate temperature response factor (0-1).

        Asymmetric response: steeper decline above optimum than below.

        Args:
            temperature: Current temperature (°C).
            pathway: C3 or C4 photosynthetic pathway.

        Returns:
            Temperature response factor (0-1).
        """
        t_opt = pathway.optimal_temperature()
        t_min = pathway.min_temperature()
        t_max = pathway.max_temperature()

        if temperature <= t_min or temperature >= t_max:
            return 0.0

        if temperature <= t_opt:
            # Linear increase from T_min to T_opt
            return (temperature - t_min) / (t_opt - t_min)
        else:
            # Linear decrease from T_opt to T_max
            return (t_max - temperature) / (t_max - t_opt)

    def light_response(self, par_umol_m2_s: float, pmax: float) -> float:
        """Calculate light response using rectangular hyperbola.

        Args:
            par_umol_m2_s: Photosynthetically active radiation (umol/m²/s).
            pmax: Maximum photosynthetic rate (umol CO2/m²/s).

        Returns:
            Photosynthetic rate (umol CO2/m²/s).
        """
        if par_umol_m2_s <= 0.0 or pmax <= 0.0:
            return 0.0
        alpha = 0.05  # Quantum yield (mol CO2 / mol photons)
        theta = 0.7  # Curvature parameter
        # Rectangular hyperbola:
        # A = (alpha*I + Pmax - sqrt((alpha*I + Pmax)^2 - 4*theta*alpha*I*Pmax)) / (2*theta)
        discriminant = (
            alpha * par_umol_m2_s + pmax
        ) ** 2 - 4.0 * theta * alpha * par_umol_m2_s * pmax
        if discriminant < 0:
            return pmax
        a = (alpha * par_umol_m2_s + pmax - math.sqrt(discriminant)) / (2.0 * theta)
        return max(0.0, a)

    def light_interception(self, lai: float) -> float:
        """Calculate light interception fraction from LAI.

        Uses Beer-Lambert law: f = 1 - exp(-k * LAI)

        Args:
            lai: Leaf area index (m²/m²).

        Returns:
            Light interception fraction (0-1).
        """
        if lai <= 0.0:
            return 0.0
        return 1.0 - math.exp(-self.extinction_coefficient * lai)

    def farquhar_gross_photosynthesis(
        self,
        par_umol_m2_s: float,
        ci_umol_mol: float,
        vcmax: float,
        jmax: float,
        temp_c: float,
    ) -> FarquharResult:
        """Calculate gross photosynthesis using Farquhar-von Caemmerer-Berry model.

        Args:
            par_umol_m2_s: Photosynthetically active radiation (umol/m²/s).
            ci_umol_mol: Intercellular CO2 concentration (umol/mol).
            vcmax: Maximum Rubisco carboxylation rate (umol/m²/s).
            jmax: Maximum electron transport rate (umol/m²/s).
            temp_c: Leaf temperature (°C).

        Returns:
            FarquharResult with gross/net photosynthesis and limiting process.
        """
        # Michaelis-Menten constants
        kc = 404.9  # umol/mol (Rubisco specificity for CO2)
        ko = 278.4  # mmol/mol (Rubisco specificity for O2)
        oi = 210.0  # mmol/mol (Intercellular O2 concentration)

        # CO2 compensation point (umol/mol)
        gamma_star = 0.5 * oi * kc / ko

        # Light-limited rate (Aj)
        # Aj = J * (Ci - Gamma*) / (4*Ci + 8*Gamma*)
        # where J is electron transport rate from light response
        alpha_q = 0.85  # Absorptance
        theta_j = 0.7  # Curvature
        # Electron transport rate from light
        discriminant = (
            alpha_q * par_umol_m2_s + jmax
        ) ** 2 - 4.0 * theta_j * alpha_q * par_umol_m2_s * jmax
        if discriminant < 0:
            j = jmax
        else:
            j = (alpha_q * par_umol_m2_s + jmax - math.sqrt(discriminant)) / (2.0 * theta_j)

        if ci_umol_mol <= gamma_star:
            aj = 0.0
        else:
            aj = j * (ci_umol_mol - gamma_star) / (4.0 * ci_umol_mol + 8.0 * gamma_star)

        # Rubisco-limited rate (Ac)
        # Ac = Vcmax * (Ci - Gamma*) / (Ci + Kc * (1 + Oi/Ko))
        if ci_umol_mol <= gamma_star:
            ac = 0.0
        else:
            km = kc * (1.0 + oi / ko)
            ac = vcmax * (ci_umol_mol - gamma_star) / (ci_umol_mol + km)

        # Gross photosynthesis is minimum of Aj and Ac
        if aj <= ac:
            gross = aj
            limiting = "light"
        else:
            gross = ac
            limiting = "rubisco"

        # Dark respiration (umol/m2/s)
        rd = self.dark_respiration_fraction * vcmax
        net = gross - rd

        return FarquharResult(
            gross_photosynthesis=gross,
            net_photosynthesis=max(0.0, net),
            aj=aj,
            ac=ac,
            limiting_process=limiting,
        )

    def simulate(
        self,
        initial_state: PhotosynthesisState,
        days: int,
        daily_par_mj_m2: float,
        pathway: C3C4Pathway = C3C4Pathway.C3,
        water_stress_factor: float = 1.0,
    ) -> PhotosynthesisResult:
        """Simulate photosynthesis over multiple days.

        Args:
            initial_state: Starting photosynthesis state.
            days: Number of days to simulate.
            daily_par_mj_m2: Daily PAR (MJ/m²/day).
            pathway: C3 or C4 photosynthetic pathway.
            water_stress_factor: Water stress factor (0-1).

        Returns:
            PhotosynthesisResult with full simulation history.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")

        state = PhotosynthesisState(
            lai=initial_state.lai,
            biomass_g_m2=initial_state.biomass_g_m2,
            temperature=initial_state.temperature,
            par_mj_m2_day=daily_par_mj_m2,
            water_stress_factor=water_stress_factor,
        )

        history: List[PhotosynthesisState] = []
        total_biomass = 0.0

        for _ in range(days):
            # Light interception from LAI
            f_int = self.light_interception(state.lai)

            # Temperature response
            t_resp = self.temperature_response(state.temperature, pathway)

            # RUE biomass
            rue = pathway.default_rue()
            rue_result = self.rue_biomass(
                par_mj_m2_day=daily_par_mj_m2,
                rue=rue,
                light_interception=f_int,
                water_stress_factor=water_stress_factor * t_resp,
            )

            state.biomass_g_m2 += rue_result.biomass_g_m2
            total_biomass += rue_result.biomass_g_m2

            history.append(
                PhotosynthesisState(
                    lai=state.lai,
                    biomass_g_m2=state.biomass_g_m2,
                    temperature=state.temperature,
                    par_mj_m2_day=daily_par_mj_m2,
                    water_stress_factor=water_stress_factor,
                )
            )

        mean_daily = total_biomass / days if days > 0 else 0.0

        return PhotosynthesisResult(
            days_simulated=days,
            final_state=state,
            history=history,
            total_biomass_g_m2=total_biomass,
            mean_daily_biomass_g_m2=mean_daily,
        )

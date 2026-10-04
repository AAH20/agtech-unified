"""Yield prediction model for agricultural digital twin (YIELD-001).

Estimates harvestable grain yield from total biomass via the harvest index
method, adjusts for water/nutrient stress, and derives grain quality metrics
(protein, moisture, test weight, thousand-kernel weight) with a market grade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GrainQuality:
    """Grain quality metrics at harvest."""

    protein_content: float  # % (e.g. wheat ~12%)
    moisture_content: float  # % wet basis
    test_weight: float  # kg/hl
    thousand_kernel_weight: float  # g
    grade: str  # Market grade label


@dataclass
class YieldPrediction:
    """Result of a yield prediction."""

    grain_yield: float  # kg/ha
    grain_yield_tons: float  # t/ha
    total_biomass: float  # kg/ha
    harvest_index: float  # effective (stress-adjusted) harvest index
    limiting_factor: str  # 'water', 'nutrient', or 'none'
    stress_adjusted: bool
    grain_quality: GrainQuality
    days_to_maturity: int


# Crop-specific parameters: base harvest index, typical biomass (kg/ha),
# typical protein (%), typical TKW (g), days to maturity.
_CROP_PARAMS: Dict[str, Dict[str, float]] = {
    "wheat": {
        "base_harvest_index": 0.45,
        "typical_biomass": 12000.0,
        "typical_protein": 12.0,
        "typical_tkw": 40.0,
        "days_to_maturity": 120,
    },
    "corn": {
        "base_harvest_index": 0.52,
        "typical_biomass": 18000.0,
        "typical_protein": 9.0,
        "typical_tkw": 300.0,
        "days_to_maturity": 140,
    },
    "rice": {
        "base_harvest_index": 0.50,
        "typical_biomass": 14000.0,
        "typical_protein": 8.0,
        "typical_tkw": 25.0,
        "days_to_maturity": 130,
    },
    "soybean": {
        "base_harvest_index": 0.40,
        "typical_biomass": 9000.0,
        "typical_protein": 38.0,
        "typical_tkw": 160.0,
        "days_to_maturity": 110,
    },
    "potato": {
        "base_harvest_index": 0.75,
        "typical_biomass": 20000.0,
        "typical_protein": 2.0,
        "typical_tkw": 200.0,
        "days_to_maturity": 100,
    },
    "tomato": {
        "base_harvest_index": 0.60,
        "typical_biomass": 25000.0,
        "typical_protein": 1.0,
        "typical_tkw": 5.0,
        "days_to_maturity": 90,
    },
    "generic": {
        "base_harvest_index": 0.45,
        "typical_biomass": 12000.0,
        "typical_protein": 10.0,
        "typical_tkw": 50.0,
        "days_to_maturity": 120,
    },
}


class YieldModel:
    """Yield prediction using harvest index × biomass with stress adjustment.

    The effective harvest index declines under water and nutrient stress
    (assimilate partitioning shifts away from grain). Grain quality metrics
    are derived from crop type and stress history.
    """

    def __init__(self, crop_type: str = "generic"):
        self.crop_type = crop_type.lower()
        params = _CROP_PARAMS.get(self.crop_type, _CROP_PARAMS["generic"])
        self.base_harvest_index = params["base_harvest_index"]
        self.typical_biomass = params["typical_biomass"]
        self.typical_protein = params["typical_protein"]
        self.typical_tkw = params["typical_tkw"]
        self.days_to_maturity = int(params["days_to_maturity"])

    def predict_yield(
        self,
        total_biomass: float,
        harvest_index: Optional[float] = None,
        water_stress: float = 1.0,
        nutrient_stress: float = 1.0,
    ) -> YieldPrediction:
        """Predict grain yield from total biomass.

        Args:
            total_biomass: Total above-ground biomass (kg/ha).
            harvest_index: Base harvest index; defaults to crop-specific value.
            water_stress: Water availability factor (0-1, 1 = no stress).
            nutrient_stress: Nutrient availability factor (0-1, 1 = no stress).

        Returns:
            YieldPrediction with grain yield, quality, and limiting factor.
        """
        if total_biomass < 0:
            raise ValueError("Total biomass must be non-negative")
        water_stress = self._clamp01(water_stress, "water_stress")
        nutrient_stress = self._clamp01(nutrient_stress, "nutrient_stress")

        hi = self.base_harvest_index if harvest_index is None else harvest_index
        if hi <= 0 or hi > 1.0:
            raise ValueError("Harvest index must be in (0, 1]")

        # Stress reduces effective harvest index multiplicatively.
        stress_factor = water_stress * nutrient_stress
        effective_hi = hi * stress_factor
        effective_hi = max(0.01, min(1.0, effective_hi))

        grain_yield = total_biomass * effective_hi

        limiting = "none"
        stress_adjusted = stress_factor < 0.999
        if stress_adjusted:
            limiting = "water" if water_stress <= nutrient_stress else "nutrient"

        quality = self._grain_quality(water_stress, nutrient_stress)

        return YieldPrediction(
            grain_yield=grain_yield,
            grain_yield_tons=grain_yield / 1000.0,
            total_biomass=total_biomass,
            harvest_index=effective_hi,
            limiting_factor=limiting,
            stress_adjusted=stress_adjusted,
            grain_quality=quality,
            days_to_maturity=self.days_to_maturity,
        )

    def estimate_biomass(self, crop_height: float, lai: float) -> float:
        """Estimate total above-ground biomass from crop height and LAI.

        Uses a simple allometric relation: biomass scales with height × LAI
        (leaf area drives photosynthetic capacity).

        Args:
            crop_height: Crop height (m).
            lai: Leaf area index (m²/m²).

        Returns:
            Estimated total biomass (kg/ha).
        """
        if crop_height < 0 or lai < 0:
            raise ValueError("Crop height and LAI must be non-negative")
        if lai == 0.0:
            return 0.0
        # Allometric coefficient calibrated so typical wheat (h=1m, LAI=3)
        # lands near its typical biomass.
        return 4000.0 * crop_height * lai

    def _grain_quality(self, water_stress: float, nutrient_stress: float) -> GrainQuality:
        """Derive grain quality metrics from stress history.

        Water stress concentrates protein (smaller grains, higher % protein)
        but reduces test weight and TKW. Nutrient stress reduces protein.
        """
        # Protein: water stress raises %, nutrient stress lowers it.
        protein = self.typical_protein * (1.0 + 0.3 * (1.0 - water_stress))
        protein *= 0.5 + 0.5 * nutrient_stress
        protein = max(0.5, protein)

        # Moisture: stressed crops dry down faster; clamp to safe storage range.
        moisture = 12.0 + 6.0 * (1.0 - water_stress)
        moisture = max(8.0, min(25.0, moisture))

        # Test weight and TKW decline under stress.
        stress = 1.0 - water_stress * nutrient_stress
        test_weight = 75.0 + 5.0 * stress
        tkw = self.typical_tkw * (0.7 + 0.3 * stress)

        grade = self._grade(protein, moisture, test_weight)
        return GrainQuality(
            protein_content=round(protein, 2),
            moisture_content=round(moisture, 2),
            test_weight=round(test_weight, 2),
            thousand_kernel_weight=round(tkw, 2),
            grade=grade,
        )

    def _grade(self, protein: float, moisture: float, test_weight: float) -> str:
        """Assign a market grade from quality metrics (wheat-oriented)."""
        if moisture > 20.0:
            return "feed"
        if protein >= 12.0 and test_weight >= 78.0:
            return "premium"
        if protein >= 10.0 and test_weight >= 74.0:
            return "standard"
        return "feed"

    @staticmethod
    def _clamp01(value: float, name: str) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

"""Decision support system for agricultural recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class FarmState:
    """Current state of the farm."""

    soil_moisture: float  # 0.0 to 1.0
    temperature: float  # Celsius
    crop_height: float  # meters
    nutrient_level: float  # 0.0 to 1.0
    pest_pressure: float  # 0.0 to 1.0


@dataclass
class RecommendationResult:
    """Decision support result."""

    recommendations: List[str]
    priority_score: float
    algorithm: str
    actions: List[str]


class DecisionEngine:
    """Rule-based decision support engine."""

    def __init__(self, algorithm: str = "rule_based"):
        self.algorithm = algorithm

    def recommend(self, state: FarmState) -> RecommendationResult:
        """Generate recommendations based on farm state."""
        recommendations = []
        actions = []
        priority_score = 0.0

        # Soil moisture check
        if state.soil_moisture < 0.2:
            recommendations.append("URGENT: Irrigate immediately — soil moisture critical")
            actions.append("irrigate")
            priority_score += 0.4
        elif state.soil_moisture < 0.4:
            recommendations.append("Irrigate soon — soil moisture low")
            actions.append("irrigate")
            priority_score += 0.2

        # Temperature check
        if state.temperature > 38:
            recommendations.append("Heat stress risk — provide shade or increase irrigation")
            actions.append("shade")
            priority_score += 0.3
        elif state.temperature < 5:
            recommendations.append("Frost risk — protect crops")
            actions.append("frost_protection")
            priority_score += 0.3

        # Nutrient check
        if state.nutrient_level < 0.2:
            recommendations.append("Apply fertilizer — nutrient level critical")
            actions.append("fertilize")
            priority_score += 0.2
        elif state.nutrient_level < 0.4:
            recommendations.append("Consider fertilization — nutrient level low")
            actions.append("fertilize")
            priority_score += 0.1

        # Pest check
        if state.pest_pressure > 0.7:
            recommendations.append("URGENT: Pest control needed — high pest pressure")
            actions.append("pest_control")
            priority_score += 0.4
        elif state.pest_pressure > 0.4:
            recommendations.append("Monitor pests — pressure elevated")
            actions.append("monitor_pests")
            priority_score += 0.1

        # Crop height check
        if state.crop_height < 0.1 and state.soil_moisture > 0.3:
            recommendations.append("Check germination — crop height low")
            actions.append("inspect")
            priority_score += 0.1

        priority_score = min(1.0, priority_score)

        return RecommendationResult(
            recommendations=recommendations,
            priority_score=priority_score,
            algorithm=self.algorithm,
            actions=actions,
        )

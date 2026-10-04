"""Decision support system for agricultural recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.integration.farm_state import FarmState  # re-export for backward compatibility

logger = logging.getLogger(__name__)


__all__ = ["FarmState", "RecommendationResult", "DecisionEngine"]


@dataclass
class RecommendationResult:
    """Decision support result."""

    recommendations: List[str]
    priority_score: float
    algorithm: str
    actions: List[str]
    tenant_id: Optional[str] = None
    confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


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

        # Confidence based on number of triggered rules (more rules = higher confidence)
        confidence = min(1.0, len(recommendations) * 0.2 + 0.3)

        # Feature importance: which metrics contributed most to the score
        feature_importance: Dict[str, float] = {}
        if state.soil_moisture < 0.4:
            feature_importance["soil_moisture"] = 0.4 if state.soil_moisture < 0.2 else 0.2
        if state.temperature > 38 or state.temperature < 5:
            feature_importance["temperature"] = 0.3
        if state.nutrient_level < 0.4:
            feature_importance["nutrient_level"] = 0.2 if state.nutrient_level < 0.2 else 0.1
        if state.pest_pressure > 0.4:
            feature_importance["pest_pressure"] = 0.4 if state.pest_pressure > 0.7 else 0.1
        if state.crop_height < 0.1 and state.soil_moisture > 0.3:
            feature_importance["crop_height"] = 0.1

        return RecommendationResult(
            recommendations=recommendations,
            priority_score=priority_score,
            algorithm=self.algorithm,
            actions=actions,
            tenant_id=state.tenant_id,
            confidence=confidence,
            feature_importance=feature_importance,
        )

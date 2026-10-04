"""Decision support system for agricultural recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.integration.event_bus import DomainEvent, EventBus, EventType
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
    """Decision support engine with rule-based and ML-based paths."""

    def __init__(self, algorithm: str = "rule_based", model=None):
        self.algorithm = algorithm
        self.model = model
        self._bus: Optional[EventBus] = None

    def recommend(self, state: FarmState) -> RecommendationResult:
        """Generate recommendations based on farm state."""
        if self.algorithm == "ensemble":
            return self._recommend_ensemble(state)
        if self.algorithm == "ml":
            return self._recommend_ml(state)
        return self._recommend_rule_based(state)

    def _recommend_rule_based(self, state: FarmState) -> RecommendationResult:
        """Generate recommendations using hardcoded rules."""
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

        result = RecommendationResult(
            recommendations=recommendations,
            priority_score=priority_score,
            algorithm=self.algorithm,
            actions=actions,
            tenant_id=state.tenant_id,
            confidence=confidence,
            feature_importance=feature_importance,
        )

        if self._bus and priority_score > 0.5:
            self._bus.publish(
                DomainEvent(
                    event_type=EventType.ALERT_TRIGGERED,
                    source="decision_support.recommender",
                    payload={
                        "priority_score": priority_score,
                        "recommendations": recommendations,
                        "actions": actions,
                    },
                )
            )

        return result

    def _recommend_ensemble(self, state: FarmState) -> RecommendationResult:
        """Generate recommendations using ensemble of rule-based and ML paths.

        Combines outputs from both approaches:
        - Merges recommendations (deduplicated)
        - Averages priority scores
        - Averages confidence scores
        - Merges feature importance (averaged)
        """
        if self.model is None:
            raise RuntimeError("Ensemble algorithm requires a model")
        if not self.model.is_trained:
            raise RuntimeError(f"Model {self.model.name} is not trained")

        # Get rule-based result
        rule_result = self._recommend_rule_based(state)

        # Get ML result
        ml_result = self._recommend_ml(state)

        # Merge recommendations (deduplicated, preserving order)
        seen = set()
        merged_recommendations = []
        for rec in rule_result.recommendations + ml_result.recommendations:
            if rec not in seen:
                seen.add(rec)
                merged_recommendations.append(rec)

        # Merge actions (deduplicated)
        seen_actions = set()
        merged_actions = []
        for action in rule_result.actions + ml_result.actions:
            if action not in seen_actions:
                seen_actions.add(action)
                merged_actions.append(action)

        # Average priority scores
        priority_score = (rule_result.priority_score + ml_result.priority_score) / 2.0
        priority_score = min(1.0, priority_score)

        # Average confidence scores
        confidence = (rule_result.confidence + ml_result.confidence) / 2.0
        confidence = min(1.0, confidence)

        # Merge feature importance (average values for shared keys)
        all_keys = set(rule_result.feature_importance.keys()) | set(
            ml_result.feature_importance.keys()
        )
        merged_importance: Dict[str, float] = {}
        for key in all_keys:
            rule_val = rule_result.feature_importance.get(key, 0.0)
            ml_val = ml_result.feature_importance.get(key, 0.0)
            merged_importance[key] = (rule_val + ml_val) / 2.0

        # Normalize feature importance to sum to 1.0
        total_importance = sum(merged_importance.values())
        if total_importance > 0:
            merged_importance = {k: v / total_importance for k, v in merged_importance.items()}

        result = RecommendationResult(
            recommendations=merged_recommendations,
            priority_score=priority_score,
            algorithm="ensemble",
            actions=merged_actions,
            tenant_id=state.tenant_id,
            confidence=confidence,
            feature_importance=merged_importance,
        )

        if self._bus and priority_score > 0.5:
            self._bus.publish(
                DomainEvent(
                    event_type=EventType.ALERT_TRIGGERED,
                    source="decision_support.recommender",
                    payload={
                        "priority_score": priority_score,
                        "recommendations": merged_recommendations,
                        "actions": merged_actions,
                    },
                )
            )

        return result

    def _recommend_ml(self, state: FarmState) -> RecommendationResult:
        """Generate recommendations using a trained ML model."""
        if self.model is None:
            raise RuntimeError("ML algorithm requires a model")
        if not self.model.is_trained:
            raise RuntimeError(f"Model {self.model.name} is not trained")

        # Build feature vector from farm state
        features = [
            state.soil_moisture,
            state.temperature,
            state.crop_height,
            state.nutrient_level,
            state.pest_pressure,
        ]
        prediction = self.model.predict(features)
        predicted_yield = prediction.value
        confidence = prediction.confidence

        # Priority score: inverse of predicted yield (lower yield = higher priority)
        priority_score = max(0.0, min(1.0, 1.0 - predicted_yield))

        # Feature importance from the model
        feature_importance = self.model.feature_importance()

        # Generate recommendations based on state thresholds
        recommendations = []
        actions = []

        # Soil moisture
        if state.soil_moisture < 0.2:
            recommendations.append("URGENT: Irrigate immediately — soil moisture critical")
            actions.append("irrigate")
        elif state.soil_moisture < 0.4:
            recommendations.append("Irrigate soon — soil moisture low")
            actions.append("irrigate")

        # Temperature
        if state.temperature > 38:
            recommendations.append("Heat stress risk — provide shade or increase irrigation")
            actions.append("shade")
        elif state.temperature < 5:
            recommendations.append("Frost risk — protect crops")
            actions.append("frost_protection")

        # Nutrient
        if state.nutrient_level < 0.2:
            recommendations.append("Apply fertilizer — nutrient level critical")
            actions.append("fertilize")
        elif state.nutrient_level < 0.4:
            recommendations.append("Consider fertilization — nutrient level low")
            actions.append("fertilize")

        # Pest
        if state.pest_pressure > 0.7:
            recommendations.append("URGENT: Pest control needed — high pest pressure")
            actions.append("pest_control")
        elif state.pest_pressure > 0.4:
            recommendations.append("Monitor pests — pressure elevated")
            actions.append("monitor_pests")

        # Crop height
        if state.crop_height < 0.1 and state.soil_moisture > 0.3:
            recommendations.append("Check germination — crop height low")
            actions.append("inspect")

        result = RecommendationResult(
            recommendations=recommendations,
            priority_score=priority_score,
            algorithm=self.algorithm,
            actions=actions,
            tenant_id=state.tenant_id,
            confidence=confidence,
            feature_importance=feature_importance,
        )

        if self._bus and priority_score > 0.5:
            self._bus.publish(
                DomainEvent(
                    event_type=EventType.ALERT_TRIGGERED,
                    source="decision_support.recommender",
                    payload={
                        "priority_score": priority_score,
                        "recommendations": recommendations,
                        "actions": actions,
                    },
                )
            )

        return result

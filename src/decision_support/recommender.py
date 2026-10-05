"""Decision support system for agricultural recommendations."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.integration.farm_state import FarmState  # re-export for backward compatibility

logger = logging.getLogger(__name__)


__all__ = [
    "FarmState",
    "RecommendationResult",
    "DecisionEngine",
    "FeedbackRecord",
    "FeedbackTracker",
]


@dataclass
class FeedbackRecord:
    """Outcome feedback for a recommendation."""

    recommendation_id: str
    action: str
    tenant_id: Optional[str] = None
    outcome: Optional[str] = None
    outcome_score: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackTracker:
    """Tracks outcome feedback for recommendations."""

    def __init__(self):
        self._records: Dict[str, List[FeedbackRecord]] = {}

    def record(
        self,
        recommendation_id: str,
        action: str,
        tenant_id: Optional[str] = None,
        outcome: Optional[str] = None,
        outcome_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """Record outcome feedback for a recommendation."""
        rec = FeedbackRecord(
            recommendation_id=recommendation_id,
            action=action,
            tenant_id=tenant_id,
            outcome=outcome,
            outcome_score=outcome_score,
            metadata=metadata or {},
        )
        if recommendation_id not in self._records:
            self._records[recommendation_id] = []
        self._records[recommendation_id].append(rec)
        return rec

    def get_history(
        self,
        recommendation_id: str,
        tenant_id: Optional[str] = None,
    ) -> List[FeedbackRecord]:
        """Get feedback history for a recommendation."""
        records = self._records.get(recommendation_id, [])
        if tenant_id is not None:
            records = [r for r in records if r.tenant_id == tenant_id]
        return list(records)

    def acceptance_rate(self, tenant_id: Optional[str] = None) -> float:
        """Calculate acceptance rate (accepted / total with outcomes)."""
        records = self._all_records(tenant_id)
        with_outcome = [r for r in records if r.outcome is not None]
        if not with_outcome:
            return 0.0
        accepted = [r for r in with_outcome if r.outcome == "accepted"]
        return len(accepted) / len(with_outcome)

    def average_outcome_score(self, tenant_id: Optional[str] = None) -> float:
        """Calculate average outcome score."""
        records = self._all_records(tenant_id)
        scores = [r.outcome_score for r in records if r.outcome_score is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def action_success_rate(self, action: str, tenant_id: Optional[str] = None) -> float:
        """Calculate success rate for a specific action."""
        records = self._all_records(tenant_id)
        action_records = [r for r in records if r.action == action and r.outcome is not None]
        if not action_records:
            return 0.0
        successes = [r for r in action_records if r.outcome == "success"]
        return len(successes) / len(action_records)

    def outcome_distribution(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        """Get distribution of outcomes."""
        records = self._all_records(tenant_id)
        dist: Dict[str, int] = {}
        for r in records:
            if r.outcome is not None:
                dist[r.outcome] = dist.get(r.outcome, 0) + 1
        return dist

    def top_performing_actions(
        self,
        n: int = 5,
        tenant_id: Optional[str] = None,
    ) -> List[tuple]:
        """Get top N performing actions by average outcome score."""
        records = self._all_records(tenant_id)
        action_scores: Dict[str, List[float]] = {}
        for r in records:
            if r.outcome_score is not None:
                if r.action not in action_scores:
                    action_scores[r.action] = []
                action_scores[r.action].append(r.outcome_score)
        if not action_scores:
            return []
        avgs = [(action, sum(scores) / len(scores)) for action, scores in action_scores.items()]
        avgs.sort(key=lambda x: x[1], reverse=True)
        return avgs[:n]

    def feedback_trend(self, window: int = 10, tenant_id: Optional[str] = None) -> Dict[str, float]:
        """Compare recent vs earlier outcome scores to detect trends."""
        records = self._all_records(tenant_id)
        scored = [r for r in records if r.outcome_score is not None]
        if len(scored) < 2:
            return {"recent_average": 0.0, "earlier_average": 0.0}
        mid = max(1, len(scored) // 2)
        earlier = scored[:mid]
        recent = scored[mid:]
        earlier_scores = [r.outcome_score for r in earlier if r.outcome_score is not None]
        recent_scores = [r.outcome_score for r in recent if r.outcome_score is not None]
        earlier_avg = sum(earlier_scores) / len(earlier) if earlier else 0.0
        recent_avg = sum(recent_scores) / len(recent) if recent else 0.0
        return {"recent_average": recent_avg, "earlier_average": earlier_avg}

    def summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics."""
        records = self._all_records(tenant_id)
        return {
            "total_records": len(records),
            "acceptance_rate": self.acceptance_rate(tenant_id),
            "average_outcome_score": self.average_outcome_score(tenant_id),
            "outcome_distribution": self.outcome_distribution(tenant_id),
        }

    def _all_records(self, tenant_id: Optional[str] = None) -> List[FeedbackRecord]:
        """Get all records, optionally filtered by tenant."""
        all_recs: List[FeedbackRecord] = []
        for records in self._records.values():
            all_recs.extend(records)
        if tenant_id is not None:
            all_recs = [r for r in all_recs if r.tenant_id == tenant_id]
        return all_recs


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
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    feedback_status: str = "pending"


class DecisionEngine:
    """Decision support engine with rule-based and ML-based paths."""

    def __init__(self, algorithm: str = "rule_based", model=None):
        self.algorithm = algorithm
        self.model = model
        self._bus: Optional[EventBus] = None
        self.feedback = FeedbackTracker()

    def recommend(self, state: FarmState) -> RecommendationResult:
        """Generate recommendations based on farm state."""
        if self.algorithm == "ensemble":
            return self._recommend_ensemble(state)
        if self.algorithm == "ml":
            return self._recommend_ml(state)
        return self._recommend_rule_based(state)

    def record_outcome(
        self,
        recommendation_id: str,
        action: str = "",
        outcome: Optional[str] = None,
        outcome_score: Optional[float] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """Record outcome feedback for a recommendation."""
        return self.feedback.record(
            recommendation_id=recommendation_id,
            action=action,
            tenant_id=tenant_id,
            outcome=outcome,
            outcome_score=outcome_score,
            metadata=metadata,
        )

    def feedback_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback summary statistics."""
        return self.feedback.summary(tenant_id=tenant_id)

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
        """Generate recommendations using ensemble of rule-based and ML paths."""
        if self.model is None:
            raise RuntimeError("Ensemble algorithm requires a model")
        if not self.model.is_trained:
            raise RuntimeError(f"Model {self.model.name} is not trained")
        rule_result = self._recommend_rule_based(state)
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

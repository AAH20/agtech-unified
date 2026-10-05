"""Tests for feedback loop / outcome tracking in DecisionEngine."""

from __future__ import annotations

import pytest

from src.decision_support.recommender import (
    DecisionEngine,
    FarmState,
    FeedbackRecord,
    FeedbackTracker,
)


def _healthy_state() -> FarmState:
    return FarmState(
        soil_moisture=0.6,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.7,
        pest_pressure=0.1,
    )


def _critical_state() -> FarmState:
    return FarmState(
        soil_moisture=0.1,
        temperature=35.0,
        crop_height=0.2,
        nutrient_level=0.1,
        pest_pressure=0.8,
    )


class TestFeedbackRecord:
    """FeedbackRecord dataclass tests."""

    def test_create_minimal_record(self):
        rec = FeedbackRecord(
            recommendation_id="rec-1",
            action="irrigate",
            tenant_id="t1",
        )
        assert rec.recommendation_id == "rec-1"
        assert rec.action == "irrigate"
        assert rec.tenant_id == "t1"
        assert rec.outcome is None
        assert rec.outcome_score is None
        assert rec.timestamp is not None

    def test_create_full_record(self):
        rec = FeedbackRecord(
            recommendation_id="rec-2",
            action="fertilize",
            tenant_id="t2",
            outcome="success",
            outcome_score=0.9,
            timestamp=1000.0,
        )
        assert rec.outcome == "success"
        assert rec.outcome_score == 0.9
        assert rec.timestamp == 1000.0


class TestFeedbackTracker:
    """FeedbackTracker tests."""

    def test_record_outcome(self):
        tracker = FeedbackTracker()
        tracker.record("rec-1", "irrigate", outcome="success", outcome_score=0.8)
        history = tracker.get_history("rec-1")
        assert len(history) == 1
        assert history[0].outcome == "success"
        assert history[0].outcome_score == 0.8

    def test_get_history_empty(self):
        tracker = FeedbackTracker()
        assert tracker.get_history("nonexistent") == []

    def test_get_history_multiple(self):
        tracker = FeedbackTracker()
        tracker.record("rec-1", "irrigate", outcome="success", outcome_score=0.8)
        tracker.record("rec-1", "irrigate", outcome="failure", outcome_score=0.2)
        history = tracker.get_history("rec-1")
        assert len(history) == 2

    def test_acceptance_rate_no_records(self):
        tracker = FeedbackTracker()
        assert tracker.acceptance_rate() == 0.0

    def test_acceptance_rate_all_accepted(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="accepted")
        tracker.record("r2", "fertilize", outcome="accepted")
        assert tracker.acceptance_rate() == 1.0

    def test_acceptance_rate_mixed(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="accepted")
        tracker.record("r2", "fertilize", outcome="rejected")
        tracker.record("r3", "shade", outcome="accepted")
        assert tracker.acceptance_rate() == pytest.approx(2.0 / 3.0)

    def test_average_outcome_score_no_records(self):
        tracker = FeedbackTracker()
        assert tracker.average_outcome_score() == 0.0

    def test_average_outcome_score_computed(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="success", outcome_score=0.8)
        tracker.record("r2", "fertilize", outcome="success", outcome_score=0.6)
        assert tracker.average_outcome_score() == pytest.approx(0.7)

    def test_average_outcome_score_ignores_none(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="success", outcome_score=0.8)
        tracker.record("r2", "fertilize", outcome="pending")
        assert tracker.average_outcome_score() == pytest.approx(0.8)

    def test_action_success_rate(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="success")
        tracker.record("r2", "irrigate", outcome="failure")
        tracker.record("r3", "fertilize", outcome="success")
        assert tracker.action_success_rate("irrigate") == pytest.approx(0.5)
        assert tracker.action_success_rate("fertilize") == pytest.approx(1.0)

    def test_action_success_rate_no_records(self):
        tracker = FeedbackTracker()
        assert tracker.action_success_rate("irrigate") == 0.0

    def test_summary_empty(self):
        tracker = FeedbackTracker()
        summary = tracker.summary()
        assert summary["total_records"] == 0
        assert summary["acceptance_rate"] == 0.0
        assert summary["average_outcome_score"] == 0.0

    def test_summary_with_data(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="accepted", outcome_score=0.9)
        tracker.record("r2", "fertilize", outcome="rejected")
        summary = tracker.summary()
        assert summary["total_records"] == 2
        assert summary["acceptance_rate"] == pytest.approx(0.5)
        assert summary["average_outcome_score"] == pytest.approx(0.9)

    def test_tenant_isolation(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", tenant_id="t1", outcome="accepted")
        tracker.record("r2", "fertilize", tenant_id="t2", outcome="rejected")
        t1_history = tracker.get_history("r1", tenant_id="t1")
        assert len(t1_history) == 1
        assert t1_history[0].tenant_id == "t1"


class TestDecisionEngineFeedbackIntegration:
    """DecisionEngine feedback loop integration tests."""

    def test_recommend_generates_id(self):
        engine = DecisionEngine()
        state = _critical_state()
        result = engine.recommend(state)
        assert result.recommendation_id is not None
        assert len(result.recommendation_id) > 0

    def test_recommend_generates_unique_ids(self):
        engine = DecisionEngine()
        state = _critical_state()
        result1 = engine.recommend(state)
        result2 = engine.recommend(state)
        assert result1.recommendation_id != result2.recommendation_id

    def test_engine_has_feedback_tracker(self):
        engine = DecisionEngine()
        assert hasattr(engine, "feedback")
        assert isinstance(engine.feedback, FeedbackTracker)

    def test_record_outcome_via_engine(self):
        engine = DecisionEngine()
        state = _critical_state()
        result = engine.recommend(state)
        engine.record_outcome(result.recommendation_id, outcome="accepted", outcome_score=0.85)
        history = engine.feedback.get_history(result.recommendation_id)
        assert len(history) == 1
        assert history[0].outcome == "accepted"
        assert history[0].outcome_score == 0.85

    def test_engine_feedback_summary(self):
        engine = DecisionEngine()
        state = _critical_state()
        result = engine.recommend(state)
        engine.record_outcome(result.recommendation_id, outcome="accepted")
        summary = engine.feedback_summary()
        assert summary["total_records"] == 1
        assert summary["acceptance_rate"] == 1.0

    def test_recommendation_result_has_feedback_fields(self):
        engine = DecisionEngine()
        state = _critical_state()
        result = engine.recommend(state)
        assert hasattr(result, "recommendation_id")
        assert hasattr(result, "feedback_status")
        assert result.feedback_status == "pending"

    def test_feedback_status_updates_after_outcome(self):
        engine = DecisionEngine()
        state = _critical_state()
        result = engine.recommend(state)
        engine.record_outcome(result.recommendation_id, outcome="accepted")
        # The result object itself is immutable; check via tracker
        history = engine.feedback.get_history(result.recommendation_id)
        assert history[0].outcome == "accepted"

    def test_multiple_engines_independent_feedback(self):
        engine1 = DecisionEngine()
        engine2 = DecisionEngine()
        state = _critical_state()
        result = engine1.recommend(state)
        engine1.record_outcome(result.recommendation_id, outcome="accepted")
        assert engine1.feedback.summary()["total_records"] == 1
        assert engine2.feedback.summary()["total_records"] == 0


class TestFeedbackLoopMetrics:
    """Tests for feedback loop metrics and analytics."""

    def test_outcome_distribution(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="success")
        tracker.record("r2", "irrigate", outcome="failure")
        tracker.record("r3", "fertilize", outcome="success")
        dist = tracker.outcome_distribution()
        assert dist["success"] == 2
        assert dist["failure"] == 1

    def test_outcome_distribution_empty(self):
        tracker = FeedbackTracker()
        assert tracker.outcome_distribution() == {}

    def test_top_performing_actions(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="success", outcome_score=0.9)
        tracker.record("r2", "irrigate", outcome="success", outcome_score=0.8)
        tracker.record("r3", "fertilize", outcome="failure", outcome_score=0.2)
        top = tracker.top_performing_actions(n=2)
        assert len(top) == 2
        assert top[0][0] == "irrigate"
        assert top[0][1] == pytest.approx(0.85)

    def test_top_performing_actions_empty(self):
        tracker = FeedbackTracker()
        assert tracker.top_performing_actions() == []

    def test_feedback_trend_improving(self):
        tracker = FeedbackTracker()
        # Early outcomes are poor, later ones are good
        for i in range(5):
            tracker.record(f"r{i}", "irrigate", outcome="failure", outcome_score=0.2)
        for i in range(5, 10):
            tracker.record(f"r{i}", "irrigate", outcome="success", outcome_score=0.9)
        trend = tracker.feedback_trend(window=5)
        assert trend["recent_average"] > trend["earlier_average"]

    def test_feedback_trend_declining(self):
        tracker = FeedbackTracker()
        for i in range(5):
            tracker.record(f"r{i}", "irrigate", outcome="success", outcome_score=0.9)
        for i in range(5, 10):
            tracker.record(f"r{i}", "irrigate", outcome="failure", outcome_score=0.2)
        trend = tracker.feedback_trend(window=5)
        assert trend["recent_average"] < trend["earlier_average"]

    def test_feedback_trend_insufficient_data(self):
        tracker = FeedbackTracker()
        tracker.record("r1", "irrigate", outcome="success", outcome_score=0.8)
        trend = tracker.feedback_trend(window=5)
        assert trend["recent_average"] == 0.0
        assert trend["earlier_average"] == 0.0

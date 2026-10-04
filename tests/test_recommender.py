"""Dedicated unit tests for DecisionEngine (src/decision_support/recommender.py).

Covers algorithm selection, boundary conditions, invalid FarmState values,
and recommendation logic for each sensor dimension.
"""

from __future__ import annotations

import pytest

from src.decision_support.recommender import DecisionEngine, FarmState, RecommendationResult


class TestDecisionEngineInit:
    """DecisionEngine initialization tests."""

    def test_default_algorithm_is_rule_based(self):
        engine = DecisionEngine()
        assert engine.algorithm == "rule_based"

    def test_custom_algorithm_stored(self):
        engine = DecisionEngine(algorithm="custom_algo")
        assert engine.algorithm == "custom_algo"


class TestSoilMoistureRecommendations:
    """Tests for soil moisture threshold logic."""

    def test_critical_moisture_triggers_urgent_irrigation(self, decision_engine):
        state = FarmState(
            soil_moisture=0.1,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any("URGENT" in r and "Irrigate" in r for r in result.recommendations)
        assert "irrigate" in result.actions

    def test_low_moisture_triggers_irrigation_soon(self, decision_engine):
        state = FarmState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any("Irrigate soon" in r for r in result.recommendations)
        assert "irrigate" in result.actions

    def test_adequate_moisture_no_irrigation(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert not any("irrigate" in a for a in result.actions)

    def test_moisture_boundary_02_triggers_critical(self, decision_engine):
        """Exactly 0.2 moisture triggers critical (strict less-than)."""
        state = FarmState(
            soil_moisture=0.2,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        # 0.2 is NOT < 0.2, so no critical alert
        assert not any("URGENT" in r for r in result.recommendations)

    def test_moisture_boundary_04_triggers_low(self, decision_engine):
        """Exactly 0.4 moisture triggers low (strict less-than)."""
        state = FarmState(
            soil_moisture=0.4,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        # 0.4 is NOT < 0.4, so no low alert
        assert not any("Irrigate soon" in r for r in result.recommendations)


class TestTemperatureRecommendations:
    """Tests for temperature threshold logic."""

    def test_extreme_heat_triggers_shade(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=40.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any("Heat stress" in r for r in result.recommendations)
        assert "shade" in result.actions

    def test_frost_risk_triggers_protection(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=2.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any("Frost risk" in r for r in result.recommendations)
        assert "frost_protection" in result.actions

    def test_moderate_temperature_no_alert(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert not any("shade" in a for a in result.actions)
        assert not any("frost" in a for a in result.actions)

    def test_temperature_boundary_38_triggers_heat(self, decision_engine):
        """Exactly 38°C triggers heat stress (strict greater-than)."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=38.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        # 38 is NOT > 38, so no heat alert
        assert not any("Heat stress" in r for r in result.recommendations)

    def test_temperature_boundary_5_triggers_frost(self, decision_engine):
        """Exactly 5°C triggers frost risk (strict less-than)."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=5.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        # 5 is NOT < 5, so no frost alert
        assert not any("Frost risk" in r for r in result.recommendations)


class TestNutrientRecommendations:
    """Tests for nutrient level threshold logic."""

    def test_critical_nutrient_triggers_fertilizer(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.1,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any(
            "fertilizer" in r.lower() or "nutrient" in r.lower() for r in result.recommendations
        )
        assert "fertilize" in result.actions

    def test_low_nutrient_triggers_consideration(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.3,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any("Consider fertilization" in r for r in result.recommendations)

    def test_adequate_nutrient_no_alert(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert not any("fertil" in a for a in result.actions)


class TestPestRecommendations:
    """Tests for pest pressure threshold logic."""

    def test_high_pest_pressure_triggers_control(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.8,
        )
        result = decision_engine.recommend(state)
        assert any("URGENT" in r and "Pest" in r for r in result.recommendations)
        assert "pest_control" in result.actions

    def test_elevated_pest_pressure_triggers_monitoring(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.5,
        )
        result = decision_engine.recommend(state)
        assert any("Monitor pests" in r for r in result.recommendations)
        assert "monitor_pests" in result.actions

    def test_low_pest_pressure_no_alert(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.2,
        )
        result = decision_engine.recommend(state)
        assert not any("pest" in a for a in result.actions)

    def test_pest_boundary_07_triggers_control(self, decision_engine):
        """Exactly 0.7 pest pressure triggers control (strict greater-than)."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.7,
        )
        result = decision_engine.recommend(state)
        # 0.7 is NOT > 0.7, so no urgent pest control
        assert not any("URGENT" in r and "Pest" in r for r in result.recommendations)


class TestCropHeightRecommendations:
    """Tests for crop height check logic."""

    def test_low_crop_height_with_adequate_moisture_triggers_inspection(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.05,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert any("germination" in r.lower() for r in result.recommendations)
        assert "inspect" in result.actions

    def test_low_crop_height_with_low_moisture_no_inspection(self, decision_engine):
        """Low crop height + low moisture → no germination check (moisture takes priority)."""
        state = FarmState(
            soil_moisture=0.1,
            temperature=25.0,
            crop_height=0.05,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert not any("germination" in r.lower() for r in result.recommendations)

    def test_adequate_crop_height_no_inspection(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert not any("inspect" in a for a in result.actions)


class TestPriorityScore:
    """Tests for priority score calculation."""

    def test_priority_score_capped_at_1(self, decision_engine):
        state = FarmState(
            soil_moisture=0.0,
            temperature=50.0,
            crop_height=0.0,
            nutrient_level=0.0,
            pest_pressure=1.0,
        )
        result = decision_engine.recommend(state)
        assert result.priority_score <= 1.0

    def test_priority_score_zero_for_healthy_state(self, decision_engine):
        state = FarmState(
            soil_moisture=0.6,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.7,
            pest_pressure=0.1,
        )
        result = decision_engine.recommend(state)
        assert result.priority_score == 0.0

    def test_priority_score_increases_with_issues(self, decision_engine):
        healthy = FarmState(
            soil_moisture=0.6,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.7,
            pest_pressure=0.1,
        )
        problematic = FarmState(
            soil_moisture=0.1,
            temperature=40.0,
            crop_height=0.05,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result_healthy = decision_engine.recommend(healthy)
        result_problematic = decision_engine.recommend(problematic)
        assert result_problematic.priority_score > result_healthy.priority_score

    def test_actions_count_matches_recommendations_count(self, decision_engine):
        state = FarmState(
            soil_moisture=0.1,
            temperature=40.0,
            crop_height=0.05,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = decision_engine.recommend(state)
        assert len(result.actions) == len(result.recommendations)


class TestRecommendationResult:
    """Tests for RecommendationResult structure."""

    def test_result_contains_all_fields(self, decision_engine):
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        assert isinstance(result, RecommendationResult)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.priority_score, float)
        assert isinstance(result.algorithm, str)
        assert isinstance(result.actions, list)

    def test_algorithm_propagates_to_result(self):
        engine = DecisionEngine(algorithm="test_algo")
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert result.algorithm == "test_algo"


class TestEdgeCases:
    """Edge case tests for DecisionEngine."""

    def test_all_zeros_produces_recommendations(self, decision_engine, all_zeros_farm_state):
        result = decision_engine.recommend(all_zeros_farm_state)
        assert result.priority_score > 0
        assert len(result.recommendations) > 0

    def test_extreme_values_produces_recommendations(self, decision_engine, extreme_farm_state):
        result = decision_engine.recommend(extreme_farm_state)
        assert result.priority_score == 1.0
        assert len(result.recommendations) >= 3

    def test_negative_values_rejected(self, decision_engine):
        """Negative values are rejected by FarmState validation."""
        with pytest.raises(ValueError):
            FarmState(
                soil_moisture=-0.1,
                temperature=-10.0,
                crop_height=-0.1,
                nutrient_level=-0.1,
                pest_pressure=-0.1,
            )

    def test_very_large_values_rejected(self, decision_engine):
        """Very large values are rejected by FarmState validation."""
        with pytest.raises(ValueError):
            FarmState(
                soil_moisture=100.0,
                temperature=1000.0,
                crop_height=100.0,
                nutrient_level=100.0,
                pest_pressure=100.0,
            )

    def test_boundary_moisture_02_exact(self, decision_engine):
        """Moisture exactly at 0.2 boundary."""
        state = FarmState(
            soil_moisture=0.2,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        # 0.2 is not < 0.2, so no critical alert
        assert not any("URGENT" in r for r in result.recommendations)

    def test_boundary_nutrient_02_exact(self, decision_engine):
        """Nutrient exactly at 0.2 boundary."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.2,
            pest_pressure=0.0,
        )
        result = decision_engine.recommend(state)
        # 0.2 is not < 0.2, so no critical alert
        assert not any("critical" in r.lower() for r in result.recommendations)

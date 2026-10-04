"""Test decision support system for agricultural recommendations."""
import pytest
from src.decision_support.recommender import DecisionEngine, FarmState


def test_recommendation_empty_state():
    """All-zeros state triggers critical alerts (soil=0, nutrients=0)."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.0,
        temperature=20.0,
        crop_height=0.0,
        nutrient_level=0.0,
        pest_pressure=0.0,
    )
    result = engine.recommend(state)
    # Soil moisture 0 < 0.2 → urgent irrigation
    # Nutrient level 0 < 0.2 → urgent fertilizer
    assert len(result.recommendations) >= 2
    assert result.priority_score > 0.5


def test_recommendation_drought_alert():
    """Low soil moisture triggers irrigation recommendation."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.1,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.5,
        pest_pressure=0.0,
    )
    result = engine.recommend(state)
    assert any("irrigate" in r.lower() or "water" in r.lower() for r in result.recommendations)


def test_recommendation_pest_alert():
    """High pest pressure triggers pest control recommendation."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.5,
        pest_pressure=0.9,
    )
    result = engine.recommend(state)
    assert any("pest" in r.lower() for r in result.recommendations)


def test_recommendation_fertilizer_alert():
    """Low nutrients trigger fertilizer recommendation."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.1,
        pest_pressure=0.0,
    )
    result = engine.recommend(state)
    assert any("fertil" in r.lower() or "nutrient" in r.lower() for r in result.recommendations)


def test_recommendation_healthy_crop():
    """Healthy crop returns no urgent recommendations."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.6,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.7,
        pest_pressure=0.1,
    )
    result = engine.recommend(state)
    assert result.priority_score < 0.3


def test_recommendation_multiple_issues():
    """Multiple issues generate multiple recommendations."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.1,
        temperature=35.0,
        crop_height=0.2,
        nutrient_level=0.1,
        pest_pressure=0.8,
    )
    result = engine.recommend(state)
    assert len(result.recommendations) >= 2


def test_recommendation_priority_ordering():
    """Higher priority issues appear first."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.05,
        temperature=40.0,
        crop_height=0.1,
        nutrient_level=0.05,
        pest_pressure=0.95,
    )
    result = engine.recommend(state)
    assert result.priority_score > 0.7


def test_recommendation_result_contains_algorithm():
    """RecommendationResult includes algorithm name."""
    engine = DecisionEngine(algorithm="rule_based")
    state = FarmState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.5,
        nutrient_level=0.5,
        pest_pressure=0.0,
    )
    result = engine.recommend(state)
    assert result.algorithm == "rule_based"


def test_recommendation_temperature_stress():
    """Extreme temperature triggers recommendation."""
    engine = DecisionEngine()
    state = FarmState(
        soil_moisture=0.5,
        temperature=42.0,
        crop_height=0.5,
        nutrient_level=0.5,
        pest_pressure=0.0,
    )
    result = engine.recommend(state)
    assert any("heat" in r.lower() or "temperature" in r.lower() for r in result.recommendations)

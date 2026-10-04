"""Tests for ML-based recommendation path in DecisionEngine."""

import pytest

from src.decision_support.ml_models import LinearRegressionModel
from src.decision_support.recommender import DecisionEngine, FarmState


def _train_yield_model():
    """Create a trained linear model: yield decreases with pest_pressure and low soil_moisture."""
    model = LinearRegressionModel(name="yield_model", version="1.0.0")
    # Features: [soil_moisture, temperature, crop_height, nutrient_level, pest_pressure]
    # Target: yield (0-1 scale) — non-collinear data to avoid singular matrix
    X = [
        [0.8, 25.0, 1.0, 0.8, 0.0],
        [0.6, 28.0, 0.7, 0.5, 0.2],
        [0.7, 22.0, 0.9, 0.6, 0.1],
        [0.4, 30.0, 0.5, 0.3, 0.5],
        [0.5, 24.0, 0.8, 0.7, 0.3],
        [0.3, 26.0, 0.6, 0.4, 0.4],
        [0.9, 20.0, 1.1, 0.9, 0.0],
        [0.2, 32.0, 0.4, 0.2, 0.6],
        [0.1, 35.0, 0.3, 0.1, 0.8],
        [0.05, 38.0, 0.2, 0.05, 0.9],
    ]
    y = [0.95, 0.72, 0.85, 0.45, 0.68, 0.52, 0.98, 0.28, 0.15, 0.05]
    model.train(
        X,
        y,
        feature_names=[
            "soil_moisture",
            "temperature",
            "crop_height",
            "nutrient_level",
            "pest_pressure",
        ],
    )
    return model


class TestMLDecisionEngine:
    """Test the ML-based recommendation path."""

    def test_ml_engine_with_trained_model(self):
        """ML engine produces recommendations using a trained model."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        assert result.algorithm == "ml"
        assert len(result.recommendations) > 0
        assert result.priority_score > 0.3
        assert result.confidence > 0.0

    def test_ml_engine_healthy_state_low_priority(self):
        """ML engine gives low priority for healthy crop state."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.8,
            temperature=25.0,
            crop_height=1.0,
            nutrient_level=0.8,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert result.priority_score < 0.5

    def test_ml_engine_untrained_model_raises(self):
        """ML engine raises RuntimeError when model is not trained."""
        model = LinearRegressionModel(name="untrained")
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        with pytest.raises(RuntimeError, match="not trained"):
            engine.recommend(state)

    def test_ml_engine_feature_importance_populated(self):
        """ML engine returns feature importance from the model."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        assert len(result.feature_importance) > 0
        # Feature importance should sum to ~1.0
        total = sum(result.feature_importance.values())
        assert abs(total - 1.0) < 0.01

    def test_ml_engine_confidence_from_model(self):
        """ML engine confidence comes from model R-squared, not rule count."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        # Confidence should be the model's R-squared (high for this clean data)
        assert result.confidence > 0.5

    def test_ml_engine_drought_detection(self):
        """ML engine detects drought conditions."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.05,
            temperature=30.0,
            crop_height=0.3,
            nutrient_level=0.3,
            pest_pressure=0.1,
        )
        result = engine.recommend(state)
        assert any(
            "irrigate" in r.lower() or "water" in r.lower() or "moisture" in r.lower()
            for r in result.recommendations
        )

    def test_ml_engine_pest_detection(self):
        """ML engine detects high pest pressure."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.9,
        )
        result = engine.recommend(state)
        assert any("pest" in r.lower() for r in result.recommendations)

    def test_ml_engine_returns_actions(self):
        """ML engine returns actionable items."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        assert len(result.actions) > 0

    def test_ml_engine_tenant_id_preserved(self):
        """ML engine preserves tenant_id from state."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
            tenant_id="tenant-123",
        )
        result = engine.recommend(state)
        assert result.tenant_id == "tenant-123"

    def test_ml_engine_priority_score_range(self):
        """ML engine priority score is always in [0, 1]."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        for soil in [0.0, 0.3, 0.5, 0.8, 1.0]:
            state = FarmState(
                soil_moisture=soil,
                temperature=25.0,
                crop_height=0.5,
                nutrient_level=0.5,
                pest_pressure=0.0,
            )
            result = engine.recommend(state)
            assert 0.0 <= result.priority_score <= 1.0

    def test_ml_engine_nutrient_detection(self):
        """ML engine detects low nutrient levels."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.05,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert any("nutrient" in r.lower() or "fertil" in r.lower() for r in result.recommendations)

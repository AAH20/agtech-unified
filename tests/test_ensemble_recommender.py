"""Tests for ensemble/hybrid recommendation in DecisionEngine."""

import pytest

from src.decision_support.ml_models import LinearRegressionModel
from src.decision_support.recommender import DecisionEngine, FarmState


def _train_yield_model():
    """Create a trained linear model: yield decreases with pest_pressure and low soil_moisture."""
    model = LinearRegressionModel(name="yield_model", version="1.0.0")
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


class TestEnsembleRecommender:
    """Test the ensemble/hybrid recommendation path."""

    def test_ensemble_algorithm_label(self):
        """Ensemble result has algorithm='ensemble'."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert result.algorithm == "ensemble"

    def test_ensemble_merges_recommendations(self):
        """Ensemble combines recommendations from both rule-based and ML paths."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        # Should have recommendations from both paths
        assert len(result.recommendations) > 0
        # Should include irrigation (from rule-based) and pest control
        rec_text = " ".join(result.recommendations).lower()
        assert "irrigate" in rec_text or "moisture" in rec_text
        assert "pest" in rec_text

    def test_ensemble_priority_score_range(self):
        """Ensemble priority score is always in [0, 1]."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
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

    def test_ensemble_confidence_range(self):
        """Ensemble confidence is always in [0, 1]."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert 0.0 <= result.confidence <= 1.0

    def test_ensemble_feature_importance_populated(self):
        """Ensemble returns merged feature importance."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
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

    def test_ensemble_healthy_state_low_priority(self):
        """Ensemble gives low priority for healthy crop state."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.8,
            temperature=25.0,
            crop_height=1.0,
            nutrient_level=0.8,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert result.priority_score < 0.5

    def test_ensemble_critical_state_high_priority(self):
        """Ensemble gives high priority for critical crop state."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.05,
            temperature=38.0,
            crop_height=0.1,
            nutrient_level=0.05,
            pest_pressure=0.9,
        )
        result = engine.recommend(state)
        assert result.priority_score > 0.5

    def test_ensemble_returns_actions(self):
        """Ensemble returns actionable items."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        assert len(result.actions) > 0

    def test_ensemble_tenant_id_preserved(self):
        """Ensemble preserves tenant_id from state."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
            tenant_id="tenant-456",
        )
        result = engine.recommend(state)
        assert result.tenant_id == "tenant-456"

    def test_ensemble_without_model_raises(self):
        """Ensemble requires a model (unlike pure rule-based)."""
        engine = DecisionEngine(algorithm="ensemble", model=None)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        with pytest.raises(RuntimeError, match="model"):
            engine.recommend(state)

    def test_ensemble_untrained_model_raises(self):
        """Ensemble raises RuntimeError when model is not trained."""
        model = LinearRegressionModel(name="untrained")
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        with pytest.raises(RuntimeError, match="not trained"):
            engine.recommend(state)

    def test_ensemble_deduplicates_recommendations(self):
        """Ensemble does not produce duplicate recommendations."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        # No exact duplicates
        assert len(result.recommendations) == len(set(result.recommendations))

    def test_ensemble_priority_at_least_as_high_as_ml_for_critical(self):
        """Ensemble priority should be at least as high as ML-only for critical states."""
        model = _train_yield_model()
        ensemble_engine = DecisionEngine(algorithm="ensemble", model=model)
        ml_engine = DecisionEngine(algorithm="ml", model=model)
        state = FarmState(
            soil_moisture=0.05,
            temperature=38.0,
            crop_height=0.1,
            nutrient_level=0.05,
            pest_pressure=0.9,
        )
        ensemble_result = ensemble_engine.recommend(state)
        ml_result = ml_engine.recommend(state)
        # Ensemble should be at least as concerned as ML-only
        assert ensemble_result.priority_score >= ml_result.priority_score

    def test_ensemble_merges_feature_importance(self):
        """Ensemble feature importance includes features from both paths."""
        model = _train_yield_model()
        engine = DecisionEngine(algorithm="ensemble", model=model)
        state = FarmState(
            soil_moisture=0.1,
            temperature=35.0,
            crop_height=0.2,
            nutrient_level=0.1,
            pest_pressure=0.8,
        )
        result = engine.recommend(state)
        # Should have importance from both rule-based features and ML features
        assert "soil_moisture" in result.feature_importance
        assert "pest_pressure" in result.feature_importance

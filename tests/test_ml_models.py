"""Tests for ML model interfaces and yield prediction."""

import pytest

from src.decision_support.ml_models import (
    FeatureVector,
    LinearRegressionModel,
    PredictionResult,
    SimpleMLModel,
    YieldPredictor,
)


class TestLinearRegressionTraining:
    def test_trains_and_predicts_known_value(self):
        """Model trained on exact linear data predicts held-out points."""
        model = LinearRegressionModel()
        # y = 2 + 3*x1 - 1*x2
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
        y = [2.0, 5.0, 1.0, 4.0, 7.0, 3.0]
        model.train(X, y, feature_names=["x1", "x2"])
        result = model.predict([3.0, 0.0])
        assert result.value == pytest.approx(11.0, abs=1e-6)

    def test_perfect_fit_has_r_squared_one(self):
        """Exact linear data yields R^2 of 1.0."""
        model = LinearRegressionModel()
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]
        model.train(X, y)
        assert model.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_train_rejects_mismatched_lengths(self):
        """X and y of different lengths raise ValueError."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="same number of samples"):
            model.train([[1.0], [2.0]], [1.0])

    def test_train_rejects_insufficient_samples(self):
        """Fewer samples than features + 1 raises ValueError."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="at least"):
            model.train([[1.0, 2.0]], [3.0])

    def test_train_rejects_empty_dataset(self):
        """Empty training data raises ValueError."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="empty"):
            model.train([], [])


class TestLinearRegressionPrediction:
    def test_predict_before_train_raises(self):
        """Predicting on an untrained model raises RuntimeError."""
        model = LinearRegressionModel()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict([1.0])

    def test_predict_rejects_wrong_feature_count(self):
        """Wrong number of features raises ValueError."""
        model = LinearRegressionModel()
        model.train([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="Expected 2 features"):
            model.predict([1.0, 2.0, 3.0])

    def test_prediction_result_carries_model_metadata(self):
        """PredictionResult includes model name, version, and features."""
        model = LinearRegressionModel(name="yield_model", version="2.1.0")
        model.train([[0.0], [1.0], [2.0], [3.0]], [0.0, 2.0, 4.0, 6.0], feature_names=["x"])
        result = model.predict([4.0])
        assert isinstance(result, PredictionResult)
        assert result.model_name == "yield_model"
        assert result.model_version == "2.1.0"
        assert result.features_used == ["x"]
        assert result.value == pytest.approx(8.0, abs=1e-6)
        assert 0.0 <= result.confidence <= 1.0

    def test_feature_importance_sums_to_one(self):
        """Relative feature importance across weights sums to 1.0."""
        model = LinearRegressionModel()
        model.train(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            feature_names=["a", "b", "c"],
        )
        importance = model.feature_importance()
        assert set(importance) == {"a", "b", "c"}
        assert sum(importance.values()) == pytest.approx(1.0)
        assert all(v >= 0.0 for v in importance.values())


class TestYieldPredictor:
    def test_default_model_is_linear_regression(self):
        """YieldPredictor wires up a LinearRegressionModel by default."""
        predictor = YieldPredictor()
        assert isinstance(predictor.model, LinearRegressionModel)
        assert not predictor.model.is_trained

    def test_predict_from_dict_uses_feature_names(self):
        """predict_from_dict orders values by the model's feature names."""
        predictor = YieldPredictor()
        # y = 1 + 1*soil_moisture + 2*temperature + 3*nutrient_level + 4*sunlight_hours + 5*rainfall_mm
        X = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        y = [2.0, 3.0, 4.0, 5.0, 6.0, 16.0]
        predictor.train(X, y)
        features = {
            "soil_moisture": 1.0,
            "temperature": 0.0,
            "nutrient_level": 0.0,
            "sunlight_hours": 0.0,
            "rainfall_mm": 0.0,
        }
        result = predictor.predict_from_dict(features)
        # y = 1 + 1*1.0 = 2.0
        assert result.value == pytest.approx(2.0, abs=1e-6)

    def test_feature_vector_to_list(self):
        """FeatureVector converts to an ordered list."""
        fv = FeatureVector({"a": 1.0, "b": 2.0, "c": 3.0})
        assert fv.to_list(["b", "a", "c"]) == [2.0, 1.0, 3.0]

    def test_simple_model_is_abstract(self):
        """SimpleMLModel cannot be instantiated directly, but a concrete
        subclass implementing the interface works."""

        class DummyModel(SimpleMLModel):
            def train(self, X, y):
                self._is_trained = True
                self._feature_names = ["x"]

            def predict(self, features):
                self.validate_features(features)
                return PredictionResult(
                    value=1.0,
                    confidence=0.9,
                    model_name=self.name,
                    model_version=self.version,
                    features_used=list(self._feature_names),
                )

        with pytest.raises(TypeError):
            SimpleMLModel("x")

        model = DummyModel("dummy")
        model.train([[1.0], [2.0]], [1.0, 2.0])
        result = model.predict([3.0])
        assert result.value == 1.0
        assert result.confidence == 0.9
        assert result.model_name == "dummy"

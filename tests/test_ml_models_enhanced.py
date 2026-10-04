"""Tests for ML model persistence, evaluation metrics, and additional models."""

import os
import tempfile

import pytest

from src.decision_support.ml_models import (
    ClassificationMetrics,
    CrossValidator,
    DecisionTreeClassifier,
    FeatureEngine,
    IsolationForestModel,
    LinearRegressionModel,
    ModelRegistry,
    RegressionMetrics,
    SimpleMLModel,
)


class TestModelPersistence:
    def test_save_and_load_pickle(self):
        """Model can be saved to and loaded from a pickle file."""
        model = LinearRegressionModel(name="test_model")
        X = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y = [1.0, 3.0, 5.0, 7.0, 9.0]
        model.train(X, y, feature_names=["x"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = SimpleMLModel.load(path)
            assert loaded.name == "test_model"
            assert loaded.is_trained
            result = loaded.predict([5.0])
            assert result.value == pytest.approx(11.0, abs=1e-6)
        finally:
            os.unlink(path)

    def test_save_and_load_joblib(self):
        """Model can be saved to and loaded from a joblib file."""
        model = LinearRegressionModel(name="test_joblib")
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 2.0, 4.0, 6.0]
        model.train(X, y, feature_names=["x"])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = SimpleMLModel.load(path)
            assert loaded.is_trained
            result = loaded.predict([4.0])
            assert result.value == pytest.approx(8.0, abs=1e-6)
        finally:
            os.unlink(path)

    def test_save_untrained_model_raises(self):
        """Saving an untrained model raises RuntimeError."""
        model = LinearRegressionModel()
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            with pytest.raises(RuntimeError, match="not trained"):
                model.save(f.name)

    def test_load_nonexistent_file_raises(self):
        """Loading from a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SimpleMLModel.load("/nonexistent/path/model.pkl")


class TestRegressionMetrics:
    def test_mae(self):
        """Mean Absolute Error is computed correctly."""
        metrics = RegressionMetrics()
        result = metrics.mae([1.0, 2.0, 3.0], [1.5, 2.5, 2.5])
        assert result == pytest.approx(0.5)

    def test_rmse(self):
        """Root Mean Squared Error is computed correctly."""
        metrics = RegressionMetrics()
        result = metrics.rmse([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        expected = ((1.0 + 0.0 + 1.0) / 3) ** 0.5
        assert result == pytest.approx(expected)

    def test_mape(self):
        """Mean Absolute Percentage Error is computed correctly."""
        metrics = RegressionMetrics()
        result = metrics.mape([100.0, 200.0], [110.0, 180.0])
        assert result == pytest.approx(10.0)  # (10% + 10%) / 2

    def test_r_squared(self):
        """R-squared is computed correctly."""
        metrics = RegressionMetrics()
        # Perfect prediction
        result = metrics.r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result == pytest.approx(1.0)

    def test_all_metrics(self):
        """All metrics computed together."""
        metrics = RegressionMetrics()
        result = metrics.compute_all([1.0, 2.0, 3.0], [1.5, 2.5, 2.5])
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert "r_squared" in result


class TestClassificationMetrics:
    def test_accuracy(self):
        """Accuracy is computed correctly."""
        metrics = ClassificationMetrics()
        result = metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0])
        assert result == pytest.approx(0.75)

    def test_precision(self):
        """Precision is computed correctly."""
        metrics = ClassificationMetrics()
        # TP=1, FP=1 -> precision = 0.5
        result = metrics.precision([1, 1, 0, 0], [1, 0, 1, 0])
        assert result == pytest.approx(0.5)

    def test_recall(self):
        """Recall is computed correctly."""
        metrics = ClassificationMetrics()
        # TP=1, FN=1 -> recall = 0.5
        result = metrics.recall([1, 1, 0, 0], [1, 0, 1, 0])
        assert result == pytest.approx(0.5)

    def test_f1(self):
        """F1 score is computed correctly."""
        metrics = ClassificationMetrics()
        # precision=0.5, recall=0.5 -> F1=0.5
        result = metrics.f1([1, 1, 0, 0], [1, 0, 1, 0])
        assert result == pytest.approx(0.5)

    def test_all_metrics(self):
        """All classification metrics computed together."""
        metrics = ClassificationMetrics()
        result = metrics.compute_all([1, 0, 1, 1], [1, 0, 0, 1])
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result


class TestCrossValidator:
    def test_k_fold_split(self):
        """K-fold cross-validation splits data correctly."""
        validator = CrossValidator(k=3)
        X = [[i] for i in range(9)]
        y = [float(i) for i in range(9)]
        folds = list(validator.split(X, y))
        assert len(folds) == 3
        # Each fold should have 3 test samples
        for train_X, train_y, test_X, test_y in folds:
            assert len(test_X) == 3
            assert len(train_X) == 6

    def test_cross_validate_returns_scores(self):
        """Cross-validation returns scores for each fold."""
        validator = CrossValidator(k=2)

        def model_factory():
            return LinearRegressionModel()

        X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        scores = validator.cross_validate(model_factory, X, y)
        assert len(scores) == 2
        # Perfect linear data should give R^2 close to 1.0
        assert all(s > 0.9 for s in scores)


class TestDecisionTreeClassifier:
    def test_trains_and_predicts(self):
        """Decision tree classifier trains and predicts."""
        model = DecisionTreeClassifier(max_depth=3)
        X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        model.train(X, y, feature_names=["x"])
        result = model.predict([0.5])
        assert result.value in [0.0, 1.0]
        assert result.confidence > 0.0

    def test_feature_importance(self):
        """Decision tree provides feature importance."""
        model = DecisionTreeClassifier(max_depth=3)
        X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
        y = [0.0, 1.0, 0.0, 1.0]
        model.train(X, y, feature_names=["a", "b"])
        importance = model.feature_importance()
        # At least one feature should have importance
        assert len(importance) >= 1
        assert sum(importance.values()) == pytest.approx(1.0)


class TestIsolationForestModel:
    def test_detects_anomalies(self):
        """Isolation forest detects anomalous points."""
        model = IsolationForestModel(n_estimators=10, sample_size=32)
        # Normal data clustered around 0
        X = [[0.0]] * 20 + [[1.0]] * 20
        y = [0.0] * 40
        model.train(X, y, feature_names=["x"])
        # Anomalous point far from cluster
        result = model.predict([100.0])
        # Score should be positive (anomaly detected)
        assert result.value > 0.0
        assert result.confidence > 0.0

    def test_normal_point_not_flagged(self):
        """Normal points are not flagged as anomalies."""
        model = IsolationForestModel(n_estimators=10, sample_size=32)
        X = [[0.0]] * 20 + [[1.0]] * 20
        y = [0.0] * 40
        model.train(X, y, feature_names=["x"])
        result = model.predict([0.5])
        assert result.value == pytest.approx(0.0, abs=0.5)


class TestFeatureEngine:
    def test_computes_growing_degree_days(self):
        """FeatureEngine computes growing degree days."""
        engine = FeatureEngine()
        features = engine.compute_growing_degree_days(temps=[20.0, 25.0, 30.0], base_temp=10.0)
        assert features == pytest.approx(45.0)  # (20-10) + (25-10) + (30-10)

    def test_computes_vpd(self):
        """FeatureEngine computes vapor pressure deficit."""
        engine = FeatureEngine()
        vpd = engine.compute_vpd(temperature=25.0, humidity=60.0)
        assert vpd > 0.0

    def test_computes_soil_moisture_trend(self):
        """FeatureEngine computes soil moisture trend."""
        engine = FeatureEngine()
        trend = engine.compute_trend([0.5, 0.4, 0.3, 0.2])
        assert trend < 0.0  # Decreasing trend

    def test_feature_names(self):
        """FeatureEngine returns computed feature names."""
        engine = FeatureEngine()
        engine.compute_growing_degree_days(temps=[20.0], base_temp=10.0)
        assert "gdd" in engine.feature_names


class TestModelRegistry:
    def test_register_and_get_model(self):
        """Models can be registered and retrieved."""
        registry = ModelRegistry()
        model = LinearRegressionModel(name="yield_model")
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 2.0, 4.0, 6.0]
        model.train(X, y, feature_names=["x"])
        registry.register("yield", model)
        retrieved = registry.get("yield")
        assert retrieved.name == "yield_model"
        assert retrieved.is_trained

    def test_list_models(self):
        """Registry lists all registered models."""
        registry = ModelRegistry()
        registry.register("a", LinearRegressionModel(name="a"))
        registry.register("b", LinearRegressionModel(name="b"))
        assert set(registry.list_models()) == {"a", "b"}

    def test_model_versioning(self):
        """Registry supports model versioning."""
        registry = ModelRegistry()
        model = LinearRegressionModel(name="yield", version="1.0")
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 2.0, 4.0, 6.0]
        model.train(X, y, feature_names=["x"])
        registry.register("yield", model, version="1.0")
        registry.register("yield", model, version="2.0")
        assert registry.get("yield", version="1.0") is not None
        assert registry.get("yield", version="2.0") is not None

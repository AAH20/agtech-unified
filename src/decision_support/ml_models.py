"""ML model interfaces and yield prediction models for decision support."""

from __future__ import annotations

import logging
import os
import pickle
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Named feature vector for model input."""

    values: Dict[str, float]

    def to_list(self, feature_names: Sequence[str]) -> List[float]:
        """Convert to ordered list using the given feature names."""
        return [self.values[name] for name in feature_names]


@dataclass
class PredictionResult:
    """Result of a model prediction."""

    value: float
    confidence: float
    model_name: str
    model_version: str
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleMLModel(ABC):
    """Interface for simple ML models used in decision support."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._is_trained = False
        self._feature_names: List[str] = []

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained

    @property
    def feature_names(self) -> List[str]:
        """Feature names the model expects, in order."""
        return list(self._feature_names)

    @abstractmethod
    def train(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        """Train the model on feature matrix X and target vector y."""

    @abstractmethod
    def predict(self, features: Sequence[float]) -> PredictionResult:
        """Predict a target value from a feature vector."""

    def validate_features(self, features: Sequence[float]) -> None:
        """Validate a feature vector against the trained feature count."""
        if not self._is_trained:
            raise RuntimeError(f"Model {self.name} is not trained")
        if len(features) != len(self._feature_names):
            raise ValueError(
                f"Expected {len(self._feature_names)} features "
                f"({self._feature_names}), got {len(features)}"
            )

    def save(self, path: str) -> None:
        """Save the model to a file (pickle or joblib based on extension)."""
        if not self._is_trained:
            raise RuntimeError(f"Model {self.name} is not trained")
        if path.endswith(".joblib"):
            try:
                import joblib

                joblib.dump(self, path)
            except ImportError:
                # Fall back to pickle
                with open(path, "wb") as f:
                    pickle.dump(self, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        logger.info("Saved model %s to %s", self.name, path)

    @staticmethod
    def load(path: str) -> "SimpleMLModel":
        """Load a model from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        if path.endswith(".joblib"):
            try:
                import joblib

                model = joblib.load(path)
                return model
            except ImportError:
                pass
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", path)
        return model


class LinearRegressionModel(SimpleMLModel):
    """Ordinary least squares linear regression for yield prediction.

    Fits y = intercept + sum(weight_i * x_i) using the normal equations.
    """

    def __init__(self, name: str = "linear_regression", version: str = "1.0.0"):
        super().__init__(name=name, version=version)
        self.weights: List[float] = []
        self.intercept: float = 0.0
        self._r_squared: Optional[float] = None

    @property
    def r_squared(self) -> Optional[float]:
        """Coefficient of determination from the last training run."""
        return self._r_squared

    def _solve(self, A: List[List[float]], b: List[float]) -> List[float]:
        """Solve a small linear system via Gaussian elimination with pivoting."""
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[pivot][col]) < 1e-12:
                raise ValueError("Singular matrix in linear regression solve")
            M[col], M[pivot] = M[pivot], M[col]
            for row in range(col + 1, n):
                factor = M[row][col] / M[col][col]
                for k in range(col, n + 1):
                    M[row][k] -= factor * M[col][k]
        x = [0.0] * n
        for row in range(n - 1, -1, -1):
            s = M[row][n] - sum(M[row][k] * x[k] for k in range(row + 1, n))
            x[row] = s / M[row][row]
        return x

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Fit the model via the normal equations (X^T X) w = X^T y."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        n_features = len(X[0])
        if n_features == 0:
            raise ValueError("Feature vectors must not be empty")
        if any(len(row) != n_features for row in X):
            raise ValueError("All feature vectors must have the same length")
        if len(X) < n_features + 1:
            raise ValueError(f"Need at least {n_features + 1} samples for {n_features} features")

        self._feature_names = (
            list(feature_names) if feature_names else [f"x{i}" for i in range(n_features)]
        )
        if len(self._feature_names) != n_features:
            raise ValueError("feature_names length must match feature count")

        p = n_features + 1
        A = [[0.0] * p for _ in range(p)]
        b = [0.0] * p
        for row, target in zip(X, y):
            aug = [1.0] + list(row)
            for i in range(p):
                b[i] += aug[i] * target
                for j in range(p):
                    A[i][j] += aug[i] * aug[j]

        solution = self._solve(A, b)
        self.intercept = solution[0]
        self.weights = solution[1:]
        self._r_squared = self._compute_r_squared(X, y)
        self._is_trained = True
        logger.info(
            "Trained %s v%s: %d features, %d samples, R^2=%.4f",
            self.name,
            self.version,
            n_features,
            len(X),
            self._r_squared,
        )

    def _compute_r_squared(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> float:
        """Compute R^2 of the fitted model on the training data."""
        n = len(y)
        mean_y = sum(y) / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        if ss_tot < 1e-12:
            return 1.0
        ss_res = sum((yi - self._raw_predict(row)) ** 2 for row, yi in zip(X, y))
        return max(0.0, 1.0 - ss_res / ss_tot)

    def _raw_predict(self, features: Sequence[float]) -> float:
        """Compute the linear combination without validation."""
        return self.intercept + sum(w * x for w, x in zip(self.weights, features))

    def predict(self, features: Sequence[float]) -> PredictionResult:
        """Predict the target value for a feature vector."""
        self.validate_features(features)
        value = self._raw_predict(features)
        confidence = 0.5 if self._r_squared is None else max(0.0, min(1.0, self._r_squared))
        return PredictionResult(
            value=value,
            confidence=confidence,
            model_name=self.name,
            model_version=self.version,
            features_used=list(self._feature_names),
            metadata={
                "intercept": self.intercept,
                "weights": list(self.weights),
                "r_squared": self._r_squared,
            },
        )

    def feature_importance(self) -> Dict[str, float]:
        """Return absolute weight per feature as a relative importance score."""
        if not self._is_trained:
            raise RuntimeError(f"Model {self.name} is not trained")
        abs_weights = [abs(w) for w in self.weights]
        total = sum(abs_weights)
        if total < 1e-12:
            share = 1.0 / len(abs_weights)
            return {name: share for name in self._feature_names}
        return {name: w / total for name, w in zip(self._feature_names, abs_weights)}


class DecisionTreeClassifier(SimpleMLModel):
    """Simple decision tree classifier for classification tasks."""

    def __init__(self, name: str = "decision_tree", version: str = "1.0.0", max_depth: int = 5):
        super().__init__(name=name, version=version)
        self.max_depth = max_depth
        self._tree: Optional[Dict] = None
        self._feature_importance: Dict[str, float] = {}

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Train the decision tree classifier."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        n_features = len(X[0])
        self._feature_names = (
            list(feature_names) if feature_names else [f"x{i}" for i in range(n_features)]
        )
        self._tree = self._build_tree(X, y, depth=0)
        self._is_trained = True

    def _build_tree(self, X: Sequence[Sequence[float]], y: Sequence[float], depth: int) -> Dict:
        """Recursively build the decision tree."""
        # Check stopping conditions
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < 2:
            # Leaf node: return majority class
            from collections import Counter

            counts = Counter(y)
            majority = counts.most_common(1)[0][0]
            confidence = counts[majority] / len(y)
            return {"type": "leaf", "value": majority, "confidence": confidence}

        # Find best split
        best_gain = -1.0
        best_feature = 0
        best_threshold = 0.0
        n_features = len(X[0])

        for feature_idx in range(n_features):
            values = sorted(set(row[feature_idx] for row in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2.0
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_gain <= 0:
            from collections import Counter

            counts = Counter(y)
            majority = counts.most_common(1)[0][0]
            confidence = counts[majority] / len(y)
            return {"type": "leaf", "value": majority, "confidence": confidence}

        # Split data
        left_X, left_y, right_X, right_y = [], [], [], []
        for row, target in zip(X, y):
            if row[best_feature] <= best_threshold:
                left_X.append(row)
                left_y.append(target)
            else:
                right_X.append(row)
                right_y.append(target)

        # Track feature importance
        fname = self._feature_names[best_feature]
        self._feature_importance[fname] = self._feature_importance.get(fname, 0.0) + best_gain

        return {
            "type": "split",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }

    def _information_gain(
        self, X: Sequence[Sequence[float]], y: Sequence[float], feature_idx: int, threshold: float
    ) -> float:
        """Compute information gain for a split."""
        import math
        from collections import Counter

        def entropy(labels):
            counts = Counter(labels)
            total = len(labels)
            return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

        parent_entropy = entropy(y)
        left_y = [t for row, t in zip(X, y) if row[feature_idx] <= threshold]
        right_y = [t for row, t in zip(X, y) if row[feature_idx] > threshold]
        if not left_y or not right_y:
            return 0.0
        n = len(y)
        child_entropy = (len(left_y) / n) * entropy(left_y) + (len(right_y) / n) * entropy(right_y)
        return parent_entropy - child_entropy

    def predict(self, features: Sequence[float]) -> PredictionResult:
        """Predict the class for a feature vector."""
        self.validate_features(features)
        result = self._traverse(self._tree, features)
        return PredictionResult(
            value=result["value"],
            confidence=result["confidence"],
            model_name=self.name,
            model_version=self.version,
            features_used=list(self._feature_names),
        )

    def _traverse(self, node: Optional[Dict], features: Sequence[float]) -> Dict:
        """Traverse the tree to make a prediction."""
        if node is None or node["type"] == "leaf":
            return node or {"type": "leaf", "value": 0.0, "confidence": 0.0}
        if features[node["feature"]] <= node["threshold"]:
            return self._traverse(node["left"], features)
        return self._traverse(node["right"], features)

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self._is_trained:
            raise RuntimeError(f"Model {self.name} is not trained")
        total = sum(self._feature_importance.values())
        if total < 1e-12:
            share = 1.0 / len(self._feature_names)
            return {name: share for name in self._feature_names}
        return {k: v / total for k, v in self._feature_importance.items()}


class IsolationForestModel(SimpleMLModel):
    """Isolation Forest for anomaly detection."""

    def __init__(
        self,
        name: str = "isolation_forest",
        version: str = "1.0.0",
        n_estimators: int = 10,
        sample_size: int = 256,
    ):
        super().__init__(name=name, version=version)
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self._trees: List[Dict] = []
        self._threshold = 0.5

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Train the isolation forest."""
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        n_features = len(X[0])
        self._feature_names = (
            list(feature_names) if feature_names else [f"x{i}" for i in range(n_features)]
        )
        self._trees = []
        for _ in range(self.n_estimators):
            sample = random.sample(list(X), min(self.sample_size, len(X)))
            tree = self._build_tree(sample, depth=0)
            self._trees.append(tree)
        self._is_trained = True

    def _build_tree(self, X: Sequence[Sequence[float]], depth: int) -> Dict:
        """Build an isolation tree."""
        if depth >= 8 or len(X) <= 1:
            return {"type": "leaf", "size": len(X)}
        n_features = len(X[0])
        feature_idx = random.randint(0, n_features - 1)
        values = [row[feature_idx] for row in X]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return {"type": "leaf", "size": len(X)}
        threshold = random.uniform(min_val, max_val)
        left = [row for row in X if row[feature_idx] < threshold]
        right = [row for row in X if row[feature_idx] >= threshold]
        return {
            "type": "split",
            "feature": feature_idx,
            "threshold": threshold,
            "left": self._build_tree(left, depth + 1),
            "right": self._build_tree(right, depth + 1),
        }

    def _path_length(self, features: Sequence[float], tree: Dict) -> float:
        """Compute path length for a sample in a tree."""
        if tree["type"] == "leaf":
            return self._c(tree["size"])
        if features[tree["feature"]] < tree["threshold"]:
            return 1.0 + self._path_length(features, tree["left"])
        return 1.0 + self._path_length(features, tree["right"])

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0.0
        import math

        return 2.0 * (math.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def predict(self, features: Sequence[float]) -> PredictionResult:
        """Predict anomaly score for a feature vector."""
        self.validate_features(features)
        avg_path = sum(self._path_length(features, tree) for tree in self._trees) / len(self._trees)
        # Anomaly score: 0.5 is normal, closer to 1.0 is anomaly
        score = 1.0 - (avg_path / self._c(self.sample_size))
        score = max(0.0, min(1.0, score))
        return PredictionResult(
            value=score,
            confidence=score,
            model_name=self.name,
            model_version=self.version,
            features_used=list(self._feature_names),
        )


class RegressionMetrics:
    """Comprehensive regression metrics."""

    def mae(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Mean Absolute Error."""
        return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

    def rmse(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Root Mean Squared Error."""
        return (sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)) ** 0.5

    def mape(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """Mean Absolute Percentage Error."""
        return sum(abs((t - p) / t) for t, p in zip(y_true, y_pred) if t != 0) / len(y_true) * 100

    def r_squared(self, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
        """R-squared coefficient of determination."""
        mean_y = sum(y_true) / len(y_true)
        ss_tot = sum((t - mean_y) ** 2 for t in y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        if ss_tot < 1e-12:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    def compute_all(self, y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
        """Compute all regression metrics."""
        return {
            "mae": self.mae(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mape": self.mape(y_true, y_pred),
            "r_squared": self.r_squared(y_true, y_pred),
        }


class ClassificationMetrics:
    """Comprehensive classification metrics."""

    def accuracy(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """Classification accuracy."""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    def precision(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """Precision (positive predictive value)."""
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """Recall (sensitivity)."""
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f1(self, y_true: Sequence[int], y_pred: Sequence[int]) -> float:
        """F1 score."""
        prec = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def compute_all(self, y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
        """Compute all classification metrics."""
        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1": self.f1(y_true, y_pred),
        }


class CrossValidator:
    """K-fold cross-validation for model evaluation."""

    def __init__(self, k: int = 5):
        self.k = k

    def split(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> List[Tuple[List, List, List, List]]:
        """Generate k-fold train/test splits."""
        n = len(X)
        indices = list(range(n))
        random.shuffle(indices)
        fold_size = n // self.k
        folds = []
        for i in range(self.k):
            test_indices = indices[i * fold_size : (i + 1) * fold_size]
            train_indices = [j for j in indices if j not in test_indices]
            train_X = [X[j] for j in train_indices]
            train_y = [y[j] for j in train_indices]
            test_X = [X[j] for j in test_indices]
            test_y = [y[j] for j in test_indices]
            folds.append((train_X, train_y, test_X, test_y))
        return folds

    def cross_validate(
        self,
        model_factory: Callable[[], SimpleMLModel],
        X: Sequence[Sequence[float]],
        y: Sequence[float],
    ) -> List[float]:
        """Run cross-validation and return R^2 scores for each fold."""
        scores = []
        for train_X, train_y, test_X, test_y in self.split(X, y):
            model = model_factory()
            model.train(train_X, train_y)
            predictions = [model.predict(row).value for row in test_X]
            metrics = RegressionMetrics()
            r2 = metrics.r_squared(test_y, predictions)
            scores.append(r2)
        return scores


class FeatureEngine:
    """Feature engineering for agricultural data."""

    def __init__(self):
        self._features: Dict[str, float] = {}

    @property
    def feature_names(self) -> List[str]:
        """Names of computed features."""
        return list(self._features.keys())

    def compute_growing_degree_days(self, temps: Sequence[float], base_temp: float = 10.0) -> float:
        """Compute growing degree days from temperature series."""
        gdd = sum(max(0.0, t - base_temp) for t in temps)
        self._features["gdd"] = gdd
        return gdd

    def compute_vpd(self, temperature: float, humidity: float) -> float:
        """Compute vapor pressure deficit (kPa)."""
        import math

        es = 0.6108 * math.exp(17.27 * temperature / (temperature + 237.3))
        ea = es * (humidity / 100.0)
        vpd = es - ea
        self._features["vpd"] = vpd
        return vpd

    def compute_trend(self, values: Sequence[float]) -> float:
        """Compute linear trend (slope) of a value series."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        slope = numerator / denominator
        self._features["trend"] = slope
        return slope

    def compute_all(self) -> Dict[str, float]:
        """Return all computed features."""
        return dict(self._features)


class ModelRegistry:
    """Central registry for trained models with versioning."""

    def __init__(self):
        self._models: Dict[str, Dict[str, SimpleMLModel]] = {}

    def register(self, name: str, model: SimpleMLModel, version: Optional[str] = None) -> None:
        """Register a model with optional version."""
        ver = version or model.version
        if name not in self._models:
            self._models[name] = {}
        self._models[name][ver] = model
        logger.info("Registered model %s version %s", name, ver)

    def get(self, name: str, version: Optional[str] = None) -> Optional[SimpleMLModel]:
        """Get a registered model by name and optional version."""
        if name not in self._models:
            return None
        if version is not None:
            return self._models[name].get(version)
        # Return latest version
        versions = sorted(self._models[name].keys())
        return self._models[name][versions[-1]] if versions else None

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions for a model."""
        if name not in self._models:
            return []
        return sorted(self._models[name].keys())


class YieldPredictor:
    """High-level yield prediction using a pluggable regression model."""

    DEFAULT_FEATURES = [
        "soil_moisture",
        "temperature",
        "nutrient_level",
        "sunlight_hours",
        "rainfall_mm",
    ]

    def __init__(self, model: Optional[SimpleMLModel] = None):
        self.model = model or LinearRegressionModel(name="yield_linear_regression")

    def train(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Train the underlying model."""
        names = list(feature_names) if feature_names else self.DEFAULT_FEATURES
        if isinstance(self.model, LinearRegressionModel):
            self.model.train(X, y, feature_names=names)
        else:
            self.model.train(X, y)

    def predict_yield(self, features: Sequence[float]) -> PredictionResult:
        """Predict crop yield (tonnes per hectare)."""
        return self.model.predict(features)

    def predict_from_dict(self, features: Dict[str, float]) -> PredictionResult:
        """Predict from a named feature dictionary."""
        ordered = [features[name] for name in self.model.feature_names]
        return self.model.predict(ordered)

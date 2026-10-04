"""ML model interfaces and yield prediction models for decision support."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        # Augmented matrix
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            # Partial pivot
            pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[pivot][col]) < 1e-12:
                raise ValueError("Singular matrix in linear regression solve")
            M[col], M[pivot] = M[pivot], M[col]
            for row in range(col + 1, n):
                factor = M[row][col] / M[col][col]
                for k in range(col, n + 1):
                    M[row][k] -= factor * M[col][k]
        # Back substitution
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
            raise ValueError(
                f"Need at least {n_features + 1} samples for {n_features} features"
            )

        self._feature_names = list(feature_names) if feature_names else [
            f"x{i}" for i in range(n_features)
        ]
        if len(self._feature_names) != n_features:
            raise ValueError("feature_names length must match feature count")

        # Build normal equations with an intercept column of ones.
        # A = X_aug^T X_aug, b = X_aug^T y where X_aug = [1, X]
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
            self.name, self.version, n_features, len(X), self._r_squared,
        )

    def _compute_r_squared(
        self, X: Sequence[Sequence[float]], y: Sequence[float]
    ) -> float:
        """Compute R^2 of the fitted model on the training data."""
        n = len(y)
        mean_y = sum(y) / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        if ss_tot < 1e-12:
            return 1.0
        ss_res = sum(
            (yi - self._raw_predict(row)) ** 2 for row, yi in zip(X, y)
        )
        return max(0.0, 1.0 - ss_res / ss_tot)

    def _raw_predict(self, features: Sequence[float]) -> float:
        """Compute the linear combination without validation."""
        return self.intercept + sum(
            w * x for w, x in zip(self.weights, features)
        )

    def predict(self, features: Sequence[float]) -> PredictionResult:
        """Predict the target value for a feature vector."""
        self.validate_features(features)
        value = self._raw_predict(features)
        # Confidence derived from training fit: R^2 clipped to [0, 1].
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
        return {
            name: w / total for name, w in zip(self._feature_names, abs_weights)
        }


class YieldPredictor:
    """High-level yield prediction using a pluggable regression model."""

    # Agronomic feature names expected by the default yield model.
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

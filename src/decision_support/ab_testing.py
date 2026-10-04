"""A/B testing framework for decision support experiments."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """An A/B test experiment."""

    name: str
    variants: List[str]
    weights: List[float]
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    _outcomes: Dict[str, List[float]] = field(default_factory=dict)
    _assignments: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        if len(self.weights) != len(self.variants):
            raise ValueError("Weights must match variants count")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise ValueError("start_time must be before end_time")
        for v in self.variants:
            self._outcomes[v] = []

    def is_active(self, now: Optional[float] = None) -> bool:
        """Check if the experiment is currently active."""
        now = now or time.time()
        if self.start_time is not None and now < self.start_time:
            return False
        if self.end_time is not None and now > self.end_time:
            return False
        return True


class ABTestManager:
    """Manages A/B testing experiments for recommendation strategies."""

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        variants: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Experiment:
        """Create a new A/B test experiment."""
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")
        if weights is None:
            n = len(variants)
            weights = [1.0 / n] * n
        exp = Experiment(
            name=name,
            variants=list(variants),
            weights=list(weights),
            start_time=start_time,
            end_time=end_time,
        )
        self._experiments[name] = exp
        logger.info("Created experiment '%s' with variants %s", name, list(variants))
        return exp

    def assign_variant(self, experiment_name: str, user_id: str) -> str:
        """Assign a user to a variant using deterministic hashing."""
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")
        now = time.time()
        if not exp.is_active(now):
            raise ValueError(f"Experiment '{experiment_name}' is expired or not yet started")
        # Deterministic assignment via hash
        key = f"{experiment_name}:{user_id}"
        hash_val = int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16)
        # Map hash to variant based on weights
        normalized = (hash_val % 10000) / 10000.0
        cumulative = 0.0
        for variant, weight in zip(exp.variants, exp.weights):
            cumulative += weight
            if normalized <= cumulative:
                exp._assignments[user_id] = variant
                return variant
        # Fallback to last variant
        exp._assignments[user_id] = exp.variants[-1]
        return exp.variants[-1]

    def record_outcome(self, experiment_name: str, variant: str, outcome: float) -> None:
        """Record an outcome for a variant in an experiment."""
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")
        if variant not in exp.variants:
            raise ValueError(f"Variant '{variant}' not in experiment '{experiment_name}'")
        exp._outcomes[variant].append(outcome)

    def get_results(self, experiment_name: str) -> Dict[str, Dict[str, float]]:
        """Get statistical results for an experiment."""
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_name}' not found")
        results = {}
        for variant in exp.variants:
            outcomes = exp._outcomes[variant]
            n = len(outcomes)
            if n == 0:
                results[variant] = {"count": 0, "mean": 0.0, "variance": 0.0, "std": 0.0}
            elif n == 1:
                results[variant] = {"count": 1, "mean": outcomes[0], "variance": 0.0, "std": 0.0}
            else:
                mean = sum(outcomes) / n
                variance = sum((x - mean) ** 2 for x in outcomes) / (n - 1)
                results[variant] = {
                    "count": n,
                    "mean": mean,
                    "variance": variance,
                    "std": math.sqrt(variance),
                }
        return results

    def is_significant(self, experiment_name: str, alpha: float = 0.05) -> bool:
        """Check if the difference between variants is statistically significant using a t-test."""
        results = self.get_results(experiment_name)
        exp = self._experiments[experiment_name]
        # Need at least 2 variants with data
        variants_with_data = [v for v in exp.variants if results[v]["count"] >= 2]
        if len(variants_with_data) < 2:
            return False
        # Compare first two variants with data
        v1, v2 = variants_with_data[0], variants_with_data[1]
        r1, r2 = results[v1], results[v2]
        # Welch's t-test
        if r1["std"] < 1e-12 and r2["std"] < 1e-12:
            # Both have zero variance — check if means differ
            return abs(r1["mean"] - r2["mean"]) > 1e-12
        se = math.sqrt(r1["variance"] / r1["count"] + r2["variance"] / r2["count"])
        if se < 1e-12:
            return abs(r1["mean"] - r2["mean"]) > 1e-12
        t_stat = abs(r1["mean"] - r2["mean"]) / se
        # Critical value approximation for two-tailed test
        # For large samples, t-distribution approximates normal
        # Use 1.96 for alpha=0.05 (two-tailed)
        critical = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        return t_stat > critical

    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        return list(self._experiments.keys())

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        return self._experiments.get(name)

"""IoT sensor placement optimization (greedy Set Cover)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlacementInstance:
    """Sensor placement problem instance."""
    sensor_positions: List[Tuple[float, float]]
    target_positions: List[Tuple[float, float]]
    coverage_radius: float

    def __post_init__(self):
        if self.coverage_radius <= 0:
            raise ValueError("Coverage radius must be positive")


@dataclass
class PlacementResult:
    """Sensor placement result."""
    selected_sensors: List[int]
    coverage_ratio: float
    algorithm: str
    covered_targets: Optional[Set[int]] = None


class SensorPlacement:
    """IoT sensor placement optimizer using greedy Set Cover."""

    def __init__(self, algorithm: str = "greedy"):
        self.algorithm = algorithm

    def optimize(self, instance: PlacementInstance) -> PlacementResult:
        """Optimize sensor placement to cover all targets."""
        if not instance.sensor_positions or not instance.target_positions:
            return PlacementResult(
                selected_sensors=[],
                coverage_ratio=0.0,
                algorithm=self.algorithm,
                covered_targets=set(),
            )

        if self.algorithm == "greedy":
            return self._greedy(instance)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _greedy(self, instance: PlacementInstance) -> PlacementResult:
        """Greedy Set Cover: iteratively select sensor covering most uncovered targets."""
        n_sensors = len(instance.sensor_positions)
        n_targets = len(instance.target_positions)
        radius = instance.coverage_radius

        # Pre-compute coverage: sensor i covers which targets
        coverage = []
        for i in range(n_sensors):
            covered = set()
            for j in range(n_targets):
                dx = instance.sensor_positions[i][0] - instance.target_positions[j][0]
                dy = instance.sensor_positions[i][1] - instance.target_positions[j][1]
                if math.sqrt(dx * dx + dy * dy) <= radius:
                    covered.add(j)
            coverage.append(covered)

        # Greedy selection
        selected = []
        uncovered = set(range(n_targets))

        while uncovered:
            # Find sensor covering most uncovered targets
            best_sensor = -1
            best_count = 0
            for i in range(n_sensors):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_sensor = i

            if best_sensor == -1 or best_count == 0:
                break  # No more coverage possible

            selected.append(best_sensor)
            uncovered -= coverage[best_sensor]

        covered_targets = set(range(n_targets)) - uncovered
        coverage_ratio = len(covered_targets) / n_targets if n_targets > 0 else 0.0

        return PlacementResult(
            selected_sensors=selected,
            coverage_ratio=coverage_ratio,
            algorithm="greedy",
            covered_targets=covered_targets,
        )

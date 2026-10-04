"""Coverage path planning for agricultural drones (boustrophedon pattern)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CoverageInstance:
    """Coverage planning problem instance."""

    field_boundary: List[Tuple[float, float]]
    swath_width: float
    start_point: Tuple[float, float]

    def __post_init__(self):
        if len(self.field_boundary) < 3:
            raise ValueError("Field boundary must have at least 3 points")
        if self.swath_width <= 0:
            raise ValueError("Swath width must be positive")


@dataclass
class CoverageResult:
    """Coverage planning result."""

    path: List[Tuple[float, float]]
    total_distance: float
    coverage_ratio: float
    algorithm: str
    num_passes: int = 0


class CoveragePlanner:
    """Coverage path planner using boustrophedon (back-and-forth) pattern."""

    def __init__(self, algorithm: str = "boustrophedon"):
        self.algorithm = algorithm

    def plan(self, instance: CoverageInstance) -> CoverageResult:
        """Plan coverage path for a field boundary."""
        if not instance.field_boundary:
            return CoverageResult(
                path=[],
                total_distance=0.0,
                coverage_ratio=0.0,
                algorithm=self.algorithm,
            )

        if self.algorithm == "boustrophedon":
            return self._boustrophedon(instance)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _boustrophedon(self, instance: CoverageInstance) -> CoverageResult:
        """Boustrophedon (lawnmower) coverage pattern."""
        boundary = instance.field_boundary
        swath = instance.swath_width

        # Find bounding box
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Generate parallel lines in Y direction
        path = []
        y = min_y
        direction = 1  # 1 = left to right, -1 = right to left
        num_passes = 0

        while y <= max_y:
            if direction == 1:
                path.append((min_x, y))
                path.append((max_x, y))
            else:
                path.append((max_x, y))
                path.append((min_x, y))
            num_passes += 1
            y += swath
            direction *= -1

        # Calculate total distance
        total_distance = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total_distance += math.sqrt(dx * dx + dy * dy)

        # Estimate coverage ratio (simplified)
        field_area = (max_x - min_x) * (max_y - min_y)
        covered_area = num_passes * swath * (max_x - min_x)
        coverage_ratio = min(1.0, covered_area / field_area) if field_area > 0 else 0.0

        return CoverageResult(
            path=path,
            total_distance=total_distance,
            coverage_ratio=coverage_ratio,
            algorithm="boustrophedon",
            num_passes=num_passes,
        )

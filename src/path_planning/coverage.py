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
        """Boustrophedon (lawnmower) coverage pattern.

        The path starts at the configured start_point and then follows
        a back-and-forth pattern covering the field bounding box.
        Passes are generated both above and below the start point.
        """
        boundary = instance.field_boundary
        swath = instance.swath_width
        start = instance.start_point

        # Find bounding box
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Generate pass rows
        rows: List[float] = []
        y = min_y
        while y <= max_y:
            rows.append(y)
            y += swath

        # Find the row closest to start y
        start_y = start[1]
        closest_row_idx = min(range(len(rows)), key=lambda i: abs(rows[i] - start_y))

        # Generate path
        path = [start]
        num_passes = 0

        # Direction for each row: even index = left to right, odd index = right to left
        def row_direction(idx: int) -> int:
            return 1 if idx % 2 == 0 else -1

        # First pass: from start to the edge at the start row
        start_row = rows[closest_row_idx]
        if row_direction(closest_row_idx) == 1:
            path.append((max_x, start_row))
        else:
            path.append((min_x, start_row))
        num_passes += 1

        # Go up from the start row
        for i in range(closest_row_idx + 1, len(rows)):
            y = rows[i]
            if row_direction(i) == 1:
                path.append((min_x, y))
                path.append((max_x, y))
            else:
                path.append((max_x, y))
                path.append((min_x, y))
            num_passes += 1

        # Go down from the row below the start row
        for i in range(closest_row_idx - 1, -1, -1):
            y = rows[i]
            if row_direction(i) == 1:
                path.append((min_x, y))
                path.append((max_x, y))
            else:
                path.append((max_x, y))
                path.append((min_x, y))
            num_passes += 1

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

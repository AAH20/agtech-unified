"""Obstacle field for path planning."""
from __future__ import annotations
import math
from typing import List, Tuple


class ObstacleField:
    """Manages circular obstacles and provides path avoidance."""

    def __init__(self, obstacles: List[Tuple[float, float, float]] = None):
        self.obstacles = obstacles or []

    def is_obstacle(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside any obstacle."""
        for obs in self.obstacles:
            if self._point_in_obstacle(point, obs):
                return True
        return False

    def avoid_path(self, start: Tuple[float, float], end: Tuple[float, float],
                   obstacle: Tuple[float, float, float]) -> List[Tuple[float, float]]:
        """Generate waypoints to detour around a circular obstacle."""
        cx, cy, r = obstacle
        b = r + 1.0  # safety buffer
        return [(cx - b, cy - b), (cx + b, cy - b),
                (cx + b, cy + b), (cx - b, cy + b)]

    def _point_in_obstacle(self, point: Tuple[float, float],
                           obstacle: Tuple[float, float, float]) -> bool:
        """Check if point is inside a circular obstacle."""
        cx, cy, r = obstacle
        return math.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2) <= r

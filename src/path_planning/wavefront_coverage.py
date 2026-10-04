"""Wavefront (brushfire) coverage algorithm for grid-based path planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class WavefrontResult:
    """Result of wavefront coverage planning."""

    path: List[Tuple[int, int]] = field(default_factory=list)
    distances: Dict[Tuple[int, int], int] = field(default_factory=dict)
    num_covered: int = 0
    coverage_ratio: float = 0.0
    algorithm: str = "wavefront"


class WavefrontCoveragePlanner:
    """Wavefront (brushfire) coverage path planner.

    Propagates a wave from the start cell, assigning distance values to all
    reachable cells. The coverage path follows the wavefront in order of
    increasing distance.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._obstacles: set = set()

    def set_obstacle(self, point: Tuple[int, int]) -> None:
        """Mark a cell as an obstacle."""
        self._obstacles.add(point)

    def clear_obstacle(self, point: Tuple[int, int]) -> None:
        """Clear an obstacle from a cell."""
        self._obstacles.discard(point)

    def is_obstacle(self, point: Tuple[int, int]) -> bool:
        """Check if a cell is an obstacle or out of bounds."""
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return point in self._obstacles

    def plan(self, start: Tuple[int, int]) -> WavefrontResult:
        """Run wavefront coverage from the given start cell.

        Args:
            start: Starting cell (x, y).

        Returns:
            WavefrontResult with path, distance field, and coverage stats.
        """
        if self.is_obstacle(start):
            return WavefrontResult()

        # BFS to compute distances
        distances: Dict[Tuple[int, int], int] = {}
        queue = deque([start])
        distances[start] = 0

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            for neighbor in self._neighbors(current):
                if neighbor not in distances and not self.is_obstacle(neighbor):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        # Generate continuous path using DFS (consecutive cells are adjacent)
        path = self._generate_path(start, distances)

        num_covered = len(path)
        total_cells = self.width * self.height
        coverage_ratio = num_covered / total_cells if total_cells > 0 else 0.0

        return WavefrontResult(
            path=path,
            distances=distances,
            num_covered=num_covered,
            coverage_ratio=coverage_ratio,
        )

    def _generate_path(
        self, start: Tuple[int, int], distances: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, int]]:
        """Generate a continuous path visiting all reachable cells using DFS.

        Neighbors are visited in order of increasing distance from start,
        which keeps the path roughly wavefront-ordered while guaranteeing
        consecutive cells are adjacent.
        """
        path: List[Tuple[int, int]] = []
        visited: set = set()

        def dfs(cell: Tuple[int, int]) -> None:
            visited.add(cell)
            path.append(cell)
            neighbors = [
                n for n in self._neighbors(cell) if n not in visited and not self.is_obstacle(n)
            ]
            neighbors.sort(key=lambda n: distances.get(n, float("inf")))
            for n in neighbors:
                if n not in visited:
                    dfs(n)

        dfs(start)
        return path

    def _neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-connected neighboring cells."""
        x, y = point
        return [
            (x, y + 1),
            (x + 1, y),
            (x, y - 1),
            (x - 1, y),
        ]

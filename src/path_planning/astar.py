"""A* pathfinding algorithm for grid-based path planning."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class GridCell:
    """A cell in the grid with A* pathfinding state."""

    x: int
    y: int
    g: float = float("inf")
    h: float = 0.0
    parent: Optional[Tuple[int, int]] = None

    @property
    def f_score(self) -> float:
        """Total estimated cost (g + h)."""
        return self.g + self.h


class AStarPlanner:
    """A* pathfinder on a 2D grid with obstacles."""

    def __init__(
        self,
        width: int,
        height: int,
        allow_diagonal: bool = True,
        heuristic: str = "euclidean",
    ):
        if heuristic not in ("manhattan", "euclidean", "octile"):
            raise ValueError(f"Unknown heuristic: {heuristic}")
        self.width = width
        self.height = height
        self.allow_diagonal = allow_diagonal
        self.heuristic_name = heuristic
        self._obstacles: set = set()

    def set_obstacle(self, point: Tuple[int, int]) -> None:
        """Mark a cell as an obstacle."""
        self._obstacles.add(point)

    def clear_obstacle(self, point: Tuple[int, int]) -> None:
        """Clear an obstacle from a cell."""
        self._obstacles.discard(point)

    def add_obstacles(self, obstacles: List[Tuple[int, int]]) -> None:
        """Add multiple obstacles at once."""
        self._obstacles.update(obstacles)

    def is_obstacle(self, point: Tuple[int, int]) -> bool:
        """Check if a cell is an obstacle or out of bounds."""
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return point in self._obstacles

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Find the shortest path from start to goal.

        Returns a list of (x, y) tuples from start to goal, or an empty list
        if no path exists.
        """
        if self.is_obstacle(start) or self.is_obstacle(goal):
            return []

        if start == goal:
            return [start]

        # Initialize
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0
        start_h = self._heuristic(start, goal)
        start_cell = GridCell(x=start[0], y=start[1], g=0.0, h=start_h)
        heapq.heappush(open_set, (start_cell.f_score, counter, start))

        cells: Dict[Tuple[int, int], GridCell] = {start: start_cell}
        closed_set: set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(cells, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self._neighbors(current):
                if neighbor in closed_set or self.is_obstacle(neighbor):
                    continue

                move_cost = self._move_cost(current, neighbor)
                g_score = cells[current].g + move_cost

                neighbor_cell = cells.get(neighbor)
                if neighbor_cell is None:
                    neighbor_cell = GridCell(x=neighbor[0], y=neighbor[1])
                    cells[neighbor] = neighbor_cell

                if g_score < neighbor_cell.g:
                    neighbor_cell.g = g_score
                    neighbor_cell.h = self._heuristic(neighbor, goal)
                    neighbor_cell.parent = current
                    counter += 1
                    heapq.heappush(open_set, (neighbor_cell.f_score, counter, neighbor))

        return []

    def _heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> float:
        """Compute heuristic distance between two points."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        if self.heuristic_name == "manhattan":
            return float(dx + dy)
        elif self.heuristic_name == "euclidean":
            return math.hypot(dx, dy)
        elif self.heuristic_name == "octile":
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic_name}")

    def _neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get neighboring cells (4 or 8 directions)."""
        x, y = point
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
        ]
        if self.allow_diagonal:
            directions.extend(
                [
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ]
            )
        return [(x + dx, y + dy) for dx, dy in directions]

    def _move_cost(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> float:
        """Cost of moving from cell a to cell b."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2)
        return 1.0

    def _reconstruct_path(
        self,
        cells: Dict[Tuple[int, int], GridCell],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start by following parent pointers."""
        path = []
        current: Optional[Tuple[int, int]] = goal
        while current is not None:
            path.append(current)
            cell = cells.get(current)
            current = cell.parent if cell else None
        path.reverse()
        return path

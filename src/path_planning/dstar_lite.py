"""D* Lite incremental replanning algorithm for unknown/changing environments.

D* Lite is an incremental heuristic search algorithm that efficiently replans
when the environment changes. It maintains cost estimates and only re-expands
affected cells, making it much faster than re-running A* from scratch.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DStarLiteResult:
    """Result of D* Lite path planning."""

    success: bool
    path: List[Tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0
    nodes_expanded: int = 0


class DStarLite:
    """D* Lite incremental path planner on a 2D grid.

    Supports dynamic obstacle addition/removal and unknown cells.
    Replanning after environment changes is faster than full A* replanning.
    """

    def __init__(
        self,
        width: int,
        height: int,
        allow_diagonal: bool = True,
    ):
        self.width = width
        self.height = height
        self.allow_diagonal = allow_diagonal
        self._obstacles: Set[Tuple[int, int]] = set()
        self._unknown: Set[Tuple[int, int]] = set()

        # D* Lite state
        self._g: Dict[Tuple[int, int], float] = {}
        self._rhs: Dict[Tuple[int, int], float] = {}
        self._open_set: List[Tuple[Tuple[float, float], Tuple[int, int]]] = []
        self._open_set_hash: Set[Tuple[int, int]] = set()
        self._km: float = 0.0
        self._goal: Optional[Tuple[int, int]] = None
        self._start: Optional[Tuple[int, int]] = None
        self._nodes_expanded: int = 0

    def set_obstacle(self, point: Tuple[int, int]) -> None:
        """Mark a cell as an obstacle and trigger replanning update."""
        if point not in self._obstacles:
            self._obstacles.add(point)
            # Only update neighbors - the obstacle itself is not a valid node
            for neighbor in self._neighbors(point):
                self._update_vertex(neighbor)

    def clear_obstacle(self, point: Tuple[int, int]) -> None:
        """Remove an obstacle and trigger replanning update."""
        if point in self._obstacles:
            self._obstacles.discard(point)
            self._update_vertex(point)
            for neighbor in self._neighbors(point):
                self._update_vertex(neighbor)

    def set_unknown(self, point: Tuple[int, int]) -> None:
        """Mark a cell as unknown (treated as obstacle until explored)."""
        self._unknown.add(point)

    def clear_unknown(self, point: Tuple[int, int]) -> None:
        """Mark an unknown cell as known (free)."""
        self._unknown.discard(point)

    def is_obstacle(self, point: Tuple[int, int]) -> bool:
        """Check if a cell is an obstacle or out of bounds."""
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return point in self._obstacles

    def is_unknown(self, point: Tuple[int, int]) -> bool:
        """Check if a cell is marked as unknown."""
        return point in self._unknown

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> DStarLiteResult:
        """Find the shortest path from start to goal.

        Uses D* Lite incremental search. If the environment has changed
        since the last call, only affected cells are re-expanded.

        Returns:
            DStarLiteResult with success flag, path, cost, and nodes expanded.
        """
        if self.is_obstacle(start) or self.is_obstacle(goal):
            return DStarLiteResult(success=False)

        if start == goal:
            return DStarLiteResult(success=True, path=[start], cost=0.0)

        # If this is a completely new query (different start/goal), reset
        if self._goal != goal or self._start != start:
            self._initialize(start, goal)

        # Compute shortest path
        self._compute_shortest_path()

        # Extract path
        if self._g.get(start, float("inf")) == float("inf"):
            return DStarLiteResult(
                success=False,
                nodes_expanded=self._nodes_expanded,
            )

        path = self._extract_path(start, goal)
        cost = self._g[start]

        return DStarLiteResult(
            success=True,
            path=path,
            cost=cost,
            nodes_expanded=self._nodes_expanded,
        )

    def _initialize(self, start: Tuple[int, int], goal: Tuple[int, int]) -> None:
        """Initialize D* Lite state for a new query."""
        self._g.clear()
        self._rhs.clear()
        self._open_set.clear()
        self._open_set_hash.clear()
        self._km = 0.0
        self._goal = goal
        self._start = start
        self._nodes_expanded = 0

        # Initialize goal
        self._rhs[goal] = 0.0
        key = self._calculate_key(goal)
        heapq.heappush(self._open_set, (key, goal))
        self._open_set_hash.add(goal)

    def _compute_shortest_path(self) -> None:
        """Run D* Lite main loop to compute shortest path."""
        while self._open_set:
            key_top = self._open_set[0][0]
            start_key = (
                self._calculate_key(self._start)
                if self._start is not None
                else (float("inf"), float("inf"))
            )

            if key_top >= start_key and self._rhs.get(self._start, float("inf")) == self._g.get(
                self._start, float("inf")
            ):
                break

            # Pop the top element
            old_key, u = heapq.heappop(self._open_set)
            self._open_set_hash.discard(u)
            self._nodes_expanded += 1

            new_key = self._calculate_key(u)

            if old_key < new_key:
                # Key has changed, reinsert with new key
                heapq.heappush(self._open_set, (new_key, u))
                self._open_set_hash.add(u)
            elif self._g.get(u, float("inf")) > self._rhs.get(u, float("inf")):
                # Overconsistent: g > rhs
                self._g[u] = self._rhs[u]
                for neighbor in self._neighbors(u):
                    self._update_vertex(neighbor)
            else:
                # Underconsistent: g < rhs
                self._g[u] = float("inf")
                self._update_vertex(u)
                for neighbor in self._neighbors(u):
                    self._update_vertex(neighbor)

    def _update_vertex(self, u: Tuple[int, int]) -> None:
        """Update the priority queue entry for vertex u."""
        if u != self._goal:
            # rhs(u) = min over successors s of (c(u, s) + g(s))
            min_rhs = float("inf")
            for neighbor in self._neighbors(u):
                cost = self._edge_cost(u, neighbor)
                g_neighbor = self._g.get(neighbor, float("inf"))
                if g_neighbor + cost < min_rhs:
                    min_rhs = g_neighbor + cost
            self._rhs[u] = min_rhs

        # Remove from open set if present
        if u in self._open_set_hash:
            self._open_set_hash.discard(u)
            # Rebuild heap without u (lazy deletion approach)
            self._open_set = [(k, v) for k, v in self._open_set if v != u]
            heapq.heapify(self._open_set)

        # Add to open set if inconsistent
        if self._g.get(u, float("inf")) != self._rhs.get(u, float("inf")):
            key = self._calculate_key(u)
            heapq.heappush(self._open_set, (key, u))
            self._open_set_hash.add(u)

    def _calculate_key(self, u: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate the priority key for vertex u."""
        g_u = self._g.get(u, float("inf"))
        rhs_u = self._rhs.get(u, float("inf"))
        min_g_rhs = min(g_u, rhs_u)

        start = self._start
        if start is None or self._goal is None:
            return (float("inf"), float("inf"))

        h = self._heuristic(u, start)
        return (min_g_rhs + self._km + h, min_g_rhs)

    def _extract_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Extract path from start to goal by following lowest-cost neighbors."""
        path = [start]
        current = start
        visited = {start}

        while current != goal:
            best_neighbor = None
            best_cost = float("inf")

            for neighbor in self._neighbors(current):
                if neighbor in visited:
                    continue
                cost = self._edge_cost(current, neighbor) + self._g.get(neighbor, float("inf"))
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor

            if best_neighbor is None:
                # No valid neighbor found - should not happen if path exists
                break

            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

            # Safety limit
            if len(path) > self.width * self.height:
                break

        return path

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Compute heuristic distance between two points."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if self.allow_diagonal:
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        return float(dx + dy)

    def _edge_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Cost of moving from cell a to cell b."""
        if self.is_obstacle(b) or self.is_unknown(b):
            return float("inf")
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2)
        return 1.0

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

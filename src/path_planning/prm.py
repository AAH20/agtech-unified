"""PRM (Probabilistic Roadmap) multi-query path planning.

PRM is a sampling-based planner that builds a roadmap of the free configuration
space once, then answers multiple queries efficiently. Ideal for repeated
navigation in static or slowly-changing environments.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PRMConfig:
    """Configuration for PRM planner."""

    num_samples: int = 200
    k_neighbors: int = 8
    max_edge_length: float = 5.0
    seed: Optional[int] = None


@dataclass
class PRMResult:
    """Result of PRM path planning."""

    success: bool
    path: List[Tuple[float, float]] = field(default_factory=list)
    cost: float = 0.0
    nodes_in_roadmap: int = 0


class PRMPlanner:
    """PRM path planner for continuous 2D environments with circular obstacles.

    Builds a probabilistic roadmap once and reuses it for multiple queries.
    Each query connects start/goal to the roadmap and runs Dijkstra search.
    """

    def __init__(self):
        self.roadmap_nodes: List[Tuple[float, float]] = []
        self.roadmap_edges: Dict[int, List[Tuple[int, float]]] = {}
        self._bounds: Optional[Tuple[float, float, float, float]] = None
        self._obstacles: List[Tuple[float, float, float]] = []
        self._rng: Optional[random.Random] = None

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        config: Optional[PRMConfig] = None,
    ) -> PRMResult:
        """Plan a path from start to goal using the probabilistic roadmap.

        Args:
            start: Starting point (x, y).
            goal: Goal point (x, y).
            obstacles: List of (x, y, r) circular obstacles.
            config: PRM configuration parameters.

        Returns:
            PRMResult with success flag, path, cost, and roadmap size.
        """
        cfg = config or PRMConfig()

        # Check if start or goal is inside an obstacle
        if self._point_in_obstacles(start, obstacles) or self._point_in_obstacles(goal, obstacles):
            return PRMResult(success=False)

        # Handle start == goal
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 1e-6:
            return PRMResult(success=True, path=[start], cost=0.0)

        # Build or rebuild roadmap if needed
        if not self.roadmap_nodes or self._obstacles != obstacles:
            self._build_roadmap(start, goal, obstacles, cfg)

        # Connect start and goal to roadmap
        start_idx = self._add_temporary_node(start)
        goal_idx = self._add_temporary_node(goal)

        # Run Dijkstra from start to goal
        path_indices = self._dijkstra(start_idx, goal_idx)

        if path_indices is None:
            return PRMResult(
                success=False,
                nodes_in_roadmap=len(self.roadmap_nodes),
            )

        # Extract path
        path = [self.roadmap_nodes[i] for i in path_indices]
        cost = self._compute_path_cost(path)

        # Remove temporary nodes (in reverse order to maintain indices)
        if start_idx > goal_idx:
            self._remove_temporary_node(start_idx)
            self._remove_temporary_node(goal_idx)
        else:
            self._remove_temporary_node(goal_idx)
            self._remove_temporary_node(start_idx)

        return PRMResult(
            success=True,
            path=path,
            cost=cost,
            nodes_in_roadmap=len(self.roadmap_nodes),
        )

    def _build_roadmap(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        config: PRMConfig,
    ) -> None:
        """Build the probabilistic roadmap."""
        self._obstacles = obstacles
        self._rng = random.Random(config.seed)

        # Compute bounds
        all_points = [start, goal]
        for ox, oy, r in obstacles:
            all_points.append((ox, oy))

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        margin = 2.0
        self._bounds = (
            min(xs) - margin,
            max(xs) + margin,
            min(ys) - margin,
            max(ys) + margin,
        )

        # Sample free nodes
        self.roadmap_nodes = []
        self.roadmap_edges = {}

        # Don't add start and goal here - they'll be added as temporary nodes
        # Sample random free nodes
        attempts = 0
        max_attempts = config.num_samples * 10
        while len(self.roadmap_nodes) < config.num_samples and attempts < max_attempts:
            attempts += 1
            x = self._rng.uniform(self._bounds[0], self._bounds[1])
            y = self._rng.uniform(self._bounds[2], self._bounds[3])
            point = (x, y)

            if not self._point_in_obstacles(point, obstacles):
                self.roadmap_nodes.append(point)

        # Build edges between k-nearest neighbors
        n = len(self.roadmap_nodes)
        for i in range(n):
            self.roadmap_edges[i] = []
            distances = []
            for j in range(n):
                if i == j:
                    continue
                dist = math.hypot(
                    self.roadmap_nodes[i][0] - self.roadmap_nodes[j][0],
                    self.roadmap_nodes[i][1] - self.roadmap_nodes[j][1],
                )
                if dist <= config.max_edge_length:
                    distances.append((dist, j))

            distances.sort()
            for dist, j in distances[: config.k_neighbors]:
                # Check if edge is collision-free
                if self._is_collision_free(self.roadmap_nodes[i], self.roadmap_nodes[j], obstacles):
                    self.roadmap_edges[i].append((j, dist))

    def _add_temporary_node(self, point: Tuple[float, float]) -> int:
        """Add a temporary node (start or goal) to the roadmap."""
        idx = len(self.roadmap_nodes)
        self.roadmap_nodes.append(point)
        self.roadmap_edges[idx] = []

        # Connect to k-nearest neighbors
        distances = []
        for j in range(idx):
            dist = math.hypot(
                point[0] - self.roadmap_nodes[j][0],
                point[1] - self.roadmap_nodes[j][1],
            )
            if dist <= 10.0:  # Temporary nodes can connect further
                distances.append((dist, j))

        distances.sort()
        for dist, j in distances[:10]:
            if self._is_collision_free(point, self.roadmap_nodes[j], self._obstacles):
                self.roadmap_edges[idx].append((j, dist))
                # Add reverse edge
                if j not in self.roadmap_edges:
                    self.roadmap_edges[j] = []
                self.roadmap_edges[j].append((idx, dist))

        return idx

    def _remove_temporary_node(self, idx: int) -> None:
        """Remove a temporary node from the roadmap."""
        if idx < len(self.roadmap_nodes):
            # Remove edges pointing to this node
            for i in range(idx):
                if i in self.roadmap_edges:
                    self.roadmap_edges[i] = [(j, d) for j, d in self.roadmap_edges[i] if j != idx]
            # Remove the node's edges
            if idx in self.roadmap_edges:
                del self.roadmap_edges[idx]
            # Remove the node itself
            self.roadmap_nodes.pop(idx)
            # Update edge indices for nodes after the removed one
            new_edges: Dict[int, List[Tuple[int, float]]] = {}
            for i, edges in self.roadmap_edges.items():
                if i > idx:
                    new_edges[i - 1] = [(j - 1 if j > idx else j, d) for j, d in edges]
                else:
                    new_edges[i] = edges
            self.roadmap_edges = new_edges

    def _dijkstra(
        self,
        start_idx: int,
        goal_idx: int,
    ) -> Optional[List[int]]:
        """Run Dijkstra's algorithm on the roadmap graph."""
        dist: Dict[int, float] = {start_idx: 0.0}
        prev: Dict[int, Optional[int]] = {start_idx: None}
        pq: List[Tuple[float, int]] = [(0.0, start_idx)]
        visited: Set[int] = set()

        while pq:
            d, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == goal_idx:
                # Reconstruct path
                path = []
                current: Optional[int] = goal_idx
                while current is not None:
                    path.append(current)
                    current = prev.get(current)
                path.reverse()
                return path

            for v, edge_cost in self.roadmap_edges.get(u, []):
                if v in visited:
                    continue
                new_dist = d + edge_cost
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        return None

    def _compute_path_cost(self, path: List[Tuple[float, float]]) -> float:
        """Compute the total cost of a path."""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        return cost

    def _is_collision_free(
        self,
        from_point: Tuple[float, float],
        to_point: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> bool:
        """Check if the line segment from from_point to to_point is collision-free."""
        for ox, oy, r in obstacles:
            if self._segment_intersects_circle(from_point, to_point, (ox, oy, r)):
                return False
        return True

    def _segment_intersects_circle(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        circle: Tuple[float, float, float],
    ) -> bool:
        """Check if line segment AB intersects or is inside a circle."""
        cx, cy, r = circle

        # Check if either endpoint is inside the circle
        if math.hypot(a[0] - cx, a[1] - cy) <= r:
            return True
        if math.hypot(b[0] - cx, b[1] - cy) <= r:
            return True

        # Project circle center onto line segment
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        ab_len_sq = abx * abx + aby * aby

        if ab_len_sq < 1e-12:
            return False

        t = max(0.0, min(1.0, ((cx - a[0]) * abx + (cy - a[1]) * aby) / ab_len_sq))
        proj_x = a[0] + t * abx
        proj_y = a[1] + t * aby

        return math.hypot(proj_x - cx, proj_y - cy) <= r

    def _point_in_obstacles(
        self,
        point: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> bool:
        """Check if a point is inside any obstacle."""
        for ox, oy, r in obstacles:
            if math.hypot(point[0] - ox, point[1] - oy) <= r:
                return True
        return False

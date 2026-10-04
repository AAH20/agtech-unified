"""RRT (Rapidly-exploring Random Tree) path planning for continuous environments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RRTConfig:
    """Configuration for RRT planner."""

    max_iterations: int = 1000
    step_size: float = 1.0
    goal_tolerance: float = 0.5
    goal_bias: float = 0.1


@dataclass
class RRTNode:
    """A node in the RRT tree."""

    x: float
    y: float
    parent: Optional[RRTNode] = None


@dataclass
class RRTResult:
    """Result of RRT path planning."""

    success: bool
    path: List[Tuple[float, float]] = field(default_factory=list)
    iterations: int = 0
    nodes: List[RRTNode] = field(default_factory=list)


class RRTPlanner:
    """RRT path planner for continuous 2D environments with circular obstacles."""

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
        config: Optional[RRTConfig] = None,
        seed: Optional[int] = None,
    ) -> RRTResult:
        """Plan a path from start to goal avoiding circular obstacles.

        Args:
            start: Starting point (x, y).
            goal: Goal point (x, y).
            obstacles: List of (x, y, r) circular obstacles.
            config: RRT configuration parameters.
            seed: Random seed for reproducibility.

        Returns:
            RRTResult with success flag, path, iterations, and tree nodes.
        """
        cfg = config or RRTConfig()
        rng = random.Random(seed)

        # Check if start or goal is inside an obstacle
        if self._point_in_obstacles(start, obstacles) or self._point_in_obstacles(goal, obstacles):
            return RRTResult(success=False)

        # Handle start == goal
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < cfg.goal_tolerance:
            return RRTResult(
                success=True,
                path=[start],
                iterations=0,
                nodes=[RRTNode(start[0], start[1])],
            )

        # Initialize tree with start node
        start_node = RRTNode(start[0], start[1])
        nodes: List[RRTNode] = [start_node]

        for iteration in range(cfg.max_iterations):
            # Sample random point (with goal bias)
            if rng.random() < cfg.goal_bias:
                sample = goal
            else:
                sample = self._random_sample(start, goal, rng)

            # Find nearest node in tree
            nearest = self._nearest_node(nodes, sample)

            # Steer from nearest towards sample
            new_node = self._steer(nearest, sample, cfg.step_size)

            # Check if new node is collision-free
            if self._is_collision_free((nearest.x, nearest.y), (new_node.x, new_node.y), obstacles):
                new_node.parent = nearest
                nodes.append(new_node)

                # Check if we reached the goal
                dist_to_goal = math.hypot(new_node.x - goal[0], new_node.y - goal[1])
                if dist_to_goal <= cfg.goal_tolerance:
                    path = self._reconstruct_path(new_node)
                    path.append(goal)
                    return RRTResult(
                        success=True,
                        path=path,
                        iterations=iteration + 1,
                        nodes=nodes,
                    )

        return RRTResult(success=False, iterations=cfg.max_iterations, nodes=nodes)

    def _random_sample(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        rng: random.Random,
    ) -> Tuple[float, float]:
        """Sample a random point in the bounding box of start and goal."""
        min_x = min(start[0], goal[0]) - 5.0
        max_x = max(start[0], goal[0]) + 5.0
        min_y = min(start[1], goal[1]) - 5.0
        max_y = max(start[1], goal[1]) + 5.0
        return (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))

    def _nearest_node(self, nodes: List[RRTNode], point: Tuple[float, float]) -> RRTNode:
        """Find the nearest node in the tree to the given point."""
        return min(nodes, key=lambda n: math.hypot(n.x - point[0], n.y - point[1]))

    def _steer(
        self,
        from_node: RRTNode,
        to_point: Tuple[float, float],
        step_size: float,
    ) -> RRTNode:
        """Steer from a node towards a point with maximum step size."""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = math.hypot(dx, dy)

        if dist <= step_size:
            return RRTNode(to_point[0], to_point[1])

        ratio = step_size / dist
        new_x = from_node.x + dx * ratio
        new_y = from_node.y + dy * ratio
        return RRTNode(new_x, new_y)

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

    def _reconstruct_path(self, node: RRTNode) -> List[Tuple[float, float]]:
        """Reconstruct path from root to given node by following parent pointers."""
        path: List[Tuple[float, float]] = []
        current: Optional[RRTNode] = node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path

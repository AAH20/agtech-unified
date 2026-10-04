"""Spiral coverage pattern for agricultural drones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SpiralConfig:
    """Configuration for spiral coverage pattern."""

    spacing: float = 2.0
    direction: str = "outward"  # "outward" or "inward"
    turns: Optional[int] = None  # None = auto based on max_radius


class SpiralCoveragePlanner:
    """Spiral coverage path planner."""

    def plan(
        self,
        center: Tuple[float, float],
        max_radius: float,
        spacing: Optional[float] = None,
        direction: Optional[str] = None,
        config: Optional[SpiralConfig] = None,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
    ) -> List[Tuple[float, float]]:
        """Generate a spiral coverage path.

        Args:
            center: Center point of the spiral (x, y).
            max_radius: Maximum radius of the spiral.
            spacing: Distance between spiral arms (overrides config).
            direction: "outward" or "inward" (overrides config).
            config: SpiralConfig object (alternative to spacing/direction).
            obstacles: List of (x, y, r) circular obstacles to avoid.

        Returns:
            List of (x, y) waypoints forming the spiral path.
        """
        # Resolve parameters
        if config is not None:
            sp = spacing if spacing is not None else config.spacing
            dirn = direction if direction is not None else config.direction
        else:
            sp = spacing if spacing is not None else 2.0
            dirn = direction if direction is not None else "outward"

        if max_radius <= 0:
            return [center]

        # Generate raw spiral points
        path = self._generate_spiral(center, max_radius, sp)

        # Reverse for inward spiral
        if dirn == "inward":
            path.reverse()

        # Filter out points inside obstacles
        if obstacles:
            path = self._filter_obstacles(path, obstacles)

        return path

    def _generate_spiral(
        self,
        center: Tuple[float, float],
        max_radius: float,
        spacing: float,
    ) -> List[Tuple[float, float]]:
        """Generate an Archimedean spiral path.

        Uses the Archimedean spiral: r = a * theta, where a controls spacing.
        The spacing between successive turns is 2*pi*a, so a = spacing / (2*pi).
        """
        path = []
        cx, cy = center
        a = spacing / (2.0 * math.pi)

        # Number of steps: enough to reach max_radius
        # r = a * theta => theta_max = max_radius / a
        # Use angular step that gives smooth curves
        theta_max = max_radius / a
        num_steps = max(int(theta_max * 20), 50)
        dtheta = theta_max / num_steps

        for i in range(num_steps + 1):
            theta = i * dtheta
            r = a * theta
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            path.append((x, y))

        return path

    def _filter_obstacles(
        self,
        path: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float]]:
        """Remove path points that fall inside any obstacle."""
        filtered = []
        for x, y in path:
            inside = False
            for ox, oy, r in obstacles:
                dist = math.hypot(x - ox, y - oy)
                if dist <= r:
                    inside = True
                    break
            if not inside:
                filtered.append((x, y))
        return filtered

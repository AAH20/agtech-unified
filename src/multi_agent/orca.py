"""ORCA (Optimal Reciprocal Collision Avoidance) for multi-agent systems.

ORCA is a local collision avoidance algorithm where each agent independently
computes a velocity that avoids collisions with neighbors, assuming neighbors
also take reciprocal avoidance actions. The result is provably collision-free
under the reciprocal assumption.

Implementation follows van den Berg et al., "Reciprocal n-Body Collision
Avoidance" (2011), using half-plane constraints and linear programming.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.multi_agent.collision_avoidance import Position, Velocity

logger = logging.getLogger(__name__)


@dataclass
class ORCAAgent:
    """An agent in the ORCA simulation."""

    agent_id: str
    position: Position
    velocity: Velocity
    preferred_velocity: Velocity
    radius: float = 0.5


class ORCAWorld:
    """World managing ORCA collision avoidance for multiple agents."""

    def __init__(
        self,
        safety_radius: float = 2.0,
        max_speed: float = 5.0,
        time_horizon: float = 5.0,
    ):
        """Initialize ORCA world.

        Args:
            safety_radius: Minimum safe distance between agents.
            max_speed: Maximum speed for any agent.
            time_horizon: Time horizon for collision prediction.
        """
        self.safety_radius = safety_radius
        self.max_speed = max_speed
        self.time_horizon = time_horizon
        self._agents: Dict[str, ORCAAgent] = {}

    def add_agent(self, agent: ORCAAgent) -> None:
        """Add an agent to the world."""
        self._agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the world."""
        self._agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[ORCAAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_agents(self) -> List[ORCAAgent]:
        """Get all agents in the world."""
        return list(self._agents.values())

    def compute_orca_velocity(self, agent_id: str) -> Optional[Velocity]:
        """Compute ORCA velocity for an agent.

        Uses velocity obstacles to find a collision-free velocity that
        minimizes deviation from the preferred velocity.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            return None

        constraints: List[Tuple[float, float, float]] = []
        for other_id, other in self._agents.items():
            if other_id == agent_id:
                continue
            constraint = self._compute_orca_half_plane(agent, other)
            if constraint is not None:
                constraints.append(constraint)

        if not constraints:
            return self._clamp_velocity(agent.preferred_velocity)

        return self._find_optimal_velocity(agent, constraints)

    def _compute_orca_half_plane(
        self,
        agent: ORCAAgent,
        other: ORCAAgent,
    ) -> Optional[Tuple[float, float, float]]:
        """Compute ORCA half-plane constraint for an agent pair.

        Returns (a, b, c) representing ax + by <= c, or None if no collision.
        """
        # Relative position: from agent to other
        dx = other.position.x - agent.position.x
        dy = other.position.y - agent.position.y
        dist_sq = dx * dx + dy * dy
        dist = math.sqrt(dist_sq)

        combined_radius = agent.radius + other.radius + self.safety_radius

        # Already colliding: push apart
        if dist < combined_radius:
            if dist < 0.001:
                return (1.0, 0.0, -1.0)
            nx = dx / dist
            ny = dy / dist
            push = (combined_radius - dist) / self.time_horizon
            return (nx, ny, -push)

        # Relative velocity: other relative to agent
        rvx = other.velocity.vx - agent.velocity.vx
        rvy = other.velocity.vy - agent.velocity.vy

        # Project relative velocity onto line of sight
        dot = rvx * dx + rvy * dy
        if dot >= 0:
            return None  # Moving apart

        # Time to closest approach
        rv_sq = rvx * rvx + rvy * rvy
        if rv_sq < 1e-10:
            return None
        t_cpa = -(dx * rvx + dy * rvy) / rv_sq
        if t_cpa < 0 or t_cpa > self.time_horizon:
            return None

        # Distance at closest approach
        cpa_x = dx + rvx * t_cpa
        cpa_y = dy + rvy * t_cpa
        cpa_dist_sq = cpa_x * cpa_x + cpa_y * cpa_y

        if cpa_dist_sq >= combined_radius * combined_radius:
            return None

        # Collision possible — compute ORCA half-plane
        # Unit normal from agent to other
        if dist < 0.001:
            nx, ny = 1.0, 0.0
        else:
            nx = dx / dist
            ny = dy / dist

        # ORCA: each agent takes half the responsibility.
        # The constraint is: (v_agent - v_other) · n >= (combined_radius - dist) / time_horizon
        # where n points from agent to other. This ensures the agent moves AWAY from the other.
        # In ax + by <= c form:
        # -nx * vx - ny * vy <= -(v_other · n + (combined_radius - dist) / time_horizon)

        other_normal = other.velocity.vx * nx + other.velocity.vy * ny
        min_separation_speed = (combined_radius - dist) / self.time_horizon

        c = -(other_normal + min_separation_speed)

        return (-nx, -ny, c)

    def _find_optimal_velocity(
        self,
        agent: ORCAAgent,
        constraints: List[Tuple[float, float, float]],
    ) -> Velocity:
        """Find optimal velocity satisfying all constraints.

        Uses iterative projection onto constraint boundaries.
        """
        vx = agent.preferred_velocity.vx
        vy = agent.preferred_velocity.vy

        max_iterations = 10
        for _ in range(max_iterations):
            violated = False
            for a, b, c in constraints:
                if a * vx + b * vy > c + 1e-6:
                    denom = a * a + b * b
                    if denom < 1e-10:
                        continue
                    t = (a * vx + b * vy - c) / denom
                    vx -= a * t
                    vy -= b * t
                    violated = True
            if not violated:
                break

        return self._clamp_velocity(Velocity(vx=vx, vy=vy))

    def _clamp_velocity(self, velocity: Velocity) -> Velocity:
        """Clamp velocity to max_speed."""
        speed = math.sqrt(velocity.vx**2 + velocity.vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            return Velocity(vx=velocity.vx * scale, vy=velocity.vy * scale)
        return velocity

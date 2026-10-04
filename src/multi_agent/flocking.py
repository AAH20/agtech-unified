"""Reynolds flocking behaviors for multi-agent agricultural swarms.

Implements the three classic boids rules:
  - Separation: steer to avoid crowding nearby agents.
  - Alignment: steer toward the average heading of nearby agents.
  - Cohesion: steer toward the average position of nearby agents.

Reference: Reynolds, C. W. (1987). "Flocks, Herds, and Schools: A Distributed
Behavioral Model." SIGGRAPH '87.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.multi_agent.collision_avoidance import Position, Velocity

logger = logging.getLogger(__name__)


@dataclass
class FlockingBehavior:
    """Reynolds flocking behavior for swarm agents.

    Each behavior (separation, alignment, cohesion) produces a steering
    force that is weighted and summed to produce the final steering velocity.
    """

    perception_radius: float = 5.0
    separation_radius: float = 2.0
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    max_force: float = 10.0

    def compute_steering(
        self,
        agent_id: str,
        positions: Dict[str, Position],
        velocities: Dict[str, Velocity],
    ) -> Velocity:
        """Compute the flocking steering force for a single agent.

        Args:
            agent_id: The agent to compute steering for.
            positions: All agent positions.
            velocities: All agent velocities.

        Returns:
            The combined steering velocity (separation + alignment + cohesion).
        """
        if agent_id not in positions:
            return Velocity(0.0, 0.0)

        neighbors = self._get_neighbors(agent_id, positions)
        if not neighbors:
            return Velocity(0.0, 0.0)

        sep = self._separation(agent_id, positions, neighbors)
        ali = self._alignment(agent_id, velocities, neighbors)
        coh = self._cohesion(agent_id, positions, neighbors)

        total_vx = (
            self.separation_weight * sep[0]
            + self.alignment_weight * ali[0]
            + self.cohesion_weight * coh[0]
        )
        total_vy = (
            self.separation_weight * sep[1]
            + self.alignment_weight * ali[1]
            + self.cohesion_weight * coh[1]
        )

        # Clamp to max_force
        force_mag = math.sqrt(total_vx**2 + total_vy**2)
        if force_mag > self.max_force:
            scale = self.max_force / force_mag
            total_vx *= scale
            total_vy *= scale

        return Velocity(vx=total_vx, vy=total_vy)

    def compute_all_steering(
        self,
        positions: Dict[str, Position],
        velocities: Dict[str, Velocity],
    ) -> Dict[str, Velocity]:
        """Compute flocking steering for all agents.

        Args:
            positions: All agent positions.
            velocities: All agent velocities.

        Returns:
            Dictionary mapping agent_id to steering Velocity.
        """
        return {
            agent_id: self.compute_steering(agent_id, positions, velocities)
            for agent_id in positions
        }

    def _get_neighbors(
        self,
        agent_id: str,
        positions: Dict[str, Position],
    ) -> List[str]:
        """Get all neighbor IDs within perception_radius (excluding self)."""
        agent_pos = positions[agent_id]
        neighbors: List[str] = []
        for other_id, other_pos in positions.items():
            if other_id == agent_id:
                continue
            if agent_pos.distance_to(other_pos) <= self.perception_radius:
                neighbors.append(other_id)
        return neighbors

    def _separation(
        self,
        agent_id: str,
        positions: Dict[str, Position],
        neighbors: List[str],
    ) -> Tuple[float, float]:
        """Compute separation steering: steer away from close neighbors.

        Only neighbors within separation_radius contribute.
        """
        agent_pos = positions[agent_id]
        steer_x = 0.0
        steer_y = 0.0
        count = 0

        for other_id in neighbors:
            other_pos = positions[other_id]
            dist = agent_pos.distance_to(other_pos)
            if dist < self.separation_radius and dist > 1e-6:
                # Vector from other to agent (away from neighbor)
                dx = agent_pos.x - other_pos.x
                dy = agent_pos.y - other_pos.y
                # Weight by inverse distance (closer = stronger)
                weight = 1.0 / dist
                steer_x += (dx / dist) * weight
                steer_y += (dy / dist) * weight
                count += 1

        if count > 0:
            steer_x /= count
            steer_y /= count

        return (steer_x, steer_y)

    def _alignment(
        self,
        agent_id: str,
        velocities: Dict[str, Velocity],
        neighbors: List[str],
    ) -> Tuple[float, float]:
        """Compute alignment steering: match average neighbor velocity."""
        avg_vx = 0.0
        avg_vy = 0.0
        count = 0

        for other_id in neighbors:
            vel = velocities.get(other_id)
            if vel is not None:
                avg_vx += vel.vx
                avg_vy += vel.vy
                count += 1

        if count > 0:
            avg_vx /= count
            avg_vy /= count

        return (avg_vx, avg_vy)

    def _cohesion(
        self,
        agent_id: str,
        positions: Dict[str, Position],
        neighbors: List[str],
    ) -> Tuple[float, float]:
        """Compute cohesion steering: move toward centroid of neighbors."""
        center_x = 0.0
        center_y = 0.0
        count = 0

        for other_id in neighbors:
            other_pos = positions[other_id]
            center_x += other_pos.x
            center_y += other_pos.y
            count += 1

        if count > 0:
            center_x /= count
            center_y /= count
            agent_pos = positions[agent_id]
            # Steer toward center
            return (center_x - agent_pos.x, center_y - agent_pos.y)

        return (0.0, 0.0)

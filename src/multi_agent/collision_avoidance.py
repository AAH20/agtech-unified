"""Multi-agent collision avoidance for agricultural swarm robots."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """2D position of an agent."""
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        """Compute Euclidean distance to another position.

        Args:
            other: The other position.

        Returns:
            Euclidean distance.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Velocity:
    """2D velocity vector."""
    vx: float
    vy: float


class CollisionAvoidance:
    """Simple collision avoidance for a swarm of agricultural robots.

    Uses a safety radius to detect potential collisions and computes
    avoidance velocities that push agents away from each other.
    """

    def __init__(self, safety_radius: float = 2.0, max_avoidance_speed: float = 1.0):
        """Initialize collision avoidance.

        Args:
            safety_radius: Minimum safe distance between agents.
            max_avoidance_speed: Maximum speed of avoidance velocity.
        """
        self.safety_radius = safety_radius
        self.max_avoidance_speed = max_avoidance_speed

    def is_collision(self, pos1: Position, pos2: Position) -> bool:
        """Check if two positions are within collision distance.

        Args:
            pos1: First agent position.
            pos2: Second agent position.

        Returns:
            True if the distance between positions is less than safety_radius.
        """
        return pos1.distance_to(pos2) < self.safety_radius

    def compute_avoidance_velocity(
        self,
        agent_pos: Position,
        obstacle_pos: Position,
        current_vel: Velocity,
    ) -> Velocity:
        """Compute avoidance velocity to steer away from an obstacle.

        Args:
            agent_pos: Current position of the agent.
            obstacle_pos: Position of the obstacle to avoid.
            current_vel: Current velocity of the agent.

        Returns:
            Avoidance velocity vector (zero if no avoidance needed).
        """
        dx = agent_pos.x - obstacle_pos.x
        dy = agent_pos.y - obstacle_pos.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist >= self.safety_radius or dist == 0:
            return Velocity(vx=0.0, vy=0.0)

        # Normalize and scale by proximity (closer = stronger avoidance)
        proximity = (self.safety_radius - dist) / self.safety_radius
        scale = proximity * self.max_avoidance_speed

        return Velocity(
            vx=(dx / dist) * scale,
            vy=(dy / dist) * scale,
        )

    def detect_collisions(self, positions: Dict[str, Position]) -> List[Tuple[str, str]]:
        """Detect all colliding pairs in a swarm.

        Args:
            positions: Dictionary mapping agent_id to Position.

        Returns:
            List of (agent_id_1, agent_id_2) tuples for colliding pairs.
        """
        collisions: List[Tuple[str, str]] = []
        agent_ids = list(positions.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                if self.is_collision(positions[id1], positions[id2]):
                    collisions.append((id1, id2))

        return collisions

    def compute_all_avoidance_velocities(
        self,
        positions: Dict[str, Position],
        velocities: Dict[str, Velocity],
    ) -> Dict[str, Velocity]:
        """Compute avoidance velocities for all agents in a swarm.

        For each agent, sums avoidance velocities from all nearby agents.

        Args:
            positions: Dictionary mapping agent_id to Position.
            velocities: Dictionary mapping agent_id to Velocity.

        Returns:
            Dictionary mapping agent_id to avoidance Velocity.
        """
        result: Dict[str, Velocity] = {}

        for agent_id, agent_pos in positions.items():
            total_vx = 0.0
            total_vy = 0.0

            for other_id, other_pos in positions.items():
                if other_id == agent_id:
                    continue
                avoidance = self.compute_avoidance_velocity(
                    agent_pos, other_pos, velocities.get(agent_id, Velocity(0.0, 0.0))
                )
                total_vx += avoidance.vx
                total_vy += avoidance.vy

            result[agent_id] = Velocity(vx=total_vx, vy=total_vy)

        return result

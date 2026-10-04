"""Multi-agent collision avoidance for agricultural swarm robots."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    """Collision avoidance for a swarm of agricultural robots.

    Uses a safety radius to detect potential collisions and computes
    avoidance velocities that push agents away from each other.
    Supports velocity obstacles, deadlock detection, and collision
    prediction with configurable horizon.
    """

    def __init__(
        self,
        safety_radius: float = 2.0,
        max_avoidance_speed: float = 1.0,
        prediction_horizon: float = 10.0,
    ):
        """Initialize collision avoidance.

        Args:
            safety_radius: Minimum safe distance between agents.
            max_avoidance_speed: Maximum speed of avoidance velocity.
            prediction_horizon: Seconds ahead to predict collisions.
        """
        self.safety_radius = safety_radius
        self.max_avoidance_speed = max_avoidance_speed
        self.prediction_horizon = prediction_horizon

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

    def detect_collisions(
        self,
        positions: Dict[str, Position],
        radii: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, str]]:
        """Detect all colliding pairs in a swarm.

        Args:
            positions: Dictionary mapping agent_id to Position.
            radii: Optional dictionary mapping agent_id to safety radius.
                If not provided, uses default safety_radius for all.

        Returns:
            List of (agent_id_1, agent_id_2) tuples for colliding pairs.
        """
        collisions: List[Tuple[str, str]] = []
        agent_ids = list(positions.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                r1 = radii.get(id1, self.safety_radius) if radii else self.safety_radius
                r2 = radii.get(id2, self.safety_radius) if radii else self.safety_radius
                effective_radius = r1 + r2
                if positions[id1].distance_to(positions[id2]) < effective_radius:
                    collisions.append((id1, id2))

        return collisions

    def compute_all_avoidance_velocities(
        self,
        positions: Dict[str, Position],
        velocities: Dict[str, Velocity],
    ) -> Dict[str, Velocity]:
        """Compute avoidance velocities for all agents in a swarm.

        For each agent, sums avoidance velocities from all nearby agents.
        The total avoidance velocity is clamped to max_avoidance_speed.

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

            # Clamp to max_avoidance_speed
            speed = math.sqrt(total_vx**2 + total_vy**2)
            if speed > self.max_avoidance_speed:
                scale = self.max_avoidance_speed / speed
                total_vx *= scale
                total_vy *= scale

            result[agent_id] = Velocity(vx=total_vx, vy=total_vy)

        return result

    def compute_velocity_obstacle(
        self,
        agent_pos: Position,
        other_pos: Position,
        agent_vel: Velocity,
        other_vel: Velocity,
    ) -> Optional[Dict[str, float]]:
        """Compute velocity obstacle for a pair of agents.

        The velocity obstacle is the set of velocities that would lead
        to a collision with the other agent.

        Args:
            agent_pos: Position of the agent.
            other_pos: Position of the other agent.
            agent_vel: Velocity of the agent.
            other_vel: Velocity of the other agent.

        Returns:
            Dictionary with velocity obstacle parameters, or None if
            no collision is possible.
        """
        dx = other_pos.x - agent_pos.x
        dy = other_pos.y - agent_pos.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist >= self.safety_radius * 4:
            return None

        # Relative velocity
        rvx = agent_vel.vx - other_vel.vx
        rvy = agent_vel.vy - other_vel.vy

        return {
            "distance": dist,
            "relative_vx": rvx,
            "relative_vy": rvy,
            "safety_radius": self.safety_radius,
        }

    def time_to_collision(
        self,
        agent_pos: Position,
        other_pos: Position,
        agent_vel: Velocity,
        other_vel: Velocity,
    ) -> Optional[float]:
        """Compute time to collision between two agents.

        Args:
            agent_pos: Position of the first agent.
            other_pos: Position of the second agent.
            agent_vel: Velocity of the first agent.
            other_vel: Velocity of the second agent.

        Returns:
            Time to collision in seconds, or None if no collision.
        """
        dx = agent_pos.x - other_pos.x
        dy = agent_pos.y - other_pos.y
        dvx = agent_vel.vx - other_vel.vx
        dvy = agent_vel.vy - other_vel.vy

        # Relative velocity squared
        dv_sq = dvx * dvx + dvy * dvy
        if dv_sq == 0:
            return None  # Moving at same velocity, no collision

        # Time of closest approach
        t_cpa = -(dx * dvx + dy * dvy) / dv_sq
        if t_cpa < 0:
            return None  # Moving apart

        # Distance at closest approach
        dx_cpa = dx + dvx * t_cpa
        dy_cpa = dy + dvy * t_cpa
        dist_cpa = math.sqrt(dx_cpa * dx_cpa + dy_cpa * dy_cpa)

        if dist_cpa < self.safety_radius:
            return t_cpa
        return None

    def detect_deadlock(
        self,
        positions: Dict[str, Position],
        velocities: Dict[str, Velocity],
    ) -> bool:
        """Detect deadlock in a swarm.

        Deadlock occurs when agents are close and moving toward each
        other but cannot escape (e.g., head-on in narrow passage).

        Args:
            positions: Dictionary mapping agent_id to Position.
            velocities: Dictionary mapping agent_id to Velocity.

        Returns:
            True if deadlock is detected.
        """
        agent_ids = list(positions.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                pos1, pos2 = positions[id1], positions[id2]
                vel1 = velocities.get(id1, Velocity(0.0, 0.0))
                vel2 = velocities.get(id2, Velocity(0.0, 0.0))

                # Check if agents are close
                if pos1.distance_to(pos2) >= self.safety_radius * 2:
                    continue

                # Check if moving toward each other
                dx = pos2.x - pos1.x
                dy = pos2.y - pos1.y
                dvx = vel1.vx - vel2.vx
                dvy = vel1.vy - vel2.vy

                # Dot product: if positive, moving toward each other
                if dx * dvx + dy * dvy > 0:
                    return True

        return False

    def predict_collisions(
        self,
        positions: Dict[str, Position],
        velocities: Dict[str, Velocity],
    ) -> List[Tuple[str, str]]:
        """Predict collisions within the prediction horizon.

        Args:
            positions: Dictionary mapping agent_id to Position.
            velocities: Dictionary mapping agent_id to Velocity.

        Returns:
            List of (agent_id_1, agent_id_2) tuples for predicted collisions.
        """
        collisions: List[Tuple[str, str]] = []
        agent_ids = list(positions.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                ttc = self.time_to_collision(
                    positions[id1],
                    positions[id2],
                    velocities.get(id1, Velocity(0.0, 0.0)),
                    velocities.get(id2, Velocity(0.0, 0.0)),
                )
                if ttc is not None and ttc <= self.prediction_horizon:
                    collisions.append((id1, id2))

        return collisions

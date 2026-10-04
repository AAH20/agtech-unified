"""Dynamic Window Approach (DWA) for local path planning.

DWA samples velocities from a feasible window (constrained by acceleration
limits), simulates short trajectories for each, and scores them based on
goal proximity, obstacle clearance, and speed. The highest-scoring velocity
is selected.

Reference: Fox, D., Burgard, W., & Thrun, S. (1997). "The Dynamic Window
Approach to Collision Avoidance." IEEE Robotics & Automation Magazine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.multi_agent.collision_avoidance import Position, Velocity

logger = logging.getLogger(__name__)


@dataclass
class DWAConfig:
    """Configuration for the Dynamic Window Approach."""

    max_speed: float = 2.0
    max_acceleration: float = 1.0
    time_step: float = 0.1
    prediction_horizon: float = 3.0
    num_samples: int = 20
    goal_weight: float = 1.0
    clearance_weight: float = 1.0
    speed_weight: float = 0.5
    obstacle_radius: float = 1.0


@dataclass
class Trajectory:
    """A simulated trajectory with its associated velocity."""

    positions: List[Position] = field(default_factory=list)
    velocity: Velocity = field(default_factory=lambda: Velocity(0.0, 0.0))


class DynamicWindowApproach:
    """Dynamic Window Approach for local collision-free path planning."""

    def __init__(self, config: Optional[DWAConfig] = None):
        """Initialize DWA with optional custom configuration.

        Args:
            config: DWA parameters. Uses defaults if not provided.
        """
        self.config = config or DWAConfig()

    def compute_best_velocity(
        self,
        current_pos: Position,
        current_vel: Velocity,
        goal: Position,
        obstacles: List[Position],
    ) -> Velocity:
        """Compute the best velocity using DWA.

        Samples velocities from the dynamic window, simulates trajectories,
        scores each, and returns the velocity with the highest score.

        Args:
            current_pos: Current position of the agent.
            current_vel: Current velocity of the agent.
            goal: Target position to reach.
            obstacles: List of obstacle positions to avoid.

        Returns:
            The best velocity to execute.
        """
        # Compute dynamic window: reachable velocities given acceleration limits
        min_vx = current_vel.vx - self.config.max_acceleration * self.config.time_step
        max_vx = current_vel.vx + self.config.max_acceleration * self.config.time_step
        min_vy = current_vel.vy - self.config.max_acceleration * self.config.time_step
        max_vy = current_vel.vy + self.config.max_acceleration * self.config.time_step

        # Clamp to max_speed
        min_vx = max(min_vx, -self.config.max_speed)
        max_vx = min(max_vx, self.config.max_speed)
        min_vy = max(min_vy, -self.config.max_speed)
        max_vy = min(max_vy, self.config.max_speed)

        best_score = -float("inf")
        best_vel = Velocity(0.0, 0.0)

        # Sample velocities from the dynamic window
        for i in range(self.config.num_samples):
            for j in range(self.config.num_samples):
                vx = min_vx + (max_vx - min_vx) * i / max(self.config.num_samples - 1, 1)
                vy = min_vy + (max_vy - min_vy) * j / max(self.config.num_samples - 1, 1)

                vel = Velocity(vx=vx, vy=vy)
                trajectory = self.simulate_trajectory(current_pos, vel)
                score = self.score_trajectory(trajectory, goal, obstacles)

                if score > best_score:
                    best_score = score
                    best_vel = vel

        return best_vel

    def simulate_trajectory(
        self,
        start: Position,
        velocity: Velocity,
    ) -> Trajectory:
        """Simulate a trajectory given a constant velocity.

        Args:
            start: Starting position.
            velocity: Constant velocity to simulate.

        Returns:
            Trajectory with positions at each time step.
        """
        positions: List[Position] = []
        x, y = start.x, start.y
        num_steps = int(self.config.prediction_horizon / self.config.time_step)

        for _ in range(num_steps):
            x += velocity.vx * self.config.time_step
            y += velocity.vy * self.config.time_step
            positions.append(Position(x=x, y=y))

        return Trajectory(positions=positions, velocity=velocity)

    def score_trajectory(
        self,
        trajectory: Trajectory,
        goal: Position,
        obstacles: List[Position],
    ) -> float:
        """Score a trajectory based on goal, clearance, and speed.

        Higher score = better trajectory.

        Args:
            trajectory: The trajectory to score.
            goal: Target position.
            obstacles: Obstacle positions.

        Returns:
            Score value (higher is better).
        """
        if not trajectory.positions:
            return -float("inf")

        final_pos = trajectory.positions[-1]

        # Goal score: closer to goal is better
        goal_dist = final_pos.distance_to(goal)
        goal_score = -goal_dist

        # Clearance score: minimum distance to any obstacle
        min_clearance = float("inf")
        for pos in trajectory.positions:
            for obs in obstacles:
                dist = pos.distance_to(obs)
                if dist < min_clearance:
                    min_clearance = dist
        if min_clearance == float("inf"):
            min_clearance = 10.0  # No obstacles = large clearance
        clearance_score = min_clearance

        # Speed score: higher speed is better
        speed = math.sqrt(trajectory.velocity.vx**2 + trajectory.velocity.vy**2)
        speed_score = speed

        total = (
            self.config.goal_weight * goal_score
            + self.config.clearance_weight * clearance_score
            + self.config.speed_weight * speed_score
        )
        return total

"""Tests for Dynamic Window Approach (DWA) local path planning."""

import math

import pytest

from src.multi_agent.collision_avoidance import Position, Velocity
from src.multi_agent.dwa import DWAConfig, DynamicWindowApproach, Trajectory


def test_dwa_config_defaults():
    """DWAConfig has sensible defaults."""
    config = DWAConfig()
    assert config.max_speed > 0
    assert config.max_acceleration > 0
    assert config.time_step > 0
    assert config.prediction_horizon > 0
    assert config.num_samples > 0
    assert config.goal_weight > 0
    assert config.clearance_weight > 0
    assert config.speed_weight > 0


def test_compute_best_velocity_returns_velocity():
    """compute_best_velocity returns a Velocity within speed limits."""
    dwa = DynamicWindowApproach()
    current_pos = Position(0.0, 0.0)
    current_vel = Velocity(0.0, 0.0)
    goal = Position(10.0, 0.0)
    obstacles = []
    best = dwa.compute_best_velocity(current_pos, current_vel, goal, obstacles)
    assert isinstance(best, Velocity)
    speed = math.sqrt(best.vx**2 + best.vy**2)
    assert speed <= dwa.config.max_speed + 1e-6


def test_dwa_prefers_velocity_toward_goal():
    """With no obstacles, DWA should pick a velocity pointing toward the goal."""
    dwa = DynamicWindowApproach()
    current_pos = Position(0.0, 0.0)
    current_vel = Velocity(0.0, 0.0)
    goal = Position(10.0, 0.0)
    obstacles = []
    best = dwa.compute_best_velocity(current_pos, current_vel, goal, obstacles)
    # Goal is to the right, so best velocity should have positive x component
    assert best.vx > 0.0


def test_dwa_avoids_obstacle():
    """DWA should avoid a velocity that leads directly into an obstacle."""
    dwa = DynamicWindowApproach()
    current_pos = Position(0.0, 0.0)
    current_vel = Velocity(0.0, 0.0)
    goal = Position(10.0, 0.0)
    # Obstacle directly on the path to the goal
    obstacles = [Position(5.0, 0.0)]
    best = dwa.compute_best_velocity(current_pos, current_vel, goal, obstacles)
    # The best velocity should not be pointing straight at the obstacle
    # It should have some y component to steer around
    assert abs(best.vy) > 1e-6 or best.vx < dwa.config.max_speed * 0.5


def test_dwa_respects_acceleration_limits():
    """The chosen velocity must be reachable within acceleration limits."""
    dwa = DynamicWindowApproach()
    current_pos = Position(0.0, 0.0)
    current_vel = Velocity(0.0, 0.0)
    goal = Position(10.0, 0.0)
    obstacles = []
    best = dwa.compute_best_velocity(current_pos, current_vel, goal, obstacles)
    # With zero initial velocity and limited acceleration, each velocity
    # component should be bounded by max_acceleration * time_step
    max_reachable = dwa.config.max_acceleration * dwa.config.time_step
    assert abs(best.vx) <= max_reachable + 1e-6
    assert abs(best.vy) <= max_reachable + 1e-6


def test_simulate_trajectory_returns_positions():
    """simulate_trajectory returns a list of positions."""
    dwa = DynamicWindowApproach()
    start = Position(0.0, 0.0)
    vel = Velocity(1.0, 0.0)
    trajectory = dwa.simulate_trajectory(start, vel)
    assert isinstance(trajectory, Trajectory)
    assert len(trajectory.positions) > 0
    # First position is after one time step: start + vel * dt
    assert trajectory.positions[0].x == pytest.approx(vel.vx * dwa.config.time_step)
    assert trajectory.positions[0].y == pytest.approx(vel.vy * dwa.config.time_step)


def test_simulate_trajectory_respects_horizon():
    """Trajectory length matches prediction_horizon / time_step."""
    dwa = DynamicWindowApproach()
    start = Position(0.0, 0.0)
    vel = Velocity(1.0, 0.0)
    trajectory = dwa.simulate_trajectory(start, vel)
    expected_steps = int(dwa.config.prediction_horizon / dwa.config.time_step)
    assert len(trajectory.positions) == expected_steps


def test_score_trajectory_prefers_goal_proximity():
    """A trajectory ending closer to the goal scores higher."""
    dwa = DynamicWindowApproach()
    goal = Position(10.0, 0.0)
    obstacles = []
    # Trajectory ending near goal
    good_positions = [Position(9.0, 0.0), Position(10.0, 0.0)]
    good_traj = Trajectory(positions=good_positions, velocity=Velocity(1.0, 0.0))
    # Trajectory ending far from goal
    bad_positions = [Position(0.0, 5.0), Position(0.0, 10.0)]
    bad_traj = Trajectory(positions=bad_positions, velocity=Velocity(0.0, 1.0))
    good_score = dwa.score_trajectory(good_traj, goal, obstacles)
    bad_score = dwa.score_trajectory(bad_traj, goal, obstacles)
    assert good_score > bad_score


def test_score_trajectory_prefers_clearance():
    """A trajectory with more obstacle clearance scores higher."""
    dwa = DynamicWindowApproach()
    goal = Position(10.0, 0.0)
    obstacles = [Position(5.0, 0.0)]
    # Trajectory passing through obstacle
    close_positions = [Position(4.0, 0.0), Position(5.0, 0.0)]
    close_traj = Trajectory(positions=close_positions, velocity=Velocity(1.0, 0.0))
    # Trajectory staying clear
    clear_positions = [Position(4.0, 3.0), Position(5.0, 3.0)]
    clear_traj = Trajectory(positions=clear_positions, velocity=Velocity(1.0, 0.0))
    close_score = dwa.score_trajectory(close_traj, goal, obstacles)
    clear_score = dwa.score_trajectory(clear_traj, goal, obstacles)
    assert clear_score > close_score


def test_score_trajectory_prefers_higher_speed():
    """A faster trajectory scores higher when other factors are equal."""
    dwa = DynamicWindowApproach()
    goal = Position(10.0, 0.0)
    obstacles = []
    positions = [Position(5.0, 0.0)]
    slow_traj = Trajectory(positions=positions, velocity=Velocity(0.5, 0.0))
    fast_traj = Trajectory(positions=positions, velocity=Velocity(2.0, 0.0))
    slow_score = dwa.score_trajectory(slow_traj, goal, obstacles)
    fast_score = dwa.score_trajectory(fast_traj, goal, obstacles)
    assert fast_score > slow_score


def test_dwa_with_obstacle_directly_ahead_still_moves():
    """Even with an obstacle ahead, DWA returns a non-zero velocity."""
    dwa = DynamicWindowApproach()
    current_pos = Position(0.0, 0.0)
    current_vel = Velocity(0.0, 0.0)
    goal = Position(10.0, 0.0)
    obstacles = [Position(3.0, 0.0)]
    best = dwa.compute_best_velocity(current_pos, current_vel, goal, obstacles)
    speed = math.sqrt(best.vx**2 + best.vy**2)
    assert speed > 0.0


def test_dwa_goal_at_current_position():
    """When already at the goal, DWA should return near-zero velocity."""
    dwa = DynamicWindowApproach()
    current_pos = Position(5.0, 5.0)
    current_vel = Velocity(0.0, 0.0)
    goal = Position(5.0, 5.0)
    obstacles = []
    best = dwa.compute_best_velocity(current_pos, current_vel, goal, obstacles)
    speed = math.sqrt(best.vx**2 + best.vy**2)
    assert speed < 0.5


def test_dwa_config_custom_values():
    """DWAConfig accepts custom values."""
    config = DWAConfig(
        max_speed=3.0,
        max_acceleration=1.5,
        time_step=0.2,
        prediction_horizon=5.0,
        num_samples=20,
        goal_weight=2.0,
        clearance_weight=1.5,
        speed_weight=0.5,
    )
    assert config.max_speed == 3.0
    assert config.max_acceleration == 1.5
    assert config.time_step == 0.2
    assert config.prediction_horizon == 5.0
    assert config.num_samples == 20
    assert config.goal_weight == 2.0
    assert config.clearance_weight == 1.5
    assert config.speed_weight == 0.5

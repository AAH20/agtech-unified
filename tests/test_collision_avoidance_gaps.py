"""Test multi-agent collision avoidance: velocity obstacles, static obstacles, deadlock detection."""

import pytest

from src.multi_agent.collision_avoidance import (
    CollisionAvoidance,
    Position,
    Velocity,
)

# ── MA-CA-002: Velocity obstacles ─────────────────────────────────────


def test_velocity_obstacle_computation():
    """Velocity obstacle is computed for a pair of agents."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    other_pos = Position(x=5.0, y=0.0)
    agent_vel = Velocity(vx=1.0, vy=0.0)
    other_vel = Velocity(vx=-1.0, vy=0.0)

    vo = ca.compute_velocity_obstacle(agent_pos, other_pos, agent_vel, other_vel)
    assert vo is not None


def test_time_to_collision():
    """Time to collision is computed for approaching agents."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    other_pos = Position(x=10.0, y=0.0)
    agent_vel = Velocity(vx=1.0, vy=0.0)
    other_vel = Velocity(vx=-1.0, vy=0.0)

    ttc = ca.time_to_collision(agent_pos, other_pos, agent_vel, other_vel)
    assert ttc is not None
    assert ttc > 0


def test_no_collision_when_moving_apart():
    """No collision when agents are moving apart."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    other_pos = Position(x=10.0, y=0.0)
    agent_vel = Velocity(vx=-1.0, vy=0.0)
    other_vel = Velocity(vx=1.0, vy=0.0)

    ttc = ca.time_to_collision(agent_pos, other_pos, agent_vel, other_vel)
    assert ttc is None or ttc == float("inf")


# ── MA-CA-004: Static obstacle avoidance ───────────────────────────────


def test_static_obstacle_avoidance():
    """CollisionAvoidance can avoid static obstacles."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    obstacle_pos = Position(x=1.0, y=0.0)
    current_vel = Velocity(vx=1.0, vy=0.0)

    avoidance = ca.compute_avoidance_velocity(agent_pos, obstacle_pos, current_vel)
    # Should push agent away from obstacle (negative x)
    assert avoidance.vx < 0


def test_static_obstacle_no_collision_when_far():
    """No avoidance when far from static obstacle."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    obstacle_pos = Position(x=100.0, y=0.0)
    current_vel = Velocity(vx=1.0, vy=0.0)

    avoidance = ca.compute_avoidance_velocity(agent_pos, obstacle_pos, current_vel)
    assert avoidance.vx == pytest.approx(0.0)
    assert avoidance.vy == pytest.approx(0.0)


# ── MA-CA-005: Deadlock detection ─────────────────────────────────────


def test_deadlock_detection():
    """Deadlock is detected when agents block each other."""
    ca = CollisionAvoidance(safety_radius=2.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=1.0, y=0.0),
    }
    velocities = {
        "agent_1": Velocity(vx=1.0, vy=0.0),
        "agent_2": Velocity(vx=-1.0, vy=0.0),
    }

    is_deadlocked = ca.detect_deadlock(positions, velocities)
    assert is_deadlocked


def test_no_deadlock_when_agents_can_move():
    """No deadlock when agents can move around each other."""
    ca = CollisionAvoidance(safety_radius=2.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=10.0, y=0.0),
    }
    velocities = {
        "agent_1": Velocity(vx=1.0, vy=0.0),
        "agent_2": Velocity(vx=-1.0, vy=0.0),
    }

    is_deadlocked = ca.detect_deadlock(positions, velocities)
    assert not is_deadlocked


# ── MA-CA-006: Agent size/radius modeling ─────────────────────────────


def test_agent_safety_radius_per_agent():
    """Different agents can have different safety radii."""
    ca = CollisionAvoidance(safety_radius=2.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=2.5, y=0.0),
    }
    radii = {"agent_1": 1.0, "agent_2": 2.0}

    collisions = ca.detect_collisions(positions, radii)
    # Combined radius = 3.0, distance 2.5 < 3.0 → collision
    assert len(collisions) > 0


# ── MA-CA-008: Collision prediction horizon ───────────────────────────


def test_collision_prediction_hazard():
    """Collision is predicted within the prediction horizon."""
    ca = CollisionAvoidance(safety_radius=2.0, prediction_horizon=5.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=10.0, y=0.0),
    }
    velocities = {
        "agent_1": Velocity(vx=1.0, vy=0.0),
        "agent_2": Velocity(vx=-1.0, vy=0.0),
    }

    collisions = ca.predict_collisions(positions, velocities)
    # Agents will collide in ~4 seconds, within 5s horizon
    assert len(collisions) > 0


def test_no_collision_prediction_when_safe():
    """No collision predicted when agents are safe."""
    ca = CollisionAvoidance(safety_radius=2.0, prediction_horizon=5.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=100.0, y=0.0),
    }
    velocities = {
        "agent_1": Velocity(vx=1.0, vy=0.0),
        "agent_2": Velocity(vx=-1.0, vy=0.0),
    }

    collisions = ca.predict_collisions(positions, velocities)
    assert len(collisions) == 0


# ── MA-CA-010: Avoidance velocity clamping ────────────────────────────


def test_avoidance_velocity_clamped():
    """Avoidance velocity is clamped to max_avoidance_speed."""
    ca = CollisionAvoidance(safety_radius=2.0, max_avoidance_speed=1.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=0.5, y=0.0),
        "agent_3": Position(x=0.0, y=0.5),
        "agent_4": Position(x=0.5, y=0.5),
    }
    velocities = {
        "agent_1": Velocity(vx=0.0, vy=0.0),
        "agent_2": Velocity(vx=0.0, vy=0.0),
        "agent_3": Velocity(vx=0.0, vy=0.0),
        "agent_4": Velocity(vx=0.0, vy=0.0),
    }

    result = ca.compute_all_avoidance_velocities(positions, velocities)
    for vel in result.values():
        speed = (vel.vx**2 + vel.vy**2) ** 0.5
        assert speed <= 1.0 + 1e-6

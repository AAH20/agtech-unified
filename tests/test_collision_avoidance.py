"""Test multi-agent collision avoidance for agricultural swarm robots."""

import pytest

from src.multi_agent.collision_avoidance import (
    CollisionAvoidance,
    Position,
    Velocity,
)

# ── Position / Velocity helpers ──────────────────────────────────────


def test_position_distance():
    """Distance between two positions is computed correctly."""
    p1 = Position(x=0.0, y=0.0)
    p2 = Position(x=3.0, y=4.0)
    assert p1.distance_to(p2) == pytest.approx(5.0)


def test_position_distance_zero():
    """Distance from a position to itself is zero."""
    p = Position(x=1.0, y=2.0)
    assert p.distance_to(p) == pytest.approx(0.0)


# ── CollisionAvoidance tests ─────────────────────────────────────────


def test_collision_avoidance_creation():
    """CollisionAvoidance can be instantiated with safety radius."""
    ca = CollisionAvoidance(safety_radius=2.0)
    assert ca.safety_radius == 2.0


def test_no_collision_when_far_apart():
    """Agents far apart are not in collision."""
    ca = CollisionAvoidance(safety_radius=2.0)
    p1 = Position(x=0.0, y=0.0)
    p2 = Position(x=10.0, y=0.0)
    assert not ca.is_collision(p1, p2)


def test_collision_when_too_close():
    """Agents within safety radius are in collision."""
    ca = CollisionAvoidance(safety_radius=2.0)
    p1 = Position(x=0.0, y=0.0)
    p2 = Position(x=1.0, y=0.0)
    assert ca.is_collision(p1, p2)


def test_collision_at_exact_boundary():
    """Agents at exactly safety radius distance are not in collision."""
    ca = CollisionAvoidance(safety_radius=2.0)
    p1 = Position(x=0.0, y=0.0)
    p2 = Position(x=2.0, y=0.0)
    assert not ca.is_collision(p1, p2)


def test_compute_avoidance_velocity():
    """Avoidance velocity pushes agent away from obstacle."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    obstacle_pos = Position(x=1.0, y=0.0)
    current_vel = Velocity(vx=1.0, vy=0.0)

    avoidance_vel = ca.compute_avoidance_velocity(agent_pos, obstacle_pos, current_vel)

    # Should push agent in negative x direction (away from obstacle)
    assert avoidance_vel.vx < 0


def test_no_avoidance_when_safe():
    """No avoidance velocity when agent is far from obstacle."""
    ca = CollisionAvoidance(safety_radius=2.0)
    agent_pos = Position(x=0.0, y=0.0)
    obstacle_pos = Position(x=10.0, y=0.0)
    current_vel = Velocity(vx=1.0, vy=0.0)

    avoidance_vel = ca.compute_avoidance_velocity(agent_pos, obstacle_pos, current_vel)

    assert avoidance_vel.vx == pytest.approx(0.0)
    assert avoidance_vel.vy == pytest.approx(0.0)


def test_detect_collisions_in_swarm():
    """Detect all colliding pairs in a swarm."""
    ca = CollisionAvoidance(safety_radius=2.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=1.0, y=0.0),
        "agent_3": Position(x=10.0, y=10.0),
    }

    collisions = ca.detect_collisions(positions)

    assert len(collisions) == 1
    pair = collisions[0]
    assert set(pair) == {"agent_1", "agent_2"}


def test_no_collisions_in_swarm():
    """No collisions when all agents are far apart."""
    ca = CollisionAvoidance(safety_radius=2.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=10.0, y=0.0),
        "agent_3": Position(x=0.0, y=10.0),
    }

    collisions = ca.detect_collisions(positions)

    assert len(collisions) == 0


def test_compute_all_avoidance_velocities():
    """Compute avoidance velocities for all agents in a swarm."""
    ca = CollisionAvoidance(safety_radius=2.0)
    positions = {
        "agent_1": Position(x=0.0, y=0.0),
        "agent_2": Position(x=1.0, y=0.0),
    }
    velocities = {
        "agent_1": Velocity(vx=1.0, vy=0.0),
        "agent_2": Velocity(vx=-1.0, vy=0.0),
    }

    result = ca.compute_all_avoidance_velocities(positions, velocities)

    assert "agent_1" in result
    assert "agent_2" in result
    # agent_1 should be pushed left (away from agent_2)
    assert result["agent_1"].vx < 0
    # agent_2 should be pushed right (away from agent_1)
    assert result["agent_2"].vx > 0

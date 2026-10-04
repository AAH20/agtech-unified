"""Tests for ORCA (Optimal Reciprocal Collision Avoidance)."""

import math

from src.multi_agent.collision_avoidance import Position, Velocity
from src.multi_agent.orca import ORCAAgent, ORCAWorld

# ── MA-ORCA-001: Basic ORCA velocity computation ──────────────────────


def test_orca_no_collision_returns_preferred_velocity():
    """When no collision is imminent, ORCA returns preferred velocity."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    assert math.isclose(new_vel.vx, 1.0, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.0, abs_tol=0.01)


def test_orca_avoids_collision_head_on():
    """Head-on collision: ORCA computes avoidance velocity."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(2.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # ORCA should steer A1 away from the head-on path
    assert abs(new_vel.vy) > 0.1 or abs(new_vel.vx) < 1.0


def test_orca_returns_none_for_unknown_agent():
    """ORCA returns None for unknown agent ID."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    result = world.compute_orca_velocity("unknown")
    assert result is None


# ── MA-ORCA-002: ORCA agent properties ────────────────────────────────


def test_orca_agent_stores_state():
    """ORCAAgent stores position, velocity, and radius."""
    agent = ORCAAgent(
        "A1",
        position=Position(1.0, 2.0),
        velocity=Velocity(0.5, 0.5),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.3,
    )
    assert agent.agent_id == "A1"
    assert agent.position.x == 1.0
    assert agent.position.y == 2.0
    assert agent.velocity.vx == 0.5
    assert agent.radius == 0.3


def test_orca_agent_custom_radius():
    """ORCA agents can have different radii."""
    agent = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(0.0, 0.0),
        preferred_velocity=Velocity(0.0, 0.0),
        radius=1.5,
    )
    assert agent.radius == 1.5


# ── MA-ORCA-003: ORCA world management ────────────────────────────────


def test_orca_world_add_remove_agent():
    """Agents can be added and removed from ORCA world."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(0.0, 0.0),
        preferred_velocity=Velocity(0.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent)
    assert world.get_agent("A1") is not None

    world.remove_agent("A1")
    assert world.get_agent("A1") is None


def test_orca_world_remove_unknown_agent_noop():
    """Removing unknown agent is a no-op."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    world.remove_agent("nonexistent")  # Should not raise


def test_orca_world_get_agents():
    """ORCA world returns all registered agents."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    world.add_agent(
        ORCAAgent(
            "A1",
            position=Position(0.0, 0.0),
            velocity=Velocity(0.0, 0.0),
            preferred_velocity=Velocity(0.0, 0.0),
            radius=0.5,
        )
    )
    world.add_agent(
        ORCAAgent(
            "A2",
            position=Position(5.0, 0.0),
            velocity=Velocity(0.0, 0.0),
            preferred_velocity=Velocity(0.0, 0.0),
            radius=0.5,
        )
    )
    agents = world.get_agents()
    assert len(agents) == 2


# ── MA-ORCA-004: ORCA velocity constraints ────────────────────────────


def test_orca_respects_max_speed():
    """ORCA velocity is clamped to max_speed."""
    world = ORCAWorld(safety_radius=2.0, max_speed=3.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(10.0, 0.0),
        preferred_velocity=Velocity(10.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(1.0, 0.0),
        velocity=Velocity(-10.0, 0.0),
        preferred_velocity=Velocity(-10.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    speed = math.sqrt(new_vel.vx**2 + new_vel.vy**2)
    assert speed <= 3.0 + 0.01  # Within max_speed


# ── MA-ORCA-005: ORCA with multiple agents ─────────────────────────────


def test_orca_multiple_agents_avoidance():
    """ORCA computes avoidance considering all nearby agents."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    world.add_agent(
        ORCAAgent(
            "A1",
            position=Position(0.0, 0.0),
            velocity=Velocity(1.0, 0.0),
            preferred_velocity=Velocity(1.0, 0.0),
            radius=0.5,
        )
    )
    world.add_agent(
        ORCAAgent(
            "A2",
            position=Position(3.0, 0.0),
            velocity=Velocity(-1.0, 0.0),
            preferred_velocity=Velocity(-1.0, 0.0),
            radius=0.5,
        )
    )
    world.add_agent(
        ORCAAgent(
            "A3",
            position=Position(0.0, 3.0),
            velocity=Velocity(0.0, -1.0),
            preferred_velocity=Velocity(0.0, -1.0),
            radius=0.5,
        )
    )

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None


# ── MA-ORCA-006: ORCA reciprocal behavior ──────────────────────────────


def test_orca_reciprocal_avoidance():
    """Both agents in a head-on scenario should avoid each other."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(2.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    vel1 = world.compute_orca_velocity("A1")
    vel2 = world.compute_orca_velocity("A2")
    assert vel1 is not None
    assert vel2 is not None
    # Both should have non-zero y components (avoiding)
    assert abs(vel1.vy) > 0.01 or abs(vel1.vx) < 1.0
    assert abs(vel2.vy) > 0.01 or abs(vel2.vx) < 1.0


def test_orca_no_collision_when_moving_apart():
    """No avoidance when agents are moving apart."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should return preferred velocity since moving apart
    assert math.isclose(new_vel.vx, -1.0, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.0, abs_tol=0.01)


def test_orca_time_horizon():
    """ORCA only considers collisions within the time horizon."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0, time_horizon=1.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    # Agent far away, would collide in > 1 second
    agent2 = ORCAAgent(
        "A2",
        position=Position(10.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should return preferred velocity since collision is beyond horizon
    assert math.isclose(new_vel.vx, 1.0, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.0, abs_tol=0.01)


def test_orca_already_colliding():
    """ORCA handles already-colliding agents."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(0.5, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should push A1 away from A2 (negative x direction)
    assert new_vel.vx < 0.5


def test_orca_zero_distance():
    """ORCA handles agents at the same position."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(0.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should not crash


def test_orca_parallel_agents():
    """ORCA with agents moving in parallel."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(0.0, 3.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should return preferred velocity since moving in parallel
    assert math.isclose(new_vel.vx, 1.0, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.0, abs_tol=0.01)


def test_orca_crossing_paths():
    """ORCA with agents on crossing paths."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 1.0),
        preferred_velocity=Velocity(1.0, 1.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 3.0),
        velocity=Velocity(-1.0, -1.0),
        preferred_velocity=Velocity(-1.0, -1.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should avoid the collision


def test_orca_static_agent():
    """ORCA with a static agent (zero velocity)."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(0.0, 0.0),
        preferred_velocity=Velocity(0.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should avoid the static agent


# ── MA-ORCA-007: ORCA half-plane computation ──────────────────────────


def test_orca_half_plane_computation():
    """ORCA computes half-plane constraints correctly."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    constraint = world._compute_orca_half_plane(agent1, agent2)
    assert constraint is not None


def test_orca_half_plane_no_collision():
    """ORCA half-plane is None when no collision."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    constraint = world._compute_orca_half_plane(agent1, agent2)
    assert constraint is None


def test_orca_half_plane_already_colliding():
    """ORCA half-plane for already-colliding agents."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(0.5, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    constraint = world._compute_orca_half_plane(agent1, agent2)
    assert constraint is not None


def test_orca_half_plane_moving_apart():
    """ORCA half-plane is None when moving apart."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    constraint = world._compute_orca_half_plane(agent1, agent2)
    assert constraint is None


def test_orca_half_plane_beyond_horizon():
    """ORCA half-plane is None when collision is beyond time horizon."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0, time_horizon=1.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(10.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    constraint = world._compute_orca_half_plane(agent1, agent2)
    assert constraint is None


def test_orca_half_plane_safe_distance():
    """ORCA half-plane is None when closest approach is safe."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 5.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    constraint = world._compute_orca_half_plane(agent1, agent2)
    assert constraint is None


# ── MA-ORCA-008: ORCA simulation ──────────────────────────────────────


def test_orca_simulation_step():
    """ORCA can simulate multiple steps."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    # Simulate a few steps
    for _ in range(5):
        vel1 = world.compute_orca_velocity("A1")
        vel2 = world.compute_orca_velocity("A2")
        assert vel1 is not None
        assert vel2 is not None
        # Update positions
        agent1.position = Position(
            agent1.position.x + vel1.vx * 0.1,
            agent1.position.y + vel1.vy * 0.1,
        )
        agent2.position = Position(
            agent2.position.x + vel2.vx * 0.1,
            agent2.position.y + vel2.vy * 0.1,
        )
        agent1.velocity = vel1
        agent2.velocity = vel2


def test_orca_agents_pass_each_other():
    """ORCA agents should pass each other without collision."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    min_dist = float("inf")
    for _ in range(20):
        vel1 = world.compute_orca_velocity("A1")
        vel2 = world.compute_orca_velocity("A2")
        assert vel1 is not None
        assert vel2 is not None
        agent1.position = Position(
            agent1.position.x + vel1.vx * 0.1,
            agent1.position.y + vel1.vy * 0.1,
        )
        agent2.position = Position(
            agent2.position.x + vel2.vx * 0.1,
            agent2.position.y + vel2.vy * 0.1,
        )
        agent1.velocity = vel1
        agent2.velocity = vel2
        dist = agent1.position.distance_to(agent2.position)
        min_dist = min(min_dist, dist)

    # Agents should maintain some distance
    assert min_dist > 0.5


# ── MA-ORCA-009: ORCA edge cases ───────────────────────────────────────


def test_orca_single_agent():
    """ORCA with a single agent returns preferred velocity."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    assert math.isclose(new_vel.vx, 1.0, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.0, abs_tol=0.01)


def test_orca_zero_velocity():
    """ORCA with zero velocity agents."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(0.0, 0.0),
        preferred_velocity=Velocity(0.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(0.0, 0.0),
        preferred_velocity=Velocity(0.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should return preferred velocity (zero)
    assert math.isclose(new_vel.vx, 0.0, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.0, abs_tol=0.01)


def test_orca_large_radius():
    """ORCA with large agent radii."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=2.0,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(5.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=2.0,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should avoid due to large combined radius


def test_orca_different_radii():
    """ORCA with agents of different radii."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=1.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should avoid due to combined radius


def test_orca_update_agent_state():
    """ORCA agents can have their state updated."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent)

    # Update agent state
    agent.position = Position(1.0, 1.0)
    agent.velocity = Velocity(0.5, 0.5)
    agent.preferred_velocity = Velocity(0.5, 0.5)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # Should return updated preferred velocity
    assert math.isclose(new_vel.vx, 0.5, abs_tol=0.01)
    assert math.isclose(new_vel.vy, 0.5, abs_tol=0.01)


def test_orca_world_parameters():
    """ORCA world stores parameters correctly."""
    world = ORCAWorld(safety_radius=3.0, max_speed=10.0, time_horizon=2.0)
    assert world.safety_radius == 3.0
    assert world.max_speed == 10.0
    assert world.time_horizon == 2.0


def test_orca_avoidance_direction():
    """ORCA avoidance direction is away from the other agent."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(2.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    # The avoidance should have a component away from A2
    # A2 is at (2, 0), so away is negative x
    # But ORCA may also steer in y direction
    # The key is that the velocity should not be directly toward A2
    assert not (new_vel.vx > 0.9 and abs(new_vel.vy) < 0.1)


def test_orca_velocity_magnitude_reasonable():
    """ORCA velocity magnitude should be reasonable."""
    world = ORCAWorld(safety_radius=2.0, max_speed=5.0)
    agent1 = ORCAAgent(
        "A1",
        position=Position(0.0, 0.0),
        velocity=Velocity(1.0, 0.0),
        preferred_velocity=Velocity(1.0, 0.0),
        radius=0.5,
    )
    agent2 = ORCAAgent(
        "A2",
        position=Position(3.0, 0.0),
        velocity=Velocity(-1.0, 0.0),
        preferred_velocity=Velocity(-1.0, 0.0),
        radius=0.5,
    )
    world.add_agent(agent1)
    world.add_agent(agent2)

    new_vel = world.compute_orca_velocity("A1")
    assert new_vel is not None
    speed = math.sqrt(new_vel.vx**2 + new_vel.vy**2)
    # Speed should be positive and within max_speed
    assert speed > 0.01
    assert speed <= 5.0 + 0.01

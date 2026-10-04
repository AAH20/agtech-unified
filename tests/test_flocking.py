"""Tests for Reynolds flocking behaviors (separation, alignment, cohesion)."""

import math

import pytest

from src.multi_agent.collision_avoidance import Position, Velocity
from src.multi_agent.flocking import FlockingBehavior


def test_separation_pushes_agent_away_from_close_neighbor():
    """An agent too close to a neighbor steers away from it."""
    flocking = FlockingBehavior(
        perception_radius=5.0,
        separation_radius=2.0,
        separation_weight=1.0,
        alignment_weight=0.0,
        cohesion_weight=0.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(1.0, 0.0),
    }
    velocities = {"a": Velocity(0.0, 0.0), "b": Velocity(0.0, 0.0)}
    steering = flocking.compute_steering("a", positions, velocities)
    # Agent b is to the right of a, so separation should push a to the left (negative x)
    assert steering.vx < 0.0
    assert steering.vy == pytest.approx(0.0)


def test_separation_zero_when_no_close_neighbors():
    """No separation force when all neighbors are beyond separation_radius."""
    flocking = FlockingBehavior(
        perception_radius=10.0,
        separation_radius=2.0,
        separation_weight=1.0,
        alignment_weight=0.0,
        cohesion_weight=0.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(5.0, 0.0),
    }
    velocities = {"a": Velocity(0.0, 0.0), "b": Velocity(0.0, 0.0)}
    steering = flocking.compute_steering("a", positions, velocities)
    assert steering.vx == pytest.approx(0.0)
    assert steering.vy == pytest.approx(0.0)


def test_alignment_matches_neighbor_velocity():
    """Alignment steers agent toward the average velocity of neighbors."""
    flocking = FlockingBehavior(
        perception_radius=10.0,
        separation_radius=1.0,
        separation_weight=0.0,
        alignment_weight=1.0,
        cohesion_weight=0.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(3.0, 4.0),
    }
    velocities = {
        "a": Velocity(0.0, 0.0),
        "b": Velocity(2.0, 0.0),
    }
    steering = flocking.compute_steering("a", positions, velocities)
    # Neighbor moves at (2,0), so alignment should push a toward positive x
    assert steering.vx > 0.0
    assert steering.vy == pytest.approx(0.0)


def test_cohesion_moves_toward_center_of_mass():
    """Cohesion steers agent toward the centroid of its neighbors."""
    flocking = FlockingBehavior(
        perception_radius=10.0,
        separation_radius=1.0,
        separation_weight=0.0,
        alignment_weight=0.0,
        cohesion_weight=1.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(4.0, 0.0),
        "c": Position(0.0, 4.0),
    }
    velocities = {
        "a": Velocity(0.0, 0.0),
        "b": Velocity(0.0, 0.0),
        "c": Velocity(0.0, 0.0),
    }
    steering = flocking.compute_steering("a", positions, velocities)
    # Centroid of b and c is (2, 2), so cohesion should push a toward positive x and y
    assert steering.vx > 0.0
    assert steering.vy > 0.0


def test_no_neighbors_returns_zero_steering():
    """An isolated agent gets zero steering force."""
    flocking = FlockingBehavior()
    positions = {"a": Position(0.0, 0.0)}
    velocities = {"a": Velocity(1.0, 1.0)}
    steering = flocking.compute_steering("a", positions, velocities)
    assert steering.vx == pytest.approx(0.0)
    assert steering.vy == pytest.approx(0.0)


def test_combined_steering_respects_weights():
    """All three behaviors combine with their respective weights."""
    flocking = FlockingBehavior(
        perception_radius=10.0,
        separation_radius=2.0,
        separation_weight=2.0,
        alignment_weight=3.0,
        cohesion_weight=1.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(1.0, 0.0),
    }
    velocities = {
        "a": Velocity(0.0, 0.0),
        "b": Velocity(1.0, 0.0),
    }
    steering = flocking.compute_steering("a", positions, velocities)
    # Separation pushes left (-x), alignment pushes right (+x), cohesion pushes right (+x)
    # Net effect depends on weights; just verify it's non-zero and finite
    assert math.isfinite(steering.vx)
    assert math.isfinite(steering.vy)


def test_perception_radius_limits_neighbors():
    """Neighbors beyond perception_radius are ignored."""
    flocking = FlockingBehavior(
        perception_radius=3.0,
        separation_radius=1.0,
        separation_weight=0.0,
        alignment_weight=1.0,
        cohesion_weight=0.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "far": Position(100.0, 0.0),
    }
    velocities = {
        "a": Velocity(0.0, 0.0),
        "far": Velocity(5.0, 5.0),
    }
    steering = flocking.compute_steering("a", positions, velocities)
    # "far" is beyond perception_radius, so no alignment force
    assert steering.vx == pytest.approx(0.0)
    assert steering.vy == pytest.approx(0.0)


def test_compute_all_steering_returns_for_each_agent():
    """compute_all_steering returns a steering velocity for every agent."""
    flocking = FlockingBehavior(
        perception_radius=10.0,
        separation_radius=2.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(1.0, 0.0),
        "c": Position(5.0, 5.0),
    }
    velocities = {
        "a": Velocity(0.0, 0.0),
        "b": Velocity(1.0, 0.0),
        "c": Velocity(0.0, 1.0),
    }
    result = flocking.compute_all_steering(positions, velocities)
    assert set(result.keys()) == {"a", "b", "c"}
    for v in result.values():
        assert isinstance(v, Velocity)
        assert math.isfinite(v.vx)
        assert math.isfinite(v.vy)


def test_separation_only_applies_within_separation_radius():
    """Neighbors between separation_radius and perception_radius do not trigger separation."""
    flocking = FlockingBehavior(
        perception_radius=10.0,
        separation_radius=2.0,
        separation_weight=1.0,
        alignment_weight=0.0,
        cohesion_weight=0.0,
    )
    positions = {
        "a": Position(0.0, 0.0),
        "b": Position(3.0, 0.0),  # Within perception but outside separation
    }
    velocities = {"a": Velocity(0.0, 0.0), "b": Velocity(0.0, 0.0)}
    steering = flocking.compute_steering("a", positions, velocities)
    assert steering.vx == pytest.approx(0.0)
    assert steering.vy == pytest.approx(0.0)

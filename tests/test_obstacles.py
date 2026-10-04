"""Tests for obstacle field path planning."""
import pytest
from src.path_planning.obstacles import ObstacleField


def test_is_obstacle_inside():
    """Point inside an obstacle returns True."""
    field = ObstacleField([(0, 0, 5)])
    assert field.is_obstacle((3, 3)) is True


def test_is_obstacle_outside():
    """Point outside all obstacles returns False."""
    field = ObstacleField([(0, 0, 5)])
    assert field.is_obstacle((10, 10)) is False


def test_avoid_path_returns_waypoints():
    """avoid_path returns detour waypoints around the obstacle."""
    field = ObstacleField()
    path = field.avoid_path((0, 0), (10, 0), (5, 0, 2))
    assert len(path) > 0
    assert all(isinstance(p, tuple) and len(p) == 2 for p in path)


def test_empty_obstacles():
    """No obstacles means no point is an obstacle."""
    field = ObstacleField([])
    assert field.is_obstacle((0, 0)) is False
    assert field.is_obstacle((100, 100)) is False


def test_multiple_obstacles():
    """Multiple obstacles are all detected."""
    field = ObstacleField([(0, 0, 2), (10, 10, 3)])
    assert field.is_obstacle((0, 0)) is True
    assert field.is_obstacle((10, 10)) is True
    assert field.is_obstacle((5, 5)) is False

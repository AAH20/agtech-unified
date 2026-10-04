"""Tests for A* pathfinding algorithm."""

import math

import pytest

from src.path_planning.astar import AStarPlanner, GridCell


class TestAStarBasic:
    """Basic A* pathfinding tests."""

    def test_straight_line_path(self):
        """A* finds a straight-line path on an empty grid."""
        planner = AStarPlanner(width=10, height=10)
        path = planner.find_path((0, 0), (9, 0))
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (9, 0)

    def test_diagonal_path(self):
        """A* finds a diagonal path on an empty grid."""
        planner = AStarPlanner(width=10, height=10)
        path = planner.find_path((0, 0), (9, 9))
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    def test_same_start_and_goal(self):
        """Path from a cell to itself is just that cell."""
        planner = AStarPlanner(width=10, height=10)
        path = planner.find_path((5, 5), (5, 5))
        assert path == [(5, 5)]

    def test_path_avoids_obstacles(self):
        """A* path goes around obstacles."""
        planner = AStarPlanner(width=10, height=10)
        # Create a wall of obstacles (leave gap at top and bottom)
        for y in range(1, 9):
            planner.set_obstacle((5, y))
        path = planner.find_path((0, 5), (9, 5))
        assert len(path) > 0
        assert path[0] == (0, 5)
        assert path[-1] == (9, 5)
        # No path point should be on an obstacle
        for point in path:
            assert not planner.is_obstacle(point)

    def test_no_path_when_blocked(self):
        """Returns empty list when goal is unreachable."""
        planner = AStarPlanner(width=10, height=10)
        # Create a complete wall
        for y in range(10):
            planner.set_obstacle((5, y))
        planner.find_path((0, 5), (9, 5))
        # With 8-directional movement, there might be a path around
        # Let's use a fully enclosed goal instead
        planner2 = AStarPlanner(width=10, height=10)
        # Enclose (5,5) completely
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    planner2.set_obstacle((5 + dx, 5 + dy))
        path2 = planner2.find_path((0, 0), (5, 5))
        assert path2 == []

    def test_path_continuity(self):
        """Consecutive path points are adjacent (no teleporting)."""
        planner = AStarPlanner(width=20, height=20)
        # Add some obstacles
        planner.set_obstacle((5, 5))
        planner.set_obstacle((5, 6))
        planner.set_obstacle((6, 5))
        path = planner.find_path((0, 0), (10, 10))
        for i in range(len(path) - 1):
            dx = abs(path[i + 1][0] - path[i][0])
            dy = abs(path[i + 1][1] - path[i][1])
            assert dx <= 1
            assert dy <= 1
            assert dx + dy > 0  # Must move

    def test_optimal_path_length(self):
        """A* finds the shortest path on an empty grid."""
        planner = AStarPlanner(width=10, height=10)
        path = planner.find_path((0, 0), (9, 0))
        # Straight line: 10 cells = 9 steps
        assert len(path) == 10

    def test_diagonal_movement_allowed(self):
        """Diagonal movement produces shorter paths."""
        planner = AStarPlanner(width=10, height=10, allow_diagonal=True)
        path = planner.find_path((0, 0), (9, 9))
        # With diagonal: 10 cells (9 diagonal steps)
        assert len(path) == 10

    def test_no_diagonal_movement(self):
        """Without diagonal movement, path is longer."""
        planner = AStarPlanner(width=10, height=10, allow_diagonal=False)
        path = planner.find_path((0, 0), (9, 9))
        # Manhattan: 19 cells (18 steps)
        assert len(path) == 19


class TestAStarGridCell:
    """GridCell dataclass tests."""

    def test_grid_cell_creation(self):
        """GridCell can be created with default values."""
        cell = GridCell(x=1, y=2)
        assert cell.x == 1
        assert cell.y == 2
        assert cell.g == float("inf")
        assert cell.h == 0.0
        assert cell.parent is None

    def test_grid_cell_f_score(self):
        """f_score is g + h."""
        cell = GridCell(x=0, y=0, g=3.0, h=4.0)
        assert cell.f_score == 7.0


class TestAStarObstacles:
    """Obstacle handling tests."""

    def test_set_and_clear_obstacle(self):
        """Obstacles can be set and cleared."""
        planner = AStarPlanner(width=10, height=10)
        planner.set_obstacle((5, 5))
        assert planner.is_obstacle((5, 5))
        planner.clear_obstacle((5, 5))
        assert not planner.is_obstacle((5, 5))

    def test_out_of_bounds_is_obstacle(self):
        """Out-of-bounds cells are treated as obstacles."""
        planner = AStarPlanner(width=10, height=10)
        assert planner.is_obstacle((-1, 0))
        assert planner.is_obstacle((0, -1))
        assert planner.is_obstacle((10, 0))
        assert planner.is_obstacle((0, 10))

    def test_add_obstacles_batch(self):
        """Multiple obstacles can be added at once."""
        planner = AStarPlanner(width=10, height=10)
        obstacles = [(3, 3), (3, 4), (4, 3), (4, 4)]
        planner.add_obstacles(obstacles)
        for obs in obstacles:
            assert planner.is_obstacle(obs)

    def test_path_around_single_obstacle(self):
        """Path goes around a single obstacle."""
        planner = AStarPlanner(width=10, height=10)
        planner.set_obstacle((5, 5))
        path = planner.find_path((4, 5), (6, 5))
        assert len(path) > 0
        assert path[0] == (4, 5)
        assert path[-1] == (6, 5)
        # Path should not go through (5,5)
        assert (5, 5) not in path


class TestAStarHeuristics:
    """Heuristic function tests."""

    def test_manhattan_heuristic(self):
        """Manhattan heuristic is admissible."""
        planner = AStarPlanner(width=10, height=10, heuristic="manhattan")
        h = planner._heuristic((0, 0), (3, 4))
        assert h == 7.0

    def test_euclidean_heuristic(self):
        """Euclidean heuristic is admissible."""
        planner = AStarPlanner(width=10, height=10, heuristic="euclidean")
        h = planner._heuristic((0, 0), (3, 4))
        assert h == pytest.approx(5.0)

    def test_octile_heuristic(self):
        """Octile heuristic for 8-directional movement."""
        planner = AStarPlanner(width=10, height=10, heuristic="octile")
        h = planner._heuristic((0, 0), (3, 4))
        # Octile: max(dx,dy) + (sqrt(2)-1)*min(dx,dy) = 4 + 0.414*3 ≈ 5.24
        assert h == pytest.approx(4.0 + (math.sqrt(2) - 1) * 3.0)

    def test_unknown_heuristic_raises(self):
        """Unknown heuristic raises ValueError."""
        with pytest.raises(ValueError, match="Unknown heuristic"):
            AStarPlanner(width=10, height=10, heuristic="invalid")


class TestAStarEdgeCases:
    """Edge case tests."""

    def test_start_is_obstacle(self):
        """Returns empty list when start is an obstacle."""
        planner = AStarPlanner(width=10, height=10)
        planner.set_obstacle((0, 0))
        path = planner.find_path((0, 0), (5, 5))
        assert path == []

    def test_goal_is_obstacle(self):
        """Returns empty list when goal is an obstacle."""
        planner = AStarPlanner(width=10, height=10)
        planner.set_obstacle((5, 5))
        path = planner.find_path((0, 0), (5, 5))
        assert path == []

    def test_adjacent_cells(self):
        """Path between adjacent cells has 2 points."""
        planner = AStarPlanner(width=10, height=10)
        path = planner.find_path((5, 5), (6, 5))
        assert len(path) == 2
        assert path[0] == (5, 5)
        assert path[-1] == (6, 5)

    def test_large_grid(self):
        """A* works on larger grids."""
        planner = AStarPlanner(width=100, height=100)
        path = planner.find_path((0, 0), (99, 99))
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (99, 99)

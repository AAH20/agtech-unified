"""Tests for D* Lite incremental replanning algorithm."""

import math

from src.path_planning.dstar_lite import DStarLite, DStarLiteResult


class TestDStarLiteBasic:
    """Basic D* Lite functionality."""

    def test_path_found_empty_grid(self):
        """D* Lite finds a path on an empty grid."""
        planner = DStarLite(width=10, height=10)
        result = planner.find_path((0, 0), (9, 9))
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (9, 9)

    def test_same_start_and_goal(self):
        """Path from a cell to itself is just that cell."""
        planner = DStarLite(width=10, height=10)
        result = planner.find_path((5, 5), (5, 5))
        assert result.success
        assert result.path == [(5, 5)]

    def test_path_avoids_obstacles(self):
        """D* Lite path goes around obstacles."""
        planner = DStarLite(width=10, height=10)
        # Wall with gap at top
        for y in range(1, 9):
            planner.set_obstacle((5, y))
        result = planner.find_path((0, 5), (9, 5))
        assert result.success
        assert result.path[0] == (0, 5)
        assert result.path[-1] == (9, 5)
        for point in result.path:
            assert not planner.is_obstacle(point)

    def test_no_path_when_fully_blocked(self):
        """Returns failure when goal is unreachable."""
        planner = DStarLite(width=10, height=10)
        # Full vertical wall
        for y in range(10):
            planner.set_obstacle((5, y))
        result = planner.find_path((0, 5), (9, 5))
        assert not result.success
        assert result.path == []

    def test_start_or_goal_is_obstacle(self):
        """Returns failure when start or goal is an obstacle."""
        planner = DStarLite(width=10, height=10)
        planner.set_obstacle((0, 0))
        result = planner.find_path((0, 0), (5, 5))
        assert not result.success

        planner2 = DStarLite(width=10, height=10)
        planner2.set_obstacle((5, 5))
        result2 = planner2.find_path((0, 0), (5, 5))
        assert not result2.success


class TestDStarLiteIncremental:
    """Incremental replanning tests."""

    def test_replan_after_adding_obstacle(self):
        """Replanning after adding an obstacle finds a new path."""
        planner = DStarLite(width=10, height=10)
        result1 = planner.find_path((0, 5), (9, 5))
        assert result1.success

        # Add obstacle on the original path
        mid = result1.path[len(result1.path) // 2]
        planner.set_obstacle(mid)
        result2 = planner.find_path((0, 5), (9, 5))
        assert result2.success
        assert result2.path[0] == (0, 5)
        assert result2.path[-1] == (9, 5)
        for point in result2.path:
            assert not planner.is_obstacle(point)

    def test_replan_after_removing_obstacle(self):
        """Replanning after removing an obstacle may find a shorter path."""
        planner = DStarLite(width=10, height=10)
        # Block direct path
        for y in range(10):
            planner.set_obstacle((5, y))
        result1 = planner.find_path((0, 5), (9, 5))
        assert not result1.success

        # Remove some obstacles to open a path
        for y in range(3, 7):
            planner.clear_obstacle((5, y))
        result2 = planner.find_path((0, 5), (9, 5))
        assert result2.success
        assert result2.path[0] == (0, 5)
        assert result2.path[-1] == (9, 5)

    def test_replan_multiple_times(self):
        """Multiple replanning cycles work correctly."""
        planner = DStarLite(width=15, height=15)
        start, goal = (0, 7), (14, 7)

        result = planner.find_path(start, goal)
        assert result.success

        # Add obstacles and replan
        for i in range(3):
            planner.set_obstacle((5 + i * 3, 7))
            result = planner.find_path(start, goal)
            assert result.success
            assert result.path[0] == start
            assert result.path[-1] == goal

    def test_replan_with_unknown_cells(self):
        """D* Lite handles unknown (unexplored) cells."""
        planner = DStarLite(width=10, height=10)
        # Mark some cells as unknown
        planner.set_unknown((3, 3))
        planner.set_unknown((3, 4))
        planner.set_unknown((4, 3))
        result = planner.find_path((0, 0), (9, 9))
        assert result.success
        # Path should avoid unknown cells
        for point in result.path:
            assert not planner.is_unknown(point)

    def test_path_cost_consistency(self):
        """Path cost is consistent with path length."""
        planner = DStarLite(width=10, height=10)
        result = planner.find_path((0, 0), (9, 0))
        assert result.success
        # Straight horizontal path: cost should be ~9
        assert abs(result.cost - 9.0) < 0.01

    def test_diagonal_movement(self):
        """D* Lite supports diagonal movement."""
        planner = DStarLite(width=10, height=10, allow_diagonal=True)
        result = planner.find_path((0, 0), (9, 9))
        assert result.success
        # Diagonal path cost should be ~9*sqrt(2)
        expected = 9.0 * math.sqrt(2)
        assert abs(result.cost - expected) < 0.1

    def test_no_diagonal_movement(self):
        """D* Lite works without diagonal movement."""
        planner = DStarLite(width=10, height=10, allow_diagonal=False)
        result = planner.find_path((0, 0), (9, 9))
        assert result.success
        # Manhattan path cost should be ~18
        assert abs(result.cost - 18.0) < 0.01


class TestDStarLiteEdgeCases:
    """Edge case tests."""

    def test_out_of_bounds_start(self):
        """Returns failure for out-of-bounds start."""
        planner = DStarLite(width=10, height=10)
        result = planner.find_path((-1, 0), (5, 5))
        assert not result.success

    def test_out_of_bounds_goal(self):
        """Returns failure for out-of-bounds goal."""
        planner = DStarLite(width=10, height=10)
        result = planner.find_path((0, 0), (10, 5))
        assert not result.success

    def test_single_cell_grid(self):
        """Works on a 1x1 grid."""
        planner = DStarLite(width=1, height=1)
        result = planner.find_path((0, 0), (0, 0))
        assert result.success
        assert result.path == [(0, 0)]

    def test_narrow_passage(self):
        """Finds path through a narrow passage."""
        planner = DStarLite(width=10, height=10)
        # Create a vertical wall at x=5 with a gap at y=5
        for y in range(10):
            if y != 5:
                planner.set_obstacle((5, y))
        result = planner.find_path((0, 0), (9, 9))
        assert result.success
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (9, 9)

    def test_result_dataclass(self):
        """DStarLiteResult has expected fields."""
        planner = DStarLite(width=5, height=5)
        result = planner.find_path((0, 0), (4, 4))
        assert isinstance(result, DStarLiteResult)
        assert hasattr(result, "success")
        assert hasattr(result, "path")
        assert hasattr(result, "cost")
        assert hasattr(result, "nodes_expanded")
        assert result.nodes_expanded >= 0

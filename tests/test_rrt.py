"""Tests for RRT sampling-based path planning."""

import math

from src.path_planning.rrt import RRTConfig, RRTPlanner


class TestRRTBasic:
    """Basic RRT functionality."""

    def test_path_found_empty_environment(self):
        """RRT finds a direct path when no obstacles exist."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=RRTConfig(max_iterations=500, step_size=1.0, goal_tolerance=0.5),
        )
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (0.0, 0.0)
        assert result.path[-1] == (10.0, 0.0)

    def test_path_starts_at_start_and_ends_at_goal(self):
        """Path endpoints match start and goal exactly."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(1.0, 2.0),
            goal=(8.0, 6.0),
            obstacles=[],
            config=RRTConfig(max_iterations=500, step_size=1.0, goal_tolerance=0.5),
        )
        assert result.success
        assert result.path[0] == (1.0, 2.0)
        assert result.path[-1] == (8.0, 6.0)

    def test_no_collision_with_obstacles(self):
        """Path does not pass through any obstacle."""
        obstacles = [(5.0, 0.0, 1.5)]
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=RRTConfig(max_iterations=2000, step_size=0.5, goal_tolerance=0.5),
        )
        assert result.success
        for x, y in result.path:
            for ox, oy, r in obstacles:
                dist = math.hypot(x - ox, y - oy)
                assert dist > r, f"Path point ({x}, {y}) inside obstacle at ({ox}, {oy}) r={r}"

    def test_path_avoids_single_obstacle(self):
        """RRT routes around a single blocking obstacle."""
        obstacles = [(5.0, 0.0, 2.0)]
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=RRTConfig(max_iterations=3000, step_size=0.5, goal_tolerance=0.5),
        )
        assert result.success
        # Path should not be a straight line (must detour)
        assert len(result.path) > 2

    def test_goal_bias_improves_convergence(self):
        """With high goal bias, RRT converges faster."""
        obstacles = [(5.0, 0.0, 1.0)]
        config = RRTConfig(max_iterations=1000, step_size=0.5, goal_tolerance=0.5, goal_bias=0.3)
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=config,
        )
        assert result.success
        assert result.iterations < 1000

    def test_returns_failure_when_goal_blocked(self):
        """RRT returns failure when goal is inside an obstacle."""
        obstacles = [(10.0, 0.0, 2.0)]
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=RRTConfig(max_iterations=500, step_size=0.5, goal_tolerance=0.5),
        )
        assert not result.success
        assert result.path == []

    def test_returns_failure_when_start_blocked(self):
        """RRT returns failure when start is inside an obstacle."""
        obstacles = [(0.0, 0.0, 2.0)]
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=RRTConfig(max_iterations=500, step_size=0.5, goal_tolerance=0.5),
        )
        assert not result.success

    def test_respects_max_iterations(self):
        """RRT stops after max_iterations."""
        config = RRTConfig(max_iterations=50, step_size=0.5, goal_tolerance=0.5)
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(100.0, 100.0),
            obstacles=[],
            config=config,
        )
        assert result.iterations <= 50

    def test_path_continuity(self):
        """Consecutive path points are within step_size distance."""
        planner = RRTPlanner()
        config = RRTConfig(max_iterations=1000, step_size=1.0, goal_tolerance=0.5)
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 5.0),
            obstacles=[(5.0, 2.0, 1.0)],
            config=config,
        )
        assert result.success
        for i in range(len(result.path) - 1):
            dx = result.path[i + 1][0] - result.path[i][0]
            dy = result.path[i + 1][1] - result.path[i][1]
            dist = math.hypot(dx, dy)
            assert dist <= config.step_size + 1e-6

    def test_multiple_obstacles(self):
        """RRT navigates through a field with multiple obstacles."""
        obstacles = [
            (3.0, 1.0, 1.0),
            (6.0, -1.0, 1.0),
            (5.0, 3.0, 0.8),
        ]
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=RRTConfig(max_iterations=5000, step_size=0.5, goal_tolerance=0.5),
        )
        assert result.success
        for x, y in result.path:
            for ox, oy, r in obstacles:
                dist = math.hypot(x - ox, y - oy)
                assert dist > r

    def test_tree_growth(self):
        """RRT tree grows with iterations."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(5.0, 5.0),
            obstacles=[],
            config=RRTConfig(max_iterations=200, step_size=1.0, goal_tolerance=0.5),
        )
        assert result.success
        assert len(result.nodes) > 1

    def test_goal_tolerance(self):
        """RRT accepts goal within tolerance distance."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=RRTConfig(max_iterations=500, step_size=2.0, goal_tolerance=3.0),
        )
        assert result.success
        # Last point should be within goal_tolerance of goal
        final_dist = math.hypot(result.path[-1][0] - 10.0, result.path[-1][1] - 0.0)
        assert final_dist <= 3.0 + 1e-6


class TestRRTEdgeCases:
    """Edge cases and boundary conditions."""

    def test_start_equals_goal(self):
        """RRT handles start == goal."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(5.0, 5.0),
            goal=(5.0, 5.0),
            obstacles=[],
            config=RRTConfig(max_iterations=100, step_size=1.0, goal_tolerance=0.5),
        )
        assert result.success
        assert result.path == [(5.0, 5.0)]

    def test_very_close_start_goal(self):
        """RRT handles very close start and goal."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(0.1, 0.1),
            obstacles=[],
            config=RRTConfig(max_iterations=100, step_size=1.0, goal_tolerance=0.5),
        )
        assert result.success
        # Within goal_tolerance, path is just [start]
        assert result.path == [(0.0, 0.0)]

    def test_narrow_passage(self):
        """RRT can find path through narrow passage between obstacles."""
        obstacles = [
            (5.0, -2.0, 1.5),
            (5.0, 2.0, 1.5),
        ]
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=RRTConfig(max_iterations=10000, step_size=0.3, goal_tolerance=0.5),
        )
        # May or may not find path depending on sampling, but should not crash
        if result.success:
            for x, y in result.path:
                for ox, oy, r in obstacles:
                    dist = math.hypot(x - ox, y - oy)
                    assert dist > r

    def test_result_contains_tree_info(self):
        """RRTResult contains tree nodes for inspection."""
        planner = RRTPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(5.0, 0.0),
            obstacles=[],
            config=RRTConfig(max_iterations=200, step_size=1.0, goal_tolerance=0.5),
        )
        assert result.success
        assert hasattr(result, "nodes")
        assert len(result.nodes) > 0
        # First node should be the start
        assert result.nodes[0].x == 0.0
        assert result.nodes[0].y == 0.0

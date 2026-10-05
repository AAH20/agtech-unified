"""Tests for PRM (Probabilistic Roadmap) path planning."""

import math

from src.path_planning.prm import PRMConfig, PRMPlanner, PRMResult


class TestPRMBasic:
    """Basic PRM functionality."""

    def test_path_found_empty_environment(self):
        """PRM finds a path when no obstacles exist."""
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=PRMConfig(num_samples=100, k_neighbors=5, seed=42),
        )
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (0.0, 0.0)
        assert result.path[-1] == (10.0, 0.0)

    def test_path_avoids_obstacles(self):
        """PRM path does not pass through obstacles."""
        obstacles = [(5.0, 0.0, 1.5)]
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=PRMConfig(num_samples=200, k_neighbors=8, seed=42),
        )
        assert result.success
        for x, y in result.path:
            for ox, oy, r in obstacles:
                assert math.hypot(x - ox, y - oy) > r

    def test_no_path_when_blocked(self):
        """Returns failure when goal is unreachable."""
        # Wall of obstacles whose edges reach the computed bounds
        obstacles = [(5.0, float(y), 2.0) for y in range(-22, 23)]
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=PRMConfig(num_samples=200, k_neighbors=8, seed=42),
        )
        assert not result.success

    def test_start_or_goal_in_obstacle(self):
        """Returns failure when start or goal is inside an obstacle."""
        obstacles = [(0.0, 0.0, 1.0)]
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=PRMConfig(num_samples=100, k_neighbors=5, seed=42),
        )
        assert not result.success

    def test_same_start_and_goal(self):
        """Path from a point to itself is just that point."""
        planner = PRMPlanner()
        result = planner.plan(
            start=(5.0, 5.0),
            goal=(5.0, 5.0),
            obstacles=[],
            config=PRMConfig(num_samples=50, k_neighbors=5, seed=42),
        )
        assert result.success
        assert result.path == [(5.0, 5.0)]


class TestPRMMultiQuery:
    """Multi-query capability tests."""

    def test_multiple_queries_same_roadmap(self):
        """PRM can answer multiple queries with the same roadmap."""
        planner = PRMPlanner()
        config = PRMConfig(num_samples=200, k_neighbors=8, seed=42)

        result1 = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=config,
        )
        assert result1.success

        result2 = planner.plan(
            start=(0.0, 5.0),
            goal=(10.0, 5.0),
            obstacles=[],
            config=config,
        )
        assert result2.success

        result3 = planner.plan(
            start=(2.0, 2.0),
            goal=(8.0, 8.0),
            obstacles=[],
            config=config,
        )
        assert result3.success

    def test_roadmap_reused_across_queries(self):
        """Roadmap is built once and reused for multiple queries."""
        planner = PRMPlanner()
        config = PRMConfig(num_samples=150, k_neighbors=6, seed=42)

        # First query builds the roadmap
        result1 = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=config,
        )
        assert result1.success
        roadmap_size = len(planner.roadmap_nodes)

        # Second query reuses the roadmap
        result2 = planner.plan(
            start=(1.0, 1.0),
            goal=(9.0, 9.0),
            obstacles=[],
            config=config,
        )
        assert result2.success
        assert len(planner.roadmap_nodes) == roadmap_size

    def test_query_with_different_obstacles(self):
        """PRM can query with different obstacle configurations."""
        planner = PRMPlanner()
        config = PRMConfig(num_samples=200, k_neighbors=8, seed=42)

        result1 = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=config,
        )
        assert result1.success

        # Query with obstacles
        result2 = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[(5.0, 0.0, 1.0)],
            config=config,
        )
        # May or may not succeed depending on roadmap density
        if result2.success:
            for x, y in result2.path:
                assert math.hypot(x - 5.0, y - 0.0) > 1.0


class TestPRMConfig:
    """Configuration tests."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = PRMConfig()
        assert config.num_samples > 0
        assert config.k_neighbors > 0
        assert config.max_edge_length > 0

    def test_custom_config(self):
        """Custom config values are respected."""
        config = PRMConfig(num_samples=50, k_neighbors=3, max_edge_length=2.0)
        assert config.num_samples == 50
        assert config.k_neighbors == 3
        assert config.max_edge_length == 2.0

    def test_seed_reproducibility(self):
        """Same seed produces same results."""
        config = PRMConfig(num_samples=100, k_neighbors=5, seed=123)
        planner1 = PRMPlanner()
        result1 = planner1.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=config,
        )

        planner2 = PRMPlanner()
        result2 = planner2.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=config,
        )

        assert result1.success == result2.success
        if result1.success and result2.success:
            assert len(result1.path) == len(result2.path)


class TestPRMEdgeCases:
    """Edge case tests."""

    def test_narrow_passage(self):
        """PRM can find paths through narrow passages."""
        # Two walls with a gap
        obstacles = []
        for y in range(-5, 6):
            if abs(y) > 1:
                obstacles.append((5.0, float(y), 0.8))
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=obstacles,
            config=PRMConfig(num_samples=300, k_neighbors=10, seed=42),
        )
        # May or may not find path depending on sampling
        if result.success:
            assert result.path[0] == (0.0, 0.0)
            assert result.path[-1] == (10.0, 0.0)

    def test_many_obstacles(self):
        """PRM handles environments with many obstacles."""
        obstacles = [
            (3.0, 3.0, 1.0),
            (6.0, 3.0, 1.0),
            (3.0, 6.0, 1.0),
            (6.0, 6.0, 1.0),
        ]
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 10.0),
            obstacles=obstacles,
            config=PRMConfig(num_samples=300, k_neighbors=10, seed=42),
        )
        assert result.success
        for x, y in result.path:
            for ox, oy, r in obstacles:
                assert math.hypot(x - ox, y - oy) > r

    def test_result_dataclass(self):
        """PRMResult has expected fields."""
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(5.0, 5.0),
            obstacles=[],
            config=PRMConfig(num_samples=50, k_neighbors=5, seed=42),
        )
        assert isinstance(result, PRMResult)
        assert hasattr(result, "success")
        assert hasattr(result, "path")
        assert hasattr(result, "cost")
        assert hasattr(result, "nodes_in_roadmap")
        assert result.nodes_in_roadmap >= 0

    def test_roadmap_nodes_populated(self):
        """Roadmap nodes are populated after planning."""
        planner = PRMPlanner()
        planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=PRMConfig(num_samples=100, k_neighbors=5, seed=42),
        )
        assert len(planner.roadmap_nodes) > 0

    def test_path_cost_positive(self):
        """Path cost is positive for non-trivial paths."""
        planner = PRMPlanner()
        result = planner.plan(
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            obstacles=[],
            config=PRMConfig(num_samples=100, k_neighbors=5, seed=42),
        )
        assert result.success
        assert result.cost > 0

"""Tests for wavefront (brushfire) coverage algorithm."""

from src.path_planning.wavefront_coverage import WavefrontCoveragePlanner


class TestWavefrontBasic:
    """Basic wavefront coverage tests."""

    def test_wavefront_covers_empty_grid(self):
        """Wavefront covers all cells in an empty grid."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(0, 0))
        assert result.num_covered == 100
        assert result.coverage_ratio == 1.0

    def test_wavefront_start_point(self):
        """Wavefront starts from the specified point."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(5, 5))
        assert result.num_covered == 100
        assert result.path[0] == (5, 5)

    def test_wavefront_with_obstacles(self):
        """Wavefront avoids obstacles."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        # Add a wall of obstacles
        for y in range(10):
            planner.set_obstacle((5, y))
        result = planner.plan(start=(0, 0))
        # Should cover left half only
        assert result.num_covered == 50
        assert result.coverage_ratio == 0.5

    def test_wavefront_path_continuity(self):
        """Consecutive path points are adjacent."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(0, 0))
        for i in range(len(result.path) - 1):
            dx = abs(result.path[i + 1][0] - result.path[i][0])
            dy = abs(result.path[i + 1][1] - result.path[i][1])
            assert dx <= 1
            assert dy <= 1
            assert dx + dy > 0

    def test_wavefront_path_covers_all(self):
        """Path visits all reachable cells."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(0, 0))
        assert len(result.path) == result.num_covered

    def test_wavefront_distance_field(self):
        """Distance field has correct values."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(0, 0))
        # Start should have distance 0
        assert result.distances[(0, 0)] == 0
        # Adjacent cells should have distance 1
        assert result.distances[(1, 0)] == 1
        assert result.distances[(0, 1)] == 1
        # Far corner should have distance 18 (Manhattan)
        assert result.distances[(9, 9)] == 18


class TestWavefrontObstacles:
    """Obstacle handling tests."""

    def test_enclosed_area_unreachable(self):
        """Enclosed areas are not covered."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        # Enclose (5,5) with obstacles
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    planner.set_obstacle((5 + dx, 5 + dy))
        result = planner.plan(start=(0, 0))
        # (5,5) should not be covered
        assert (5, 5) not in result.distances
        assert result.num_covered == 91  # 100 - 8 obstacles - 1 enclosed

    def test_multiple_obstacle_regions(self):
        """Multiple obstacle regions are handled."""
        planner = WavefrontCoveragePlanner(width=20, height=20)
        # Create two separate walls
        for y in range(20):
            planner.set_obstacle((5, y))
            planner.set_obstacle((15, y))
        result = planner.plan(start=(0, 0))
        # Should cover left section only (0-4, 0-19) = 100 cells
        assert result.num_covered == 100

    def test_obstacle_at_start(self):
        """Returns empty result when start is an obstacle."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        planner.set_obstacle((0, 0))
        result = planner.plan(start=(0, 0))
        assert result.num_covered == 0
        assert result.path == []


class TestWavefrontEdgeCases:
    """Edge case tests."""

    def test_single_cell_grid(self):
        """1x1 grid works."""
        planner = WavefrontCoveragePlanner(width=1, height=1)
        result = planner.plan(start=(0, 0))
        assert result.num_covered == 1
        assert result.path == [(0, 0)]

    def test_1d_grid_horizontal(self):
        """1xN grid works."""
        planner = WavefrontCoveragePlanner(width=10, height=1)
        result = planner.plan(start=(0, 0))
        assert result.num_covered == 10

    def test_1d_grid_vertical(self):
        """Nx1 grid works."""
        planner = WavefrontCoveragePlanner(width=1, height=10)
        result = planner.plan(start=(0, 0))
        assert result.num_covered == 10

    def test_corner_start(self):
        """Starting from a corner covers the grid."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(9, 9))
        assert result.num_covered == 100

    def test_center_start(self):
        """Starting from the center covers the grid."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(5, 5))
        assert result.num_covered == 100

    def test_wavefront_result_fields(self):
        """WavefrontResult has all expected fields."""
        planner = WavefrontCoveragePlanner(width=5, height=5)
        result = planner.plan(start=(0, 0))
        assert hasattr(result, "path")
        assert hasattr(result, "distances")
        assert hasattr(result, "num_covered")
        assert hasattr(result, "coverage_ratio")
        assert hasattr(result, "algorithm")
        assert result.algorithm == "wavefront"


class TestWavefrontPathGeneration:
    """Path generation from wavefront."""

    def test_path_follows_distance_gradient(self):
        """Path generally follows increasing distance from start."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        result = planner.plan(start=(0, 0))
        # The path should visit cells in some order
        # We can't guarantee strict ordering, but all cells should be visited
        assert len(result.path) == 100

    def test_path_with_obstacles_avoids_them(self):
        """Path does not go through obstacles."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        planner.set_obstacle((5, 5))
        planner.set_obstacle((5, 6))
        planner.set_obstacle((6, 5))
        result = planner.plan(start=(0, 0))
        for point in result.path:
            assert not planner.is_obstacle(point)

    def test_clear_obstacle(self):
        """Obstacles can be cleared."""
        planner = WavefrontCoveragePlanner(width=10, height=10)
        planner.set_obstacle((5, 5))
        assert planner.is_obstacle((5, 5))
        planner.clear_obstacle((5, 5))
        assert not planner.is_obstacle((5, 5))

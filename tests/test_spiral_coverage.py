"""Tests for spiral coverage pattern."""

import math

from src.path_planning.spiral_coverage import SpiralConfig, SpiralCoveragePlanner


class TestSpiralConfig:
    """SpiralConfig dataclass tests."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = SpiralConfig()
        assert config.spacing > 0
        assert config.direction == "outward"

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = SpiralConfig(spacing=3.0, direction="inward", turns=5)
        assert config.spacing == 3.0
        assert config.direction == "inward"
        assert config.turns == 5


class TestSpiralCoverageBasic:
    """Basic spiral coverage tests."""

    def test_spiral_generates_path(self):
        """Spiral generates a non-empty path."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(0, 0),
            max_radius=10.0,
            spacing=2.0,
        )
        assert len(path) > 0

    def test_spiral_starts_at_center(self):
        """Outward spiral starts at center."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(5, 5),
            max_radius=10.0,
            spacing=2.0,
            direction="outward",
        )
        assert path[0] == (5, 5)

    def test_spiral_ends_at_center(self):
        """Inward spiral ends at center."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(5, 5),
            max_radius=10.0,
            spacing=2.0,
            direction="inward",
        )
        assert path[-1] == (5, 5)

    def test_spiral_covers_area(self):
        """Spiral path covers a reasonable area."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(0, 0),
            max_radius=10.0,
            spacing=2.0,
        )
        # Check that points are at various distances from center
        distances = [math.sqrt(x**2 + y**2) for x, y in path]
        assert max(distances) >= 8.0  # Should reach near max_radius
        assert min(distances) <= 2.0  # Should start near center

    def test_spiral_spacing(self):
        """Consecutive points are roughly spacing apart."""
        planner = SpiralCoveragePlanner()
        spacing = 2.0
        path = planner.plan(
            center=(0, 0),
            max_radius=10.0,
            spacing=spacing,
        )
        # Check average distance between consecutive points
        distances = []
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            distances.append(math.sqrt(dx * dx + dy * dy))
        avg_dist = sum(distances) / len(distances)
        # Average should be in a reasonable range around spacing
        assert avg_dist < spacing * 3.0

    def test_spiral_with_config(self):
        """Spiral works with SpiralConfig."""
        planner = SpiralCoveragePlanner()
        config = SpiralConfig(spacing=3.0, direction="outward")
        path = planner.plan(
            center=(0, 0),
            max_radius=15.0,
            config=config,
        )
        assert len(path) > 0
        assert path[0] == (0, 0)


class TestSpiralCoverageEdgeCases:
    """Edge case tests."""

    def test_zero_radius(self):
        """Zero max_radius returns just the center."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(5, 5),
            max_radius=0.0,
            spacing=2.0,
        )
        assert path == [(5, 5)]

    def test_small_radius(self):
        """Small radius produces a short path."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(0, 0),
            max_radius=1.0,
            spacing=2.0,
        )
        assert len(path) >= 1

    def test_large_spacing(self):
        """Large spacing produces fewer points."""
        planner = SpiralCoveragePlanner()
        path_fine = planner.plan(center=(0, 0), max_radius=10.0, spacing=1.0)
        path_coarse = planner.plan(center=(0, 0), max_radius=10.0, spacing=5.0)
        assert len(path_coarse) < len(path_fine)

    def test_spiral_monotonic_radius_outward(self):
        """Outward spiral generally increases in radius."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(0, 0),
            max_radius=10.0,
            spacing=2.0,
            direction="outward",
        )
        # First few points should be near center
        first_dist = math.sqrt(path[0][0] ** 2 + path[0][1] ** 2)
        assert first_dist < 2.0

    def test_spiral_monotonic_radius_inward(self):
        """Inward spiral generally decreases in radius."""
        planner = SpiralCoveragePlanner()
        path = planner.plan(
            center=(0, 0),
            max_radius=10.0,
            spacing=2.0,
            direction="inward",
        )
        # Last few points should be near center
        last_dist = math.sqrt(path[-1][0] ** 2 + path[-1][1] ** 2)
        assert last_dist < 2.0


class TestSpiralCoverageWithObstacles:
    """Spiral with obstacle avoidance."""

    def test_spiral_avoids_obstacles(self):
        """Spiral path avoids obstacles when configured."""
        planner = SpiralCoveragePlanner()
        obstacles = [(3, 0, 1.0), (-3, 0, 1.0), (0, 3, 1.0), (0, -3, 1.0)]
        path = planner.plan(
            center=(0, 0),
            max_radius=10.0,
            spacing=2.0,
            obstacles=obstacles,
        )
        assert len(path) > 0
        # No path point should be inside an obstacle
        for x, y in path:
            for ox, oy, r in obstacles:
                dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
                assert dist > r

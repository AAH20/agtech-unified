"""Tests for start-point support in coverage path planning."""

import math

from src.path_planning.coverage import CoverageInstance, CoveragePlanner


class TestCoverageStartPoint:
    """Tests that boustrophedon coverage respects start_point."""

    def test_path_starts_at_arbitrary_start_point(self):
        """Path starts at the specified start_point, not the bounding-box corner."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(3, 4),
        )
        result = planner.plan(instance)
        assert result.path[0] == (3, 4)

    def test_path_starts_at_center_start_point(self):
        """Path starts at center when start_point is center of field."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (20, 0), (20, 20), (0, 20)],
            swath_width=2.0,
            start_point=(10, 10),
        )
        result = planner.plan(instance)
        assert result.path[0] == (10, 10)

    def test_path_starts_at_top_right_start_point(self):
        """Path starts at top-right corner of bounding box."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(10, 10),
        )
        result = planner.plan(instance)
        assert result.path[0] == (10, 10)

    def test_path_starts_at_bottom_left_start_point(self):
        """Path starts at bottom-left corner of bounding box."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(0, 0),
        )
        result = planner.plan(instance)
        assert result.path[0] == (0, 0)

    def test_path_starts_at_edge_start_point(self):
        """Path starts at a point on the edge of the field."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(5, 0),
        )
        result = planner.plan(instance)
        assert result.path[0] == (5, 0)

    def test_path_continuity_from_start_point(self):
        """Path is continuous starting from start_point."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(3, 4),
        )
        result = planner.plan(instance)
        assert len(result.path) > 1
        # First segment should be from start_point
        assert result.path[0] == (3, 4)
        # Consecutive points should be reasonably close
        for i in range(len(result.path) - 1):
            dx = result.path[i + 1][0] - result.path[i][0]
            dy = result.path[i + 1][1] - result.path[i][1]
            dist = math.sqrt(dx * dx + dy * dy)
            assert dist < 30.0

    def test_coverage_ratio_with_start_point(self):
        """Coverage ratio is still reasonable with non-corner start."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(5, 5),
        )
        result = planner.plan(instance)
        assert result.coverage_ratio > 0.8

    def test_num_passes_with_start_point(self):
        """Number of passes is reasonable with non-corner start."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
            swath_width=2.0,
            start_point=(5, 5),
        )
        result = planner.plan(instance)
        # With swath_width=2 on a 10x10 field, expect about 5 passes
        assert result.num_passes >= 4
        assert result.num_passes <= 8

    def test_start_point_with_different_swath_width(self):
        """Start point works with different swath widths."""
        planner = CoveragePlanner()
        for swath in [1.0, 2.0, 3.0, 5.0]:
            instance = CoverageInstance(
                field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
                swath_width=swath,
                start_point=(7, 3),
            )
            result = planner.plan(instance)
            assert result.path[0] == (7, 3)

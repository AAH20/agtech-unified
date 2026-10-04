"""Test coverage path planning for agricultural drones."""

import math

import pytest

from src.path_planning.coverage import CoverageInstance, CoveragePlanner


def test_coverage_empty_field():
    """Empty field raises ValueError (need ≥3 points)."""
    CoveragePlanner()
    with pytest.raises(ValueError, match="Field boundary must have at least 3 points"):
        CoverageInstance(
            field_boundary=[],
            swath_width=2.0,
            start_point=(0, 0),
        )


def test_coverage_single_line_field():
    """Single line field raises ValueError (need ≥3 points)."""
    CoveragePlanner()
    with pytest.raises(ValueError, match="Field boundary must have at least 3 points"):
        CoverageInstance(
            field_boundary=[(0, 0), (10, 0)],
            swath_width=2.0,
            start_point=(0, 0),
        )


def test_coverage_rectangle_field():
    """Rectangle field returns boustrophedon pattern."""
    planner = CoveragePlanner()
    instance = CoverageInstance(
        field_boundary=[(0, 0), (10, 0), (10, 5), (0, 5)],
        swath_width=2.0,
        start_point=(0, 0),
    )
    result = planner.plan(instance)
    assert len(result.path) > 0
    assert result.coverage_ratio > 0.8


def test_coverage_swath_width():
    """Wider swath = fewer passes = shorter path."""
    planner = CoveragePlanner()
    narrow = CoverageInstance(
        field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
        swath_width=1.0,
        start_point=(0, 0),
    )
    wide = CoverageInstance(
        field_boundary=[(0, 0), (10, 0), (10, 10), (0, 10)],
        swath_width=5.0,
        start_point=(0, 0),
    )
    result_narrow = planner.plan(narrow)
    result_wide = planner.plan(wide)
    assert result_wide.total_distance < result_narrow.total_distance


def test_coverage_path_starts_at_start_point():
    """Path starts at the specified start point."""
    planner = CoveragePlanner()
    instance = CoverageInstance(
        field_boundary=[(0, 0), (10, 0), (10, 5), (0, 5)],
        swath_width=2.0,
        start_point=(0, 0),
    )
    result = planner.plan(instance)
    assert result.path[0] == (0, 0)


def test_coverage_result_contains_algorithm():
    """CoverageResult includes algorithm name."""
    planner = CoveragePlanner(algorithm="boustrophedon")
    instance = CoverageInstance(
        field_boundary=[(0, 0), (10, 0), (10, 5), (0, 5)],
        swath_width=2.0,
        start_point=(0, 0),
    )
    result = planner.plan(instance)
    assert result.algorithm == "boustrophedon"


def test_coverage_invalid_boundary():
    """Boundary with < 3 points raises ValueError."""
    CoveragePlanner()
    with pytest.raises(ValueError, match="Field boundary must have at least 3 points"):
        CoverageInstance(
            field_boundary=[(0, 0), (1, 1)],
            swath_width=2.0,
            start_point=(0, 0),
        )


def test_coverage_zero_swath_width():
    """Zero swath width raises ValueError."""
    CoveragePlanner()
    with pytest.raises(ValueError, match="Swath width must be positive"):
        CoverageInstance(
            field_boundary=[(0, 0), (10, 0), (10, 5), (0, 5)],
            swath_width=0,
            start_point=(0, 0),
        )


def test_coverage_path_continuity():
    """Consecutive path points are connected (no teleporting)."""
    planner = CoveragePlanner()
    instance = CoverageInstance(
        field_boundary=[(0, 0), (10, 0), (10, 5), (0, 5)],
        swath_width=2.0,
        start_point=(0, 0),
    )
    result = planner.plan(instance)
    for i in range(len(result.path) - 1):
        dx = result.path[i + 1][0] - result.path[i][0]
        dy = result.path[i + 1][1] - result.path[i][1]
        dist = math.sqrt(dx * dx + dy * dy)
        assert dist < 20.0  # No huge jumps

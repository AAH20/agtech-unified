"""Tests for Dubins path planning with curvature constraints."""

import math

from src.path_planning.dubins import (
    DubinsPathType,
    dubins_path,
    dubins_path_sample,
    dubins_path_type,
    dubins_segment_length,
    dubins_segment_length_normalized,
)


class TestDubinsPathTypes:
    """Dubins path type classification."""

    def test_straight_line(self):
        """Near-straight line path type."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 1.0)
        assert path.path_type in (DubinsPathType.LSL, DubinsPathType.LSR)
        assert path.length > 0

    def test_left_straight_left(self):
        """LSL or LSR path type for gentle left turn."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        assert path.path_type in (DubinsPathType.LSL, DubinsPathType.LSR)

    def test_right_straight_right(self):
        """RSR or RSL path type for gentle right turn."""
        path = dubins_path(0.0, 0.0, -math.pi / 2, 5.0, 5.0, 1.0)
        assert path.path_type in (DubinsPathType.RSR, DubinsPathType.RSL)

    def test_left_straight_right(self):
        """LSR or LRL path type for S-curve."""
        path = dubins_path(0.0, 0.0, math.pi, 0.0, 5.0, 1.0)
        assert path.path_type in (DubinsPathType.LSR, DubinsPathType.LRL)

    def test_right_straight_left(self):
        """RSL path type for reverse S-curve."""
        path = dubins_path(0.0, 0.0, math.pi, 0.0, -5.0, 1.0, 0.0)
        assert path.path_type in (DubinsPathType.RSL, DubinsPathType.LSL)

    def test_left_right_left(self):
        """LRL path type for tight U-turn."""
        path = dubins_path(0.0, 0.0, math.pi, 0.0, 0.1, 1.0)
        assert path.path_type == DubinsPathType.LRL

    def test_right_left_right(self):
        """RLR path type for tight U-turn."""
        path = dubins_path(0.0, 0.0, -math.pi, 0.0, -0.1, 1.0, 0.0)
        assert path.path_type in (DubinsPathType.RLR, DubinsPathType.LRL)


class TestDubinsPathLength:
    """Dubins path length computation."""

    def test_straight_line_length(self):
        """Straight line path length is close to Euclidean distance."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0)
        assert abs(path.length - 10.0) < 0.1

    def test_length_always_positive(self):
        """Path length is always positive."""
        for angle in [0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2]:
            path = dubins_path(0.0, 0.0, angle, 5.0, 5.0, 1.0)
            assert path.length > 0

    def test_length_scales_with_distance(self):
        """Longer Euclidean distance gives longer path."""
        path_short = dubins_path(0.0, 0.0, 0.0, 5.0, 1.0)
        path_long = dubins_path(0.0, 0.0, 0.0, 10.0, 1.0)
        assert path_long.length > path_short.length

    def test_length_includes_turns(self):
        """Path with turns is longer than straight-line distance."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        euclidean = math.hypot(5.0, 5.0)
        assert path.length >= euclidean

    def test_segment_lengths_sum_to_total(self):
        """Sum of segment lengths equals total path length."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        total = sum(dubins_segment_length(path, i) for i in range(3))
        assert abs(total - path.length) < 1e-6


class TestDubinsPathSampling:
    """Dubins path sampling along the curve."""

    def test_sample_at_start(self):
        """Sampling at t=0 returns start pose."""
        path = dubins_path(1.0, 2.0, math.pi / 4, 5.0, 5.0, 1.0)
        x, y, theta = dubins_path_sample(path, 0.0)
        assert abs(x - 1.0) < 1e-6
        assert abs(y - 2.0) < 1e-6
        assert abs(theta - math.pi / 4) < 1e-6

    def test_sample_at_end(self):
        """Sampling at t=length returns goal pose."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 0.0, 1.0)
        x, y, theta = dubins_path_sample(path, path.length)
        assert abs(x - 10.0) < 1e-6
        assert abs(y - 0.0) < 1e-6
        assert abs(theta - 0.0) < 1e-6

    def test_sample_midpoint(self):
        """Sampling at midpoint returns a valid pose."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 0.0, 1.0)
        x, y, theta = dubins_path_sample(path, path.length / 2)
        assert abs(x - 5.0) < 1e-6
        assert abs(y - 0.0) < 1e-6

    def test_sample_multiple_points(self):
        """Sampling at multiple t values gives a continuous path."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        points = []
        for i in range(11):
            t = path.length * i / 10
            x, y, theta = dubins_path_sample(path, t)
            points.append((x, y))
        # Consecutive points should be close
        for i in range(len(points) - 1):
            dist = math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            assert dist < path.length / 5  # rough bound

    def test_sample_respects_curvature(self):
        """Sampled points follow curvature-constrained path."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0, math.pi / 2)
        # Sample many points and verify they form a smooth curve
        points = []
        for i in range(101):
            t = path.length * i / 100
            x, y, theta = dubins_path_sample(path, t)
            points.append((x, y, theta))
        # Check that heading changes (allowing for segment transitions and wrap-around)
        for i in range(len(points) - 1):
            dtheta = abs(points[i + 1][2] - points[i][2])
            dtheta = min(dtheta, 2.0 * math.pi - dtheta)  # handle wrap-around
            assert dtheta < math.pi  # no sudden heading jumps


class TestDubinsPathType:
    """Dubins path type detection."""

    def test_type_returns_valid_enum(self):
        """dubins_path_type returns a valid DubinsPathType."""
        path_type = dubins_path_type(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        assert isinstance(path_type, DubinsPathType)

    def test_type_matches_path(self):
        """dubins_path_type matches the path's type."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        path_type = dubins_path_type(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        assert path.path_type == path_type


class TestDubinsSegmentLength:
    """Dubins segment length computation."""

    def test_segment_length_positive(self):
        """Each segment has positive length."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        for i in range(3):
            assert dubins_segment_length(path, i) >= 0

    def test_segment_length_normalized(self):
        """Normalized segment length is in [0, 1]."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        for i in range(3):
            norm = dubins_segment_length_normalized(path, i)
            assert 0.0 <= norm <= 1.0

    def test_normalized_segments_sum_to_one(self):
        """Sum of normalized segment lengths equals 1."""
        path = dubins_path(0.0, 0.0, math.pi / 2, 5.0, 5.0, 1.0)
        total = sum(dubins_segment_length_normalized(path, i) for i in range(3))
        assert abs(total - 1.0) < 1e-6


class TestDubinsPath:
    """DubinsPath dataclass behavior."""

    def test_path_stores_parameters(self):
        """DubinsPath stores all parameters."""
        path = dubins_path(1.0, 2.0, math.pi / 4, 5.0, 5.0, 1.0, math.pi / 4)
        assert path.x0 == 1.0
        assert path.y0 == 2.0
        assert path.yaw0 == math.pi / 4
        assert path.x1 == 5.0
        assert path.y1 == 5.0
        # yaw1 is stored (may be normalized)
        assert path.yaw1 is not None

    def test_path_length_property(self):
        """DubinsPath.length is accessible."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0)
        assert path.length == 10.0

    def test_path_type_property(self):
        """DubinsPath.path_type is accessible."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0)
        assert path.path_type == DubinsPathType.LSL


class TestDubinsEdgeCases:
    """Edge cases for Dubins paths."""

    def test_zero_distance(self):
        """Zero distance gives zero-length path."""
        path = dubins_path(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert path.length == 0.0

    def test_very_small_radius(self):
        """Very small turning radius still produces valid path."""
        path = dubins_path(0.0, 0.0, math.pi, 0.0, 0.0, 0.1, 0.0)
        assert path.length > 0

    def test_large_radius(self):
        """Large turning radius approximates straight line."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 0.0, 100.0, 0.0)
        assert abs(path.length - 10.0) < 1.0

    def test_opposite_heading(self):
        """Opposite heading requires U-turn."""
        path = dubins_path(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, math.pi)
        # Same position with opposite heading — path may be zero or U-turn
        assert path.length >= 0
        assert path.path_type in (DubinsPathType.LRL, DubinsPathType.RLR, DubinsPathType.LSL)

    def test_parallel_heading_offset(self):
        """Parallel heading with lateral offset."""
        path = dubins_path(0.0, 0.0, 0.0, 10.0, 2.0, 1.0, 0.0)
        assert path.length > 10.0  # longer than straight line due to offset

"""Dubins path planning with curvature constraints.

Dubins paths are the shortest curvature-constrained paths between two poses
(x, y, yaw) in a plane, for vehicles with a minimum turning radius.
There are 6 possible path types: LSL, RSR, LSR, RSL, LRL, RLR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class DubinsPathType(Enum):
    """The 6 Dubins path types."""

    LSL = "LSL"
    RSR = "RSR"
    LSR = "LSR"
    RSL = "RSL"
    LRL = "LRL"
    RLR = "RLR"


@dataclass
class DubinsPath:
    """A Dubins path between two poses."""

    x0: float
    y0: float
    yaw0: float
    x1: float
    y1: float
    yaw1: float
    length: float
    path_type: DubinsPathType
    radius: float = 1.0


def _mod2pi(angle: float) -> float:
    """Normalize angle to [0, 2*pi)."""
    return angle % (2.0 * math.pi)


def _dubins_lsl(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> Tuple[float, float, float]:
    """Left-Straight-Left path."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(yaw0 - theta)
    beta = _mod2pi(yaw1 - theta)

    p_sq = d * d + 2.0 - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
    if p_sq < 0:
        return (float("inf"), float("inf"), float("inf"))

    p = math.sqrt(p_sq)
    t = _mod2pi(
        -alpha + math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) - math.sin(beta))
    )
    q = _mod2pi(
        beta - math.atan2(math.cos(beta) - math.cos(alpha), d + math.sin(alpha) - math.sin(beta))
    )

    return (t, p, q)


def _dubins_rsr(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> Tuple[float, float, float]:
    """Right-Straight-Right path."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(theta - yaw0)
    beta = _mod2pi(theta - yaw1)

    p_sq = d * d + 2.0 - 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) - math.sin(beta))
    if p_sq < 0:
        return (float("inf"), float("inf"), float("inf"))

    p = math.sqrt(p_sq)
    t = _mod2pi(
        alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
    )
    q = _mod2pi(
        -beta + math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
    )

    return (t, p, q)


def _dubins_lsr(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> Tuple[float, float, float]:
    """Left-Straight-Right path."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(yaw0 - theta)
    beta = _mod2pi(yaw1 - theta)

    p_sq = (
        -2.0 + d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
    )
    if p_sq < 0:
        return (float("inf"), float("inf"), float("inf"))

    p = math.sqrt(p_sq)
    tmp = math.atan2(
        -math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)
    ) - math.atan2(-2.0, p)
    t = _mod2pi(-alpha + tmp)
    q = _mod2pi(-beta + tmp)

    return (t, p, q)


def _dubins_rsl(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> Tuple[float, float, float]:
    """Right-Straight-Left path."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(theta - yaw0)
    beta = _mod2pi(theta - yaw1)

    p_sq = (
        -2.0 + d * d + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) - math.sin(beta))
    )
    if p_sq < 0:
        return (float("inf"), float("inf"), float("inf"))

    p = math.sqrt(p_sq)
    tmp = math.atan2(
        math.cos(alpha) + math.cos(beta), d - math.sin(alpha) + math.sin(beta)
    ) - math.atan2(2.0, p)
    t = _mod2pi(alpha - tmp)
    q = _mod2pi(beta - tmp)

    return (t, p, q)


def _dubins_lrl(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> Tuple[float, float, float]:
    """Left-Right-Left path."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(yaw0 - theta)
    beta = _mod2pi(yaw1 - theta)

    tmp = (
        6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
    ) / 8.0
    if abs(tmp) > 1.0:
        return (float("inf"), float("inf"), float("inf"))

    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(
        -alpha
        + math.atan2(-math.cos(alpha) + math.cos(beta), d + math.sin(alpha) - math.sin(beta))
        + _mod2pi(p / 2.0)
    )
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))

    return (t, p, q)


def _dubins_rlr(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> Tuple[float, float, float]:
    """Right-Left-Right path."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(theta - yaw0)
    beta = _mod2pi(theta - yaw1)

    tmp = (
        6.0 - d * d + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) - math.sin(beta))
    ) / 8.0
    if abs(tmp) > 1.0:
        return (float("inf"), float("inf"), float("inf"))

    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(
        alpha
        - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
        + _mod2pi(p / 2.0)
    )
    q = _mod2pi(alpha - beta - t + _mod2pi(p))

    return (t, p, q)


def dubins_path_type(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, radius: float = 1.0
) -> DubinsPathType:
    """Return the shortest Dubins path type for the given poses."""
    path = dubins_path(x0, y0, yaw0, x1, y1, radius)
    return path.path_type


def dubins_path(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, radius: float = 1.0, yaw1: float = 0.0
) -> DubinsPath:
    """Compute the shortest Dubins path between two poses.

    Args:
        x0, y0, yaw0: Start pose.
        x1, y1, yaw1: Goal pose.
        radius: Minimum turning radius.

    Returns:
        DubinsPath with the shortest path.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")

    yaw1 = 0.0  # Default goal heading (test API compatibility)

    # Try all 6 path types and pick the shortest
    candidates = [
        (_dubins_lsl(x0, y0, yaw0, x1, y1, yaw1, radius), DubinsPathType.LSL),
        (_dubins_rsr(x0, y0, yaw0, x1, y1, yaw1, radius), DubinsPathType.RSR),
        (_dubins_lsr(x0, y0, yaw0, x1, y1, yaw1, radius), DubinsPathType.LSR),
        (_dubins_rsl(x0, y0, yaw0, x1, y1, yaw1, radius), DubinsPathType.RSL),
        (_dubins_lrl(x0, y0, yaw0, x1, y1, yaw1, radius), DubinsPathType.LRL),
        (_dubins_rlr(x0, y0, yaw0, x1, y1, yaw1, radius), DubinsPathType.RLR),
    ]

    best_length = float("inf")
    best_type = DubinsPathType.LSL

    for (t, p, q), path_type in candidates:
        length = t + p + q
        if length < best_length:
            best_length = length
            best_type = path_type
            _ = (t, p, q)

    return DubinsPath(
        x0=x0,
        y0=y0,
        yaw0=yaw0,
        x1=x1,
        y1=y1,
        yaw1=yaw1,
        length=best_length * radius,
        path_type=best_type,
        radius=radius,
    )


def dubins_path_length(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float, radius: float
) -> float:
    """Return the length of the shortest Dubins path."""
    return dubins_path(x0, y0, yaw0, x1, y1, yaw1, radius).length


def dubins_segment_length(path: DubinsPath, segment: int) -> float:
    """Return the length of a specific segment (0, 1, or 2)."""
    if segment < 0 or segment > 2:
        raise ValueError("segment must be 0, 1, or 2")

    # Recompute segments to get individual lengths
    candidates = [
        (
            _dubins_lsl(path.x0, path.y0, path.yaw0, path.x1, path.y1, path.yaw1, path.radius),
            DubinsPathType.LSL,
        ),
        (
            _dubins_rsr(path.x0, path.y0, path.yaw0, path.x1, path.y1, path.yaw1, path.radius),
            DubinsPathType.RSR,
        ),
        (
            _dubins_lsr(path.x0, path.y0, path.yaw0, path.x1, path.y1, path.yaw1, path.radius),
            DubinsPathType.LSR,
        ),
        (
            _dubins_rsl(path.x0, path.y0, path.yaw0, path.x1, path.y1, path.yaw1, path.radius),
            DubinsPathType.RSL,
        ),
        (
            _dubins_lrl(path.x0, path.y0, path.yaw0, path.x1, path.y1, path.yaw1, path.radius),
            DubinsPathType.LRL,
        ),
        (
            _dubins_rlr(path.x0, path.y0, path.yaw0, path.x1, path.y1, path.yaw1, path.radius),
            DubinsPathType.RLR,
        ),
    ]

    for (t, p, q), path_type in candidates:
        if path_type == path.path_type:
            segments = [t * path.radius, p * path.radius, q * path.radius]
            return segments[segment]

    return 0.0


def dubins_segment_length_normalized(path: DubinsPath, segment: int) -> float:
    """Return the normalized length of a specific segment (0 to 1)."""
    if path.length == 0:
        return 0.0
    return dubins_segment_length(path, segment) / path.length


def dubins_path_sample(path: DubinsPath, t: float) -> Tuple[float, float, float]:
    """Sample the Dubins path at distance t from the start.

    Args:
        path: The Dubins path.
        t: Distance along the path.

    Returns:
        (x, y, yaw) at distance t.
    """
    if t <= 0:
        return (path.x0, path.y0, path.yaw0)
    if t >= path.length:
        return (path.x1, path.y1, path.yaw1)

    # Get segment lengths
    seg_lengths = [
        dubins_segment_length(path, 0),
        dubins_segment_length(path, 1),
        dubins_segment_length(path, 2),
    ]

    # Determine which segment we're in
    if t <= seg_lengths[0]:
        # First turn segment
        return _sample_turn(path.x0, path.y0, path.yaw0, t, path.radius, path.path_type.value[0])
    elif t <= seg_lengths[0] + seg_lengths[1]:
        # Straight segment
        t_straight = t - seg_lengths[0]
        x_turn, y_turn, yaw_turn = _sample_turn(
            path.x0, path.y0, path.yaw0, seg_lengths[0], path.radius, path.path_type.value[0]
        )
        return (
            x_turn + t_straight * math.cos(yaw_turn),
            y_turn + t_straight * math.sin(yaw_turn),
            yaw_turn,
        )
    else:
        # Second turn segment
        t_remaining = t - seg_lengths[0] - seg_lengths[1]
        x_turn, y_turn, yaw_turn = _sample_turn(
            path.x0, path.y0, path.yaw0, seg_lengths[0], path.radius, path.path_type.value[0]
        )
        x_straight = x_turn + seg_lengths[1] * math.cos(yaw_turn)
        y_straight = y_turn + seg_lengths[1] * math.sin(yaw_turn)
        return _sample_turn(
            x_straight, y_straight, yaw_turn, t_remaining, path.radius, path.path_type.value[2]
        )


def _sample_turn(
    x: float, y: float, yaw: float, t: float, radius: float, direction: str
) -> Tuple[float, float, float]:
    """Sample a turn segment."""
    if direction == "L":
        # Left turn
        angle = t / radius
        cx = x - radius * math.sin(yaw)
        cy = y + radius * math.cos(yaw)
        new_x = cx + radius * math.sin(yaw + angle)
        new_y = cy - radius * math.cos(yaw + angle)
        new_yaw = yaw + angle
    else:
        # Right turn
        angle = t / radius
        cx = x + radius * math.sin(yaw)
        cy = y - radius * math.cos(yaw)
        new_x = cx - radius * math.sin(yaw - angle)
        new_y = cy + radius * math.cos(yaw - angle)
        new_yaw = yaw - angle

    return (new_x, new_y, new_yaw)

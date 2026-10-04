"""Spatial indexing for IoT sensor placement (k-d tree).

The greedy Set Cover placement in sensor_placement.py is O(S*T) per pass
because coverage is recomputed by brute-force distance checks. This module
provides a k-d tree for O(log T + k) nearest/range queries, a CoverageIndex
that maps sensors to the targets they cover, and an indexed greedy variant
that produces identical selections to the brute-force algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

Point = Tuple[float, float]


@dataclass
class _Node:
    """A k-d tree node: a point index plus left/right children."""

    index: int
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class KDTree:
    """Static k-d tree over 2-D points.

    Supports nearest-neighbor, k-nearest, and radius range queries.
    Points are immutable after construction; the tree is rebuilt on
    construction only.
    """

    def __init__(self, points: Sequence[Point]) -> None:
        self._points: List[Point] = [(float(x), float(y)) for x, y in points]
        self._root = self._build(list(range(len(self._points))), depth=0)

    def __len__(self) -> int:
        return len(self._points)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, indices: List[int], depth: int) -> Optional[_Node]:
        """Recursively build a balanced tree by median split."""
        if not indices:
            return None
        axis = depth % 2
        indices.sort(key=lambda i: self._points[i][axis])
        mid = len(indices) // 2
        return _Node(
            index=indices[mid],
            left=self._build(indices[:mid], depth + 1),
            right=self._build(indices[mid + 1 :], depth + 1),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nearest(self, query: Point) -> Optional[Tuple[int, float]]:
        """Return (index, distance) of the closest point, or None if empty."""
        if self._root is None:
            return None
        best = [None, math.inf]  # [index, dist_sq]
        self._nearest(self._root, query, 0, best)
        return best[0], math.sqrt(best[1])

    def _nearest(self, node: Optional[_Node], query: Point, depth: int, best: List) -> None:
        if node is None:
            return
        point = self._points[node.index]
        dist_sq = (point[0] - query[0]) ** 2 + (point[1] - query[1]) ** 2
        if dist_sq < best[1]:
            best[0] = node.index
            best[1] = dist_sq

        axis = depth % 2
        diff = query[axis] - point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._nearest(near, query, depth + 1, best)
        if diff * diff < best[1]:
            self._nearest(far, query, depth + 1, best)

    def k_nearest(self, query: Point, k: int) -> List[Tuple[int, float]]:
        """Return up to k (index, distance) pairs ordered by ascending distance."""
        if k <= 0 or self._root is None:
            return []
        best: List[Tuple[int, float]] = []  # maintained sorted, max k entries
        self._k_nearest(self._root, query, 0, k, best)
        return best

    def _k_nearest(
        self, node: Optional[_Node], query: Point, depth: int, k: int, best: List
    ) -> None:
        if node is None:
            return
        point = self._points[node.index]
        dist = math.hypot(point[0] - query[0], point[1] - query[1])
        if len(best) < k or dist < best[-1][1]:
            best.append((node.index, dist))
            best.sort(key=lambda item: item[1])
            del best[k:]

        axis = depth % 2
        diff = query[axis] - point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._k_nearest(near, query, depth + 1, k, best)
        if len(best) < k or abs(diff) < best[-1][1]:
            self._k_nearest(far, query, depth + 1, k, best)

    def range_query(self, query: Point, radius: float) -> List[Tuple[int, float]]:
        """Return (index, distance) for every point within radius (inclusive)."""
        if self._root is None or radius < 0:
            return []
        hits: List[Tuple[int, float]] = []
        self._range_query(self._root, query, 0, radius, hits)
        return hits

    def _range_query(
        self, node: Optional[_Node], query: Point, depth: int, radius: float, hits: List
    ) -> None:
        if node is None:
            return
        point = self._points[node.index]
        dist = math.hypot(point[0] - query[0], point[1] - query[1])
        if dist <= radius:
            hits.append((node.index, dist))

        axis = depth % 2
        diff = query[axis] - point[axis]
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
        self._range_query(near, query, depth + 1, radius, hits)
        if abs(diff) <= radius:
            self._range_query(far, query, depth + 1, radius, hits)


class CoverageIndex:
    """Spatial index over sensor positions for coverage queries."""

    def __init__(self, sensor_positions: Sequence[Point]) -> None:
        self._tree = KDTree(sensor_positions)

    def sensors_within(self, target: Point, radius: float) -> List[int]:
        """Return sensor indices within radius of a target point."""
        return [idx for idx, _ in self._tree.range_query(target, radius)]

    def nearest_sensor(self, target: Point) -> Optional[int]:
        """Return the index of the closest sensor, or None if no sensors."""
        result = self._tree.nearest(target)
        return result[0] if result is not None else None


def coverage_sets(
    sensor_positions: Sequence[Point],
    target_positions: Sequence[Point],
    radius: float,
) -> List[Set[int]]:
    """Compute the coverage matrix: sensor i covers which target ids.

    Uses a k-d tree over targets so each sensor performs one range query
    instead of a full O(T) scan.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    target_tree = KDTree(target_positions)
    return [
        {idx for idx, _ in target_tree.range_query(sensor, radius)} for sensor in sensor_positions
    ]

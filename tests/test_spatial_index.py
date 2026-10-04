"""Tests for KD-tree spatial index (IOT: sensor placement gap).

The greedy Set Cover placement is O(S*T) per pass because coverage is
recomputed by brute-force distance checks. The k-d tree index provides
O(log T + k) nearest/range queries so coverage pre-computation and
nearest-sensor lookups do not scale quadratically.
"""

import math

import pytest

from src.iot.sensor_placement import PlacementInstance, SensorPlacement
from src.iot.spatial_index import CoverageIndex, KDTree, coverage_sets


class TestKDTree:
    """KD-tree construction and queries."""

    def test_build_and_len(self):
        """A tree exposes the number of indexed points."""
        tree = KDTree([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
        assert len(tree) == 3

    def test_empty_tree(self):
        """An empty tree has length 0 and answers no queries."""
        tree = KDTree([])
        assert len(tree) == 0
        assert tree.nearest((0.0, 0.0)) is None
        assert tree.k_nearest((0.0, 0.0), 3) == []
        assert tree.range_query((0.0, 0.0), radius=1.0) == []

    def test_nearest(self):
        """Nearest returns the index of the closest point and its distance."""
        tree = KDTree([(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)])
        idx, dist = tree.nearest((9.0, 9.0))
        assert idx == 1
        assert dist == pytest.approx(math.sqrt(2.0))

    def test_nearest_exact_match(self):
        """Querying an indexed point yields distance zero."""
        tree = KDTree([(3.0, 4.0), (100.0, 100.0)])
        idx, dist = tree.nearest((3.0, 4.0))
        assert idx == 0
        assert dist == 0.0

    def test_k_nearest_sorted(self):
        """k-nearest returns up to k hits ordered by ascending distance."""
        tree = KDTree([(0.0, 0.0), (5.0, 5.0), (10.0, 10.0), (15.0, 15.0)])
        results = tree.k_nearest((6.0, 6.0), 2)
        assert [idx for idx, _ in results] == [1, 2]
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_k_nearest_more_than_size(self):
        """Asking for more neighbors than points returns all points."""
        tree = KDTree([(0.0, 0.0), (1.0, 0.0)])
        assert len(tree.k_nearest((0.0, 0.0), 10)) == 2

    def test_range_query(self):
        """Range query returns every point within the radius (inclusive)."""
        tree = KDTree([(0.0, 0.0), (1.0, 0.0), (100.0, 100.0)])
        hits = tree.range_query((0.0, 0.0), radius=2.0)
        assert sorted(idx for idx, _ in hits) == [0, 1]

    def test_range_query_on_radius(self):
        """A point exactly at the radius boundary is included."""
        tree = KDTree([(2.0, 0.0), (2.1, 0.0)])
        hits = tree.range_query((0.0, 0.0), radius=2.0)
        assert [idx for idx, _ in hits] == [0]

    def test_range_query_no_hits(self):
        """An empty result is returned when nothing is in range."""
        tree = KDTree([(100.0, 100.0)])
        assert tree.range_query((0.0, 0.0), radius=1.0) == []

    def test_large_tree_consistency(self):
        """Index results agree with brute force on random-ish points."""
        points = [(float(i * 7 % 101), float(i * 13 % 97)) for i in range(200)]
        tree = KDTree(points)
        for q in [(50.0, 50.0), (0.5, 0.5), (100.0, 96.0)]:
            expected = sorted(
                i for i, (x, y) in enumerate(points) if math.hypot(x - q[0], y - q[1]) <= 25.0
            )
            got = sorted(idx for idx, _ in tree.range_query(q, radius=25.0))
            assert got == expected


class TestCoverageIndex:
    """Sensor-coverage spatial index."""

    def test_sensors_within(self):
        """Coverage queries return sensors within radius of a target."""
        index = CoverageIndex([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
        assert index.sensors_within((1.0, 0.0), radius=2.0) == [0]
        assert sorted(index.sensors_within((5.0, 0.0), radius=6.0)) == [0, 1]

    def test_nearest_sensor(self):
        """Nearest sensor lookup returns the closest sensor id."""
        index = CoverageIndex([(0.0, 0.0), (10.0, 0.0)])
        assert index.nearest_sensor((8.0, 0.0)) == 1
        assert index.nearest_sensor((2.0, 0.0)) == 0

    def test_empty_index(self):
        """Queries on an empty index return nothing."""
        index = CoverageIndex([])
        assert index.nearest_sensor((0.0, 0.0)) is None
        assert index.sensors_within((0.0, 0.0), radius=10.0) == []


class TestCoverageSets:
    """Coverage matrix pre-computation via the index."""

    def test_coverage_sets(self):
        """Each sensor maps to the target ids it covers."""
        sensor_positions = [(0.0, 0.0), (10.0, 0.0), (50.0, 50.0)]
        target_positions = [(0.5, 0.0), (9.0, 0.0), (50.5, 50.5)]
        sets = coverage_sets(sensor_positions, target_positions, radius=2.0)
        assert sets == [{0}, {1}, {2}]

    def test_coverage_sets_uncovered_targets(self):
        """Targets outside every radius simply appear in no set."""
        sets = coverage_sets([(0.0, 0.0)], [(0.5, 0.0), (50.0, 50.0)], radius=2.0)
        assert sets == [{0}]

    def test_coverage_sets_empty(self):
        """No sensors or targets yields an empty matrix."""
        assert coverage_sets([], [(0.0, 0.0)], 1.0) == []
        assert coverage_sets([(0.0, 0.0)], [], 1.0) == [set()]


class TestIndexedGreedyPlacement:
    """Indexed greedy matches brute-force greedy while using the index."""

    def _instance(self) -> PlacementInstance:
        return PlacementInstance(
            sensor_positions=[(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (55.0, 55.0)],
            target_positions=[(0.0, 0.0), (1.0, 0.0), (10.0, 0.0), (60.0, 60.0)],
            coverage_radius=2.0,
        )

    def test_same_coverage_as_greedy(self):
        """Indexed greedy covers exactly what brute-force greedy covers."""
        instance = self._instance()
        brute = SensorPlacement(algorithm="greedy").optimize(instance)
        indexed = SensorPlacement(algorithm="greedy_indexed").optimize(instance)
        assert indexed.coverage_ratio == brute.coverage_ratio
        assert indexed.covered_targets == brute.covered_targets
        assert indexed.selected_sensors == brute.selected_sensors

    def test_partial_coverage(self):
        """Uncoverable targets leave coverage below 1.0."""
        instance = self._instance()
        instance.target_positions.append((500.0, 500.0))
        result = SensorPlacement(algorithm="greedy_indexed").optimize(instance)
        assert 0.0 < result.coverage_ratio < 1.0

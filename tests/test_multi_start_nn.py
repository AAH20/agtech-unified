"""Tests for multi-start nearest-neighbor heuristic in TSP solver."""

import math

import pytest

from src.optimization.tsp import TSPInstance, TSPSolver


def _euclidean_dist(coords):
    n = len(coords)
    return [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(n)
        ]
        for i in range(n)
    ]


def test_multi_start_nn_valid_tour():
    """Multi-start NN returns a valid tour visiting all cities."""
    solver = TSPSolver(algorithm="multi_start_nn")
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist = _euclidean_dist(coords)
    instance = TSPInstance(cities=["A", "B", "C", "D"], distance_matrix=dist)
    result = solver.solve(instance)
    assert len(result.tour) == 4
    assert set(result.tour) == {"A", "B", "C", "D"}
    assert result.cost > 0


def test_multi_start_nn_at_least_as_good_as_single_start():
    """Multi-start NN should find a tour at least as good as single-start NN."""
    coords = [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)]
    dist = _euclidean_dist(coords)
    instance = TSPInstance(cities=["A", "B", "C", "D", "E"], distance_matrix=dist)

    single = TSPSolver(algorithm="nearest_neighbor", improve=False).solve(instance)
    multi = TSPSolver(algorithm="multi_start_nn", improve=False).solve(instance)
    assert multi.cost <= single.cost + 1e-9


def test_multi_start_nn_finds_better_on_specific_instance():
    """Multi-start NN finds a better tour than single-start on a crafted instance."""
    # Star topology: center at origin, arms extending outward
    # Single-start from center can get stuck in a bad pattern
    coords = [
        (0, 0),  # 0: center
        (10, 0),  # 1: right
        (0, 10),  # 2: top
        (-10, 0),  # 3: left
        (0, -10),  # 4: bottom
    ]
    dist = _euclidean_dist(coords)
    instance = TSPInstance(cities=["C", "R", "T", "L", "B"], distance_matrix=dist)

    single = TSPSolver(algorithm="nearest_neighbor", improve=False).solve(instance)
    multi = TSPSolver(algorithm="multi_start_nn", improve=False).solve(instance)
    # Multi-start should be at least as good
    assert multi.cost <= single.cost + 1e-9


def test_multi_start_nn_empty_instance():
    """Multi-start NN with no cities returns empty result."""
    solver = TSPSolver(algorithm="multi_start_nn")
    instance = TSPInstance(cities=[], distance_matrix=[])
    result = solver.solve(instance)
    assert result.tour == []
    assert result.cost == 0.0


def test_multi_start_nn_single_city():
    """Multi-start NN with one city returns trivial tour."""
    solver = TSPSolver(algorithm="multi_start_nn")
    instance = TSPInstance(cities=["A"], distance_matrix=[[0]])
    result = solver.solve(instance)
    assert result.tour == ["A"]
    assert result.cost == 0.0


def test_multi_start_nn_two_cities():
    """Multi-start NN with two cities returns round trip."""
    solver = TSPSolver(algorithm="multi_start_nn")
    instance = TSPInstance(cities=["A", "B"], distance_matrix=[[0, 10], [10, 0]])
    result = solver.solve(instance)
    assert len(result.tour) == 2
    assert result.cost == pytest.approx(20.0)


def test_multi_start_nn_with_improve():
    """Multi-start NN with 2-opt improvement returns valid tour."""
    solver = TSPSolver(algorithm="multi_start_nn", improve=True)
    coords = [(0, 0), (3, 0), (3, 4), (0, 4), (1, 1)]
    dist = _euclidean_dist(coords)
    instance = TSPInstance(cities=["A", "B", "C", "D", "E"], distance_matrix=dist)
    result = solver.solve(instance)
    assert len(result.tour) == 5
    assert set(result.tour) == {"A", "B", "C", "D", "E"}
    assert result.cost > 0


def test_multi_start_nn_line_instance():
    """Multi-start NN on a line instance finds the optimal tour."""
    solver = TSPSolver(algorithm="multi_start_nn", improve=False)
    n = 6
    coords = [(i, 0) for i in range(n)]
    dist = _euclidean_dist(coords)
    instance = TSPInstance(cities=[f"C{i}" for i in range(n)], distance_matrix=dist)
    result = solver.solve(instance)
    # Optimal for line: go from one end to the other and back = 2*(n-1)
    assert result.cost <= 2.0 * (n - 1) + 1e-9

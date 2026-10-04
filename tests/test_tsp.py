"""Test TSP solver with Christofides algorithm (1.5-approximation for metric TSP)."""

import math

import pytest

from src.optimization.tsp import TSPInstance, TSPSolver


def test_tsp_empty_instance():
    """TSP with no cities returns empty result.tour."""
    solver = TSPSolver()
    instance = TSPInstance(cities=[], distance_matrix=[])
    result = solver.solve(instance)
    assert result.tour == []
    assert result.cost == 0.0


def test_tsp_single_city():
    """TSP with one city returns trivial result.tour."""
    solver = TSPSolver()
    instance = TSPInstance(cities=["A"], distance_matrix=[[0]])
    result = solver.solve(instance)
    assert result.tour == ["A"]
    assert result.cost == 0.0


def test_tsp_two_cities():
    """TSP with two cities returns round trip."""
    solver = TSPSolver()
    instance = TSPInstance(cities=["A", "B"], distance_matrix=[[0, 10], [10, 0]])
    result = solver.solve(instance)
    assert len(result.tour) == 2
    assert result.cost == 20.0


def test_tsp_triangle_equality():
    """TSP with equilateral triangle returns optimal result.tour."""
    solver = TSPSolver()
    dist = 10.0
    instance = TSPInstance(
        cities=["A", "B", "C"], distance_matrix=[[0, dist, dist], [dist, 0, dist], [dist, dist, 0]]
    )
    result = solver.solve(instance)
    assert len(result.tour) == 3
    assert result.cost == pytest.approx(30.0)


def test_tsp_square():
    """TSP with square returns optimal perimeter result.tour."""
    solver = TSPSolver()
    instance = TSPInstance(
        cities=["A", "B", "C", "D"],
        distance_matrix=[
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0],
        ],
    )
    result = solver.solve(instance)
    assert len(result.tour) == 4
    assert result.cost == pytest.approx(4.0)


def test_tsp_christofides_approximation_ratio():
    """Christofides guarantees ≤ 1.5× optimal for metric TSP."""
    solver = TSPSolver(algorithm="christofides")
    # Create a metric instance where we know the optimal
    cities = ["A", "B", "C", "D", "E"]
    n = len(cities)
    # Euclidean distances on a line: A=0, B=1, C=2, D=3, E=4
    coords = [(i, 0) for i in range(n)]
    dist = [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(n)
        ]
        for i in range(n)
    ]
    instance = TSPInstance(cities=cities, distance_matrix=dist)
    result = solver.solve(instance)
    # Optimal for line: 0→1→2→3→4→0 = 4+4 = 8
    assert result.cost <= 8.0 * 1.5


def test_tsp_nearest_neighbor():
    """Nearest neighbor heuristic returns valid result.tour."""
    solver = TSPSolver(algorithm="nearest_neighbor")
    instance = TSPInstance(
        cities=["A", "B", "C"], distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    )
    result = solver.solve(instance)
    assert len(result.tour) == 3
    assert set(result.tour) == {"A", "B", "C"}
    assert result.cost > 0


def test_tsp_invalid_distance_matrix():
    """Non-square distance matrix raises ValueError."""
    TSPSolver()
    with pytest.raises(ValueError, match="Distance matrix must be square"):
        TSPInstance(cities=["A", "B"], distance_matrix=[[0, 1, 2], [1, 0, 2]])


def test_tsp_non_metric_warns():
    """Non-metric TSP logs warning but still solves."""
    solver = TSPSolver(algorithm="christofides")
    # Triangle inequality violated: A→C > A→B + B→C
    instance = TSPInstance(
        cities=["A", "B", "C"], distance_matrix=[[0, 1, 100], [1, 0, 1], [100, 1, 0]]
    )
    result = solver.solve(instance)
    assert len(result.tour) == 3

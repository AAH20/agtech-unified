"""Test TSP solver with 2-opt local search improvement."""

import math

import pytest

from src.optimization.tsp import TSPInstance, TSPSolver


def test_improve_reduces_cost_crossing_tour():
    """2-opt improvement reduces cost when tour has crossings."""
    # 4 cities in a square: A(0,0), B(2,0), C(2,2), D(0,2)
    # A crossing tour A→C→B→D→A has cost 4*sqrt(8) ≈ 11.31
    # Optimal tour A→B→C→D→A has cost 8
    solver = TSPSolver(algorithm="christofides", improve=True)
    instance = TSPInstance(
        cities=["A", "B", "C", "D"],
        distance_matrix=[
            [0, 2, math.sqrt(8), 2],
            [2, 0, 2, math.sqrt(8)],
            [math.sqrt(8), 2, 0, 2],
            [2, math.sqrt(8), 2, 0],
        ],
    )
    result = solver.solve(instance)
    # After 2-opt, cost should be ≤ 8 (optimal for this instance)
    assert result.cost <= 8.0 + 1e-9


def test_improve_maintains_optimal_tour():
    """2-opt improvement does not increase cost for already optimal tours."""
    solver = TSPSolver(algorithm="christofides", improve=True)
    # Equilateral triangle - any tour is optimal
    dist = 10.0
    instance = TSPInstance(
        cities=["A", "B", "C"],
        distance_matrix=[[0, dist, dist], [dist, 0, dist], [dist, dist, 0]],
    )
    result = solver.solve(instance)
    assert result.cost == pytest.approx(30.0)


def test_improve_with_christofides():
    """Christofides + 2-opt produces valid tour with reasonable cost."""
    solver = TSPSolver(algorithm="christofides", improve=True)
    cities = ["A", "B", "C", "D", "E"]
    n = len(cities)
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
    assert len(result.tour) == n
    assert set(result.tour) == set(cities)
    assert result.cost <= 8.0 * 1.5  # Christofides guarantee


def test_improve_with_nearest_neighbor():
    """Nearest neighbor + 2-opt produces valid tour."""
    solver = TSPSolver(algorithm="nearest_neighbor", improve=True)
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
    assert set(result.tour) == {"A", "B", "C", "D"}
    assert result.cost <= 4.0 + 1e-9  # Optimal for this instance


def test_no_improve_when_disabled():
    """When improve=False, no 2-opt post-processing is applied."""
    solver_no_improve = TSPSolver(algorithm="christofides", improve=False)
    solver_with_improve = TSPSolver(algorithm="christofides", improve=True)
    instance = TSPInstance(
        cities=["A", "B", "C", "D"],
        distance_matrix=[
            [0, 2, math.sqrt(8), 2],
            [2, 0, 2, math.sqrt(8)],
            [math.sqrt(8), 2, 0, 2],
            [2, math.sqrt(8), 2, 0],
        ],
    )
    result_no_improve = solver_no_improve.solve(instance)
    result_with_improve = solver_with_improve.solve(instance)
    # Improvement should reduce or maintain cost
    assert result_with_improve.cost <= result_no_improve.cost + 1e-9

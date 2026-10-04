"""Tests for TSP solver input validation and approximation ratio."""

import pytest

from src.optimization.tsp import TSPInstance, TSPSolver


def test_solve_rejects_nan_in_distance_matrix():
    """solve() raises ValueError when distance matrix contains NaN."""
    solver = TSPSolver()
    instance = TSPInstance(
        cities=["A", "B", "C"],
        distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    )
    instance.distance_matrix[0][2] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver.solve(instance)


def test_solve_rejects_inf_in_distance_matrix():
    """solve() raises ValueError when distance matrix contains Inf."""
    solver = TSPSolver()
    instance = TSPInstance(
        cities=["A", "B", "C"],
        distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    )
    instance.distance_matrix[0][2] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver.solve(instance)


def test_solve_rejects_negative_distance():
    """solve() raises ValueError when distance matrix contains negative value."""
    solver = TSPSolver()
    instance = TSPInstance(
        cities=["A", "B", "C"],
        distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    )
    instance.distance_matrix[0][2] = -5
    with pytest.raises(ValueError, match="negative"):
        solver.solve(instance)


def test_christofides_approximation_ratio_is_none():
    """Christofides with greedy matching does not guarantee 1.5 bound."""
    solver = TSPSolver(algorithm="christofides")
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
    assert result.approximation_ratio is None


def test_valid_instance_passes_validation():
    """Valid distance matrix passes solve() without error."""
    solver = TSPSolver()
    instance = TSPInstance(
        cities=["A", "B", "C"],
        distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    )
    result = solver.solve(instance)
    assert len(result.tour) == 3
    assert result.cost > 0

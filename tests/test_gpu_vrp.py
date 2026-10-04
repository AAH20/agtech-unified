"""Test GPU-accelerated VRP solver."""
import pytest
import math
from src.optimization.gpu_vrp import GPUVRPSolver, GPUVRPResult


def test_gpu_vrp_empty():
    """GPU VRP with no customers returns empty routes."""
    solver = GPUVRPSolver()
    result = solver.solve((0, 0), [], [], 100, [])
    assert result.routes == []
    assert result.total_cost == 0.0


def test_gpu_vrp_single_customer():
    """GPU VRP with one customer returns single route."""
    solver = GPUVRPSolver()
    result = solver.solve(
        (0, 0),
        [(3, 4)],
        [10],
        100,
        [[0, 5], [5, 0]],
    )
    assert len(result.routes) == 1
    assert len(result.routes[0]) == 1
    assert result.total_cost == pytest.approx(10.0)


def test_gpu_vrp_capacity_constraint():
    """GPU VRP respects vehicle capacity."""
    solver = GPUVRPSolver()
    result = solver.solve(
        (0, 0),
        [(1, 0), (2, 0), (3, 0)],
        [60, 60, 60],
        100,
        [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]],
    )
    assert len(result.routes) == 3
    for route in result.routes:
        total_demand = sum(60 for _ in route)
        assert total_demand <= 100


def test_gpu_vrp_two_customers_one_route():
    """Two customers fitting in one vehicle share a route."""
    solver = GPUVRPSolver()
    result = solver.solve(
        (0, 0),
        [(3, 0), (6, 0)],
        [30, 30],
        100,
        [[0, 3, 6], [3, 0, 3], [6, 3, 0]],
    )
    assert len(result.routes) == 1
    assert len(result.routes[0]) == 2


def test_gpu_vrp_savings_algorithm():
    """GPU savings algorithm produces valid routes covering all customers."""
    solver = GPUVRPSolver(algorithm="savings")
    result = solver.solve(
        (0, 0),
        [(1, 0), (0, 1), (-1, 0), (0, -1)],
        [10, 10, 10, 10],
        100,
        [
            [0, 1, 1.41, 1.41, 1],
            [1, 0, 1.41, 2, 1.41],
            [1.41, 1.41, 0, 1.41, 2],
            [1.41, 2, 1.41, 0, 1.41],
            [1, 1.41, 2, 1.41, 0],
        ],
    )
    assert len(result.routes) >= 1
    visited = [c for r in result.routes for c in r]
    assert sorted(visited) == [0, 1, 2, 3]


def test_gpu_vrp_route_cost_calculation():
    """Route cost includes depot->customer->depot."""
    solver = GPUVRPSolver()
    result = solver.solve(
        (0, 0),
        [(3, 4)],
        [10],
        100,
        [[0, 5], [5, 0]],
    )
    assert result.total_cost == pytest.approx(10.0)


def test_gpu_vrp_multiple_vehicles():
    """GPU VRP uses multiple vehicles when needed."""
    solver = GPUVRPSolver()
    result = solver.solve(
        (0, 0),
        [(1, 0), (2, 0), (3, 0), (4, 0)],
        [80, 80, 80, 80],
        100,
        [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]],
    )
    assert len(result.routes) == 4


def test_gpu_vrp_savings_tensor():
    """GPU savings tensor computation produces correct values."""
    solver = GPUVRPSolver()
    dist_matrix = [
        [0, 2, 3, 4],
        [2, 0, 1, 2],
        [3, 1, 0, 1],
        [4, 2, 1, 0],
    ]
    savings = solver._compute_savings_tensor(
        __import__("torch").tensor(dist_matrix, dtype=__import__("torch").float32)
    )
    # s(0,1) = d(0,1) + d(0,2) - d(1,2) = 2 + 3 - 1 = 4
    assert savings[0, 1].item() == pytest.approx(4.0)
    # s(0,2) = d(0,1) + d(0,3) - d(1,3) = 2 + 4 - 2 = 4
    assert savings[0, 2].item() == pytest.approx(4.0)
    # s(1,2) = d(0,2) + d(0,3) - d(2,3) = 3 + 4 - 1 = 6
    assert savings[1, 2].item() == pytest.approx(6.0)
    # Diagonal should be zero
    assert savings[0, 0].item() == pytest.approx(0.0)


def test_gpu_vrp_savings_batch():
    """Batch savings computation returns correct number of tensors."""
    solver = GPUVRPSolver()
    dist_matrices = [
        [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        [[0, 5], [5, 0]],
    ]
    savings_list = solver.compute_savings_batch(dist_matrices)
    assert len(savings_list) == 2
    assert savings_list[0].shape == (2, 2)
    assert savings_list[1].shape == (1, 1)


def test_gpu_vrp_result_contains_device():
    """Result includes device information."""
    solver = GPUVRPSolver()
    result = solver.solve(
        (0, 0),
        [(1, 0)],
        [10],
        100,
        [[0, 1], [1, 0]],
    )
    assert result.device in ("cpu", "cuda")
    assert result.algorithm == "savings"

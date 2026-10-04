"""Test GPU-accelerated TSP solver."""
import pytest
import math
from src.optimization.gpu_tsp import GPUTSPSolver, GPUTSPResult


def test_gpu_tsp_empty_instance():
    """GPU TSP with no cities returns empty result."""
    solver = GPUTSPSolver()
    result = solver.solve([], [])
    assert result.tour == []
    assert result.cost == 0.0


def test_gpu_tsp_single_city():
    """GPU TSP with one city returns trivial tour."""
    solver = GPUTSPSolver()
    result = solver.solve(["A"], [[0]])
    assert result.tour == ["A"]
    assert result.cost == 0.0


def test_gpu_tsp_two_cities():
    """GPU TSP with two cities returns round trip."""
    solver = GPUTSPSolver()
    result = solver.solve(["A", "B"], [[0, 10], [10, 0]])
    assert len(result.tour) == 2
    assert result.cost == pytest.approx(20.0)


def test_gpu_tsp_triangle():
    """GPU TSP with equilateral triangle returns valid tour."""
    solver = GPUTSPSolver()
    dist = 10.0
    result = solver.solve(
        ["A", "B", "C"],
        [[0, dist, dist], [dist, 0, dist], [dist, dist, 0]]
    )
    assert len(result.tour) == 3
    assert set(result.tour) == {"A", "B", "C"}
    assert result.cost == pytest.approx(30.0)


def test_gpu_tsp_nearest_neighbor():
    """GPU nearest neighbor produces valid tour."""
    solver = GPUTSPSolver(algorithm="nearest_neighbor")
    result = solver.solve(
        ["A", "B", "C", "D"],
        [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    )
    assert len(result.tour) == 4
    assert set(result.tour) == {"A", "B", "C", "D"}
    assert result.cost > 0


def test_gpu_tsp_two_opt():
    """GPU 2-opt improves or maintains tour quality."""
    solver = GPUTSPSolver(algorithm="two_opt")
    result = solver.solve(
        ["A", "B", "C", "D"],
        [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
    )
    assert len(result.tour) == 4
    assert set(result.tour) == {"A", "B", "C", "D"}
    assert result.cost > 0


def test_gpu_tsp_batched_distance_matrix():
    """Batched distance matrix computation produces correct shapes and values."""
    solver = GPUTSPSolver()
    coords_batch = [
        [(0, 0), (3, 4), (6, 8)],
        [(0, 0), (1, 0)],
    ]
    dist = solver.compute_distance_matrix_batch(coords_batch)
    assert dist.shape == (2, 3, 3)
    # First instance: (0,0) to (3,4) = 5
    assert dist[0, 0, 1].item() == pytest.approx(5.0)
    # First instance: (0,0) to (6,8) = 10
    assert dist[0, 0, 2].item() == pytest.approx(10.0)
    # Second instance: (0,0) to (1,0) = 1
    assert dist[1, 0, 1].item() == pytest.approx(1.0)


def test_gpu_tsp_solve_batch():
    """Batch solving multiple instances returns correct number of results."""
    solver = GPUTSPSolver()
    instances = [
        (["A", "B"], [[0, 5], [5, 0]]),
        (["X", "Y", "Z"], [[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
    ]
    results = solver.solve_batch(instances)
    assert len(results) == 2
    assert results[0].cost == pytest.approx(10.0)
    assert len(results[1].tour) == 3


def test_gpu_tsp_result_contains_device():
    """Result includes device information."""
    solver = GPUTSPSolver()
    result = solver.solve(["A", "B"], [[0, 1], [1, 0]])
    assert result.device in ("cpu", "cuda")
    assert result.algorithm == "nearest_neighbor"


def test_gpu_tsp_invalid_algorithm():
    """Unknown algorithm raises ValueError."""
    solver = GPUTSPSolver(algorithm="invalid_algo")
    with pytest.raises(ValueError, match="Unknown algorithm"):
        solver.solve(["A", "B"], [[0, 1], [1, 0]])

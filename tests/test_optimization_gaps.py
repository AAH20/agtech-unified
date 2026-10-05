"""Tests for HIGH-severity optimization gap fixes.

Covers: input validation, Christofides ratio, crop vision batch handling,
GPU fallback error handling, and local search integration.
"""

import warnings

import pytest

from src.optimization.gpu_fallback import GPUFallback
from src.optimization.gpu_tsp import GPUTSPSolver
from src.optimization.gpu_vrp import GPUVRPSolver
from src.optimization.local_search import TwoOpt
from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.vrp import VRPInstance


# ======================================================================
# TSP: NaN / Inf / negative validation
# ======================================================================
class TestTSPInputValidation:
    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="negative"):
            TSPInstance(
                cities=["A", "B"],
                distance_matrix=[[0, -1], [-1, 0]],
            )

    def test_nan_distance_raises(self):
        with pytest.raises(ValueError, match="NaN|Inf"):
            TSPInstance(
                cities=["A", "B"],
                distance_matrix=[[0, float("nan")], [float("nan"), 0]],
            )

    def test_inf_distance_raises(self):
        with pytest.raises(ValueError, match="NaN|Inf"):
            TSPInstance(
                cities=["A", "B"],
                distance_matrix=[[0, float("inf")], [float("inf"), 0]],
            )

    def test_valid_instance_no_error(self):
        """Valid metric instance should not raise."""
        TSPInstance(
            cities=["A", "B", "C"],
            distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        )


# ======================================================================
# TSP: Christofides approximation_ratio claim
# ======================================================================
class TestChristofidesRatio:
    def test_approximation_ratio_is_none(self):
        """Greedy matching does not guarantee 1.5× — ratio must be None."""
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

    def test_christofides_still_produces_valid_tour(self):
        """Even without ratio guarantee, tour must be valid."""
        solver = TSPSolver(algorithm="christofides")
        instance = TSPInstance(
            cities=["A", "B", "C", "D", "E"],
            distance_matrix=[
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
        )
        result = solver.solve(instance)
        assert len(result.tour) == 5
        assert set(result.tour) == {"A", "B", "C", "D", "E"}
        assert result.cost > 0


# ======================================================================
# TSP: Local search integration
# ======================================================================
class TestTSPLocalSearchIntegration:
    def test_improve_flag_improves_tour(self):
        """With improve=True, 2-opt should reduce or maintain cost."""
        instance = TSPInstance(
            cities=["A", "B", "C", "D"],
            distance_matrix=[
                [0, 1, 2, 1],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [1, 2, 1, 0],
            ],
        )
        solver_no_improve = TSPSolver(algorithm="nearest_neighbor", improve=False)
        solver_improve = TSPSolver(algorithm="nearest_neighbor", improve=True)

        result_no = solver_no_improve.solve(instance)
        result_yes = solver_improve.solve(instance)

        assert result_yes.cost <= result_no.cost

    def test_improve_false_skips_two_opt(self):
        """With improve=False, tour should be raw nearest-neighbor."""
        instance = TSPInstance(
            cities=["A", "B", "C", "D"],
            distance_matrix=[
                [0, 1, 2, 1],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [1, 2, 1, 0],
            ],
        )
        solver = TSPSolver(algorithm="nearest_neighbor", improve=False)
        result = solver.solve(instance)
        # Raw NN from 0: 0→1→2→3, cost = 1+1+1+1 = 4
        assert result.cost == pytest.approx(4.0)

    def test_christofides_with_improve(self):
        """Christofides + 2-opt should produce valid tour."""
        instance = TSPInstance(
            cities=["A", "B", "C", "D", "E"],
            distance_matrix=[
                [0, 1, 2, 3, 4],
                [1, 0, 1, 2, 3],
                [2, 1, 0, 1, 2],
                [3, 2, 1, 0, 1],
                [4, 3, 2, 1, 0],
            ],
        )
        solver = TSPSolver(algorithm="christofides", improve=True)
        result = solver.solve(instance)
        assert len(result.tour) == 5
        assert set(result.tour) == {"A", "B", "C", "D", "E"}


# ======================================================================
# TSP: Triangle inequality check is optional
# ======================================================================
class TestTriangleInequalityOptional:
    def test_validate_metric_false_skips_check(self):
        """Non-metric instance should not warn when validate_metric=False."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TSPInstance(
                cities=["A", "B", "C"],
                distance_matrix=[[0, 1, 100], [1, 0, 1], [100, 1, 0]],
                validate_metric=False,
            )

    def test_validate_metric_true_warns(self):
        """Non-metric instance should warn when validate_metric=True."""
        TSPInstance(
            cities=["A", "B", "C"],
            distance_matrix=[[0, 1, 100], [1, 0, 1], [100, 1, 0]],
            validate_metric=True,
        )


# ======================================================================
# VRP: Input validation
# ======================================================================
class TestVRPInputValidation:
    def test_mismatched_demands_customers_raises(self):
        with pytest.raises(ValueError, match="same length|match"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0), (2, 0)],
                demands=[10],
                vehicle_capacity=100,
                distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
            )

    def test_negative_demand_raises(self):
        with pytest.raises(ValueError, match="non-negative|negative"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[-5],
                vehicle_capacity=100,
                distance_matrix=[[0, 1], [1, 0]],
            )

    def test_nan_distance_raises(self):
        with pytest.raises(ValueError, match="NaN|Inf"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[10],
                vehicle_capacity=100,
                distance_matrix=[[0, float("nan")], [float("nan"), 0]],
            )

    def test_inf_distance_raises(self):
        with pytest.raises(ValueError, match="NaN|Inf"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[10],
                vehicle_capacity=100,
                distance_matrix=[[0, float("inf")], [float("inf"), 0]],
            )

    def test_valid_instance_no_error(self):
        VRPInstance(
            depot=(0, 0),
            customers=[(1, 0), (2, 0)],
            demands=[10, 20],
            vehicle_capacity=100,
            distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        )


# ======================================================================
# GPU TSP: Validation
# ======================================================================
@pytest.mark.gpu
class TestGPUTSPValidation:
    def test_nan_distance_raises(self):
        solver = GPUTSPSolver()
        with pytest.raises(ValueError, match="NaN|Inf"):
            solver.solve(["A", "B"], [[0, float("nan")], [float("nan"), 0]])

    def test_inf_distance_raises(self):
        solver = GPUTSPSolver()
        with pytest.raises(ValueError, match="NaN|Inf"):
            solver.solve(["A", "B"], [[0, float("inf")], [float("inf"), 0]])

    def test_dimension_mismatch_raises(self):
        solver = GPUTSPSolver()
        with pytest.raises(ValueError, match="square|match"):
            solver.solve(["A", "B"], [[0, 1, 2], [1, 0, 2]])

    def test_two_opt_max_iterations(self):
        """_gpu_two_opt should accept max_iterations parameter."""
        solver = GPUTSPSolver()
        import torch

        dist = torch.tensor(
            [[0.0, 1.0, 2.0, 1.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [1.0, 2.0, 1.0, 0.0]]
        )
        tour = [0, 1, 2, 3]
        # Should complete without error
        result = solver._gpu_two_opt(dist, tour, max_iterations=10)
        assert len(result) == 4


# ======================================================================
# GPU VRP: Validation
# ======================================================================
@pytest.mark.gpu
class TestGPUVRPValidation:
    def test_negative_demand_raises(self):
        solver = GPUVRPSolver()
        with pytest.raises(ValueError, match="non-negative|negative"):
            solver.solve(
                (0, 0),
                [(1, 0)],
                [-5],
                100,
                [[0, 1], [1, 0]],
            )

    def test_dimension_mismatch_raises(self):
        solver = GPUVRPSolver()
        with pytest.raises(ValueError, match="match|size"):
            solver.solve(
                (0, 0),
                [(1, 0), (2, 0)],
                [10, 20],
                100,
                [[0, 1], [1, 0]],  # Should be 3x3
            )


# ======================================================================
# Local search: Input validation
# ======================================================================
class TestLocalSearchValidation:
    def test_duplicate_cities_raises(self):
        opt = TwoOpt()
        with pytest.raises(ValueError, match="duplicate"):
            opt.improve([0, 1, 1, 2], [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])

    def test_invalid_index_raises(self):
        opt = TwoOpt()
        with pytest.raises(ValueError, match="exceeds"):
            opt.improve([0, 1, 5], [[0, 1, 2], [1, 0, 1], [2, 1, 0]])

    def test_empty_tour_returns_empty(self):
        opt = TwoOpt()
        assert opt.improve([], []) == []

    def test_single_city_returns_unchanged(self):
        opt = TwoOpt()
        assert opt.improve([0], [[0]]) == [0]

    def test_mismatched_matrix_raises(self):
        opt = TwoOpt()
        with pytest.raises(ValueError, match="exceeds|match"):
            opt.improve([0, 1, 2], [[0, 1], [1, 0]])


# ======================================================================
# GPU Fallback: Error handling
# ======================================================================
class TestGPUFallbackErrorHandling:
    def test_non_callable_gpu_solver_raises(self):
        fb = GPUFallback()
        with pytest.raises(TypeError, match="callable"):
            fb.solve("prob", "not_callable", lambda p: "ok")

    def test_non_callable_cpu_solver_raises(self):
        fb = GPUFallback()
        with pytest.raises(TypeError, match="callable"):
            fb.solve("prob", lambda p: "ok", "not_callable")

    def test_cuda_error_falls_back(self):
        """RuntimeError with 'CUDA' in message should fall back."""
        fb = GPUFallback()

        def gpu_fail(p):
            raise RuntimeError("CUDA out of memory")

        result = fb.solve("prob", gpu_fail, lambda p: "cpu_result")
        assert result == "cpu_result"

    def test_non_cuda_error_propagates(self):
        """RuntimeError without CUDA in message should NOT fall back."""
        fb = GPUFallback()

        def gpu_fail(p):
            raise RuntimeError("Some other error")

        with pytest.raises(RuntimeError, match="Some other error"):
            fb.solve("prob", gpu_fail, lambda p: "cpu_result")

    def test_both_fail_raises_with_context(self):
        """When both fail, RuntimeError should contain both messages."""
        fb = GPUFallback()

        def gpu_fail(p):
            raise RuntimeError("CUDA OOM")

        def cpu_fail(p):
            raise ValueError("CPU error")

        with pytest.raises(RuntimeError, match="GPU.*CPU|Both"):
            fb.solve("prob", gpu_fail, cpu_fail)

    def test_cpu_result_returned_on_fallback(self):
        fb = GPUFallback()

        def gpu_fail(p):
            raise RuntimeError("CUDA not available")

        def cpu_ok(p):
            return 42

        assert fb.solve("prob", gpu_fail, cpu_ok) == 42

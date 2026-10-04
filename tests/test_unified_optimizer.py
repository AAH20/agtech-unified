"""Test unified optimizer interface wrapping TSP/VRP/GPU solvers behind one API."""

import pytest

from src.integration.unified_optimizer import (
    OptimizationProblem,
    OptimizationResult,
    SolverType,
    UnifiedOptimizer,
)


def _tsp_dist():
    return [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]


def _vrp_dist():
    return [
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ]


class TestUnifiedTSP:
    """TSP through the unified interface."""

    def test_solve_tsp_returns_result(self):
        """UnifiedOptimizer solves a TSP problem and returns an OptimizationResult."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B", "C", "D"],
            distance_matrix=_tsp_dist(),
        )

        result = opt.solve(problem)

        assert isinstance(result, OptimizationResult)
        assert result.algorithm == "christofides"
        assert result.cost == pytest.approx(4.0)
        assert set(result.solution) == {"A", "B", "C", "D"}

    def test_solve_tsp_nearest_neighbor(self):
        """TSP with nearest_neighbor algorithm is selectable via solver_type."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B", "C"],
            distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
            algorithm="nearest_neighbor",
        )

        result = opt.solve(problem)

        assert result.algorithm == "nearest_neighbor"
        assert len(result.solution) == 3

    def test_solve_tsp_empty(self):
        """Empty TSP instance yields zero cost and empty solution."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=[],
            distance_matrix=[],
        )

        result = opt.solve(problem)

        assert result.cost == 0.0
        assert result.solution == []


class TestUnifiedVRP:
    """VRP through the unified interface."""

    def test_solve_vrp_returns_routes(self):
        """UnifiedOptimizer solves a VRP problem and returns routes."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.VRP,
            depot=(0, 0),
            customers=[(1, 0), (2, 0), (3, 0)],
            demands=[60, 60, 60],
            vehicle_capacity=100,
            distance_matrix=_vrp_dist(),
        )

        result = opt.solve(problem)

        assert result.algorithm == "savings"
        assert result.cost > 0
        # 60+60 > 100, so each customer needs its own route
        assert len(result.solution) == 3

    def test_solve_vrp_capacity_respected(self):
        """VRP routes respect vehicle capacity."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.VRP,
            depot=(0, 0),
            customers=[(1, 0), (2, 0), (3, 0)],
            demands=[60, 60, 60],
            vehicle_capacity=100,
            distance_matrix=_vrp_dist(),
        )

        result = opt.solve(problem)

        for route in result.solution:
            assert sum(60 for _ in route) <= 100

    def test_solve_vrp_all_customers_visited(self):
        """Every customer appears in exactly one route."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.VRP,
            depot=(0, 0),
            customers=[(1, 0), (2, 0), (3, 0)],
            demands=[60, 60, 60],
            vehicle_capacity=100,
            distance_matrix=_vrp_dist(),
        )

        result = opt.solve(problem)

        visited = [c for route in result.solution for c in route]
        assert sorted(visited) == [0, 1, 2]


class TestUnifiedGPU:
    """GPU solvers through the unified interface."""

    def test_solve_gpu_tsp(self):
        """GPUTSP solver type produces a valid tour."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.GPU_TSP,
            cities=["A", "B", "C", "D"],
            distance_matrix=_tsp_dist(),
        )

        result = opt.solve(problem)

        assert result.algorithm in ("nearest_neighbor", "two_opt")
        assert set(result.solution) == {"A", "B", "C", "D"}
        assert result.cost > 0

    def test_solve_gpu_vrp(self):
        """GPUVRP solver type produces valid routes."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.GPU_VRP,
            depot=(0, 0),
            customers=[(1, 0), (2, 0), (3, 0)],
            demands=[60, 60, 60],
            vehicle_capacity=100,
            distance_matrix=_vrp_dist(),
        )

        result = opt.solve(problem)

        assert result.algorithm == "savings"
        visited = [c for route in result.solution for c in route]
        assert sorted(visited) == [0, 1, 2]


class TestUnifiedInterface:
    """Cross-cutting behavior of the unified API."""

    def test_solve_batch(self):
        """solve_batch solves multiple problems and returns one result each."""
        opt = UnifiedOptimizer()
        problems = [
            OptimizationProblem(
                solver_type=SolverType.TSP,
                cities=["A", "B"],
                distance_matrix=[[0, 5], [5, 0]],
            ),
            OptimizationProblem(
                solver_type=SolverType.TSP,
                cities=["X", "Y"],
                distance_matrix=[[0, 7], [7, 0]],
            ),
        ]

        results = opt.solve_batch(problems)

        assert len(results) == 2
        assert results[0].cost == pytest.approx(10.0)
        assert results[1].cost == pytest.approx(14.0)

    def test_unknown_solver_type_raises(self):
        """An unrecognized solver type raises ValueError."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(solver_type="quantum", cities=["A"])

        with pytest.raises(ValueError, match="Unknown solver type"):
            opt.solve(problem)

    def test_result_metadata(self):
        """OptimizationResult carries solver_type and algorithm metadata."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B"],
            distance_matrix=[[0, 5], [5, 0]],
        )

        result = opt.solve(problem)

        assert result.solver_type == SolverType.TSP
        assert result.algorithm == "christofides"
        assert result.cost == pytest.approx(10.0)

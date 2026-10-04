"""Unified optimizer interface wrapping TSP/VRP/GPU solvers behind a single API.

Addresses GAP-017: consumers no longer need to know which specific solver
to import. One OptimizationProblem in, one OptimizationResult out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from src.optimization.tsp import TSPSolver, TSPInstance
from src.optimization.vrp import VRPSolver, VRPInstance
from src.optimization.gpu_tsp import GPUTSPSolver
from src.optimization.gpu_vrp import GPUVRPSolver


class SolverType(str, Enum):
    """Available solver backends."""

    TSP = "tsp"
    VRP = "vrp"
    GPU_TSP = "gpu_tsp"
    GPU_VRP = "gpu_vrp"


@dataclass
class OptimizationProblem:
    """A routing problem specification.

    Fields used depend on solver_type:
        TSP/GPU_TSP:    cities, distance_matrix
        VRP/GPU_VRP:    depot, customers, demands, vehicle_capacity, distance_matrix
    """
    solver_type: SolverType
    cities: Optional[List[str]] = None
    distance_matrix: Optional[List[List[float]]] = None
    depot: Optional[Tuple[float, float]] = None
    customers: Optional[List[Tuple[float, float]]] = None
    demands: Optional[List[float]] = None
    vehicle_capacity: Optional[float] = None
    algorithm: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Unified result from any solver.

    Attributes:
        solution: TSP tour (list of city names) or VRP routes (list of int lists).
        cost: Total cost (tour length or route distance).
        algorithm: Algorithm name reported by the underlying solver.
        solver_type: Which solver produced this result.
    """
    solution: List[Any]
    cost: float
    algorithm: str
    solver_type: SolverType


class UnifiedOptimizer:
    """Single entry point for all routing solvers.

    Usage:
        opt = UnifiedOptimizer()
        result = opt.solve(OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B", "C"],
            distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        ))
    """

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve a single problem with the appropriate backend solver."""
        st = problem.solver_type
        if st == SolverType.TSP:
            return self._solve_tsp(problem)
        elif st == SolverType.VRP:
            return self._solve_vrp(problem)
        elif st == SolverType.GPU_TSP:
            return self._solve_gpu_tsp(problem)
        elif st == SolverType.GPU_VRP:
            return self._solve_gpu_vrp(problem)
        raise ValueError(f"Unknown solver type: {st!r}")

    def solve_batch(self, problems: List[OptimizationProblem]) -> List[OptimizationResult]:
        """Solve multiple problems, returning one result per problem."""
        return [self.solve(p) for p in problems]

    def _solve_tsp(self, problem: OptimizationProblem) -> OptimizationResult:
        algorithm = problem.algorithm or "christofides"
        solver = TSPSolver(algorithm=algorithm)
        instance = TSPInstance(
            cities=problem.cities or [],
            distance_matrix=problem.distance_matrix or [],
        )
        result = solver.solve(instance)
        return OptimizationResult(
            solution=result.tour,
            cost=result.cost,
            algorithm=result.algorithm,
            solver_type=SolverType.TSP,
        )

    def _solve_vrp(self, problem: OptimizationProblem) -> OptimizationResult:
        algorithm = problem.algorithm or "savings"
        solver = VRPSolver(algorithm=algorithm)
        instance = VRPInstance(
            depot=problem.depot or (0.0, 0.0),
            customers=problem.customers or [],
            demands=problem.demands or [],
            vehicle_capacity=problem.vehicle_capacity or 0.0,
            distance_matrix=problem.distance_matrix or [],
        )
        result = solver.solve(instance)
        return OptimizationResult(
            solution=result.routes,
            cost=result.total_cost,
            algorithm=result.algorithm,
            solver_type=SolverType.VRP,
        )

    def _solve_gpu_tsp(self, problem: OptimizationProblem) -> OptimizationResult:
        algorithm = problem.algorithm or "nearest_neighbor"
        solver = GPUTSPSolver(algorithm=algorithm)
        result = solver.solve(
            cities=problem.cities or [],
            distance_matrix=problem.distance_matrix or [],
        )
        return OptimizationResult(
            solution=result.tour,
            cost=result.cost,
            algorithm=result.algorithm,
            solver_type=SolverType.GPU_TSP,
        )

    def _solve_gpu_vrp(self, problem: OptimizationProblem) -> OptimizationResult:
        algorithm = problem.algorithm or "savings"
        solver = GPUVRPSolver(algorithm=algorithm)
        result = solver.solve(
            depot=problem.depot or (0.0, 0.0),
            customers=problem.customers or [],
            demands=problem.demands or [],
            vehicle_capacity=problem.vehicle_capacity or 0.0,
            distance_matrix=problem.distance_matrix or [],
        )
        return OptimizationResult(
            solution=result.routes,
            cost=result.total_cost,
            algorithm=result.algorithm,
            solver_type=SolverType.GPU_VRP,
        )

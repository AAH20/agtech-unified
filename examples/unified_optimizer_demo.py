"""Unified optimizer demo: single API for TSP/VRP/GPU solvers."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.integration.unified_optimizer import UnifiedOptimizer, SolverType, OptimizationProblem


def main():
    optimizer = UnifiedOptimizer()
    problem = OptimizationProblem(
        solver_type=SolverType.TSP,
        cities=["A", "B", "C", "D"],
        distance_matrix=[[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]],
    )
    result = optimizer.solve(problem)
    print(f"Solver: {result.solver_type}")
    print(f"Tour: {result.tour}")
    print(f"Cost: {result.cost:.1f}")


if __name__ == "__main__":
    main()

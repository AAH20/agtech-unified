"""Basic TSP example: solve a small routing problem."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.tsp import TSPInstance, TSPSolver


def main():
    cities = ["Farm", "Field-A", "Field-B", "Field-C"]
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    result = TSPSolver("christofides").solve(TSPInstance(cities, dist))
    print(f"TSP Tour: {result.tour}")
    print(f"TSP Cost: {result.cost:.1f}")


if __name__ == "__main__":
    main()

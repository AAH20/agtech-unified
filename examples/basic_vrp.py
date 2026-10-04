"""Basic VRP example: solve a vehicle routing problem."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.vrp import VRPInstance, VRPSolver


def main():
    depot = (0.0, 0.0)
    customers = [(1.0, 2.0), (3.0, 1.0), (2.0, 4.0)]
    demands = [5.0, 8.0, 6.0]
    dist = [
        [0, 2.2, 3.2, 4.5],
        [2.2, 0, 2.8, 3.6],
        [3.2, 2.8, 0, 3.2],
        [4.5, 3.6, 3.2, 0],
    ]
    result = VRPSolver("savings").solve(VRPInstance(depot, customers, demands, 15.0, dist))
    print(f"VRP Routes: {result.routes}")
    print(f"VRP Cost: {result.total_cost:.1f}")
    print(f"VRP Vehicles: {result.num_vehicles}")


if __name__ == "__main__":
    main()

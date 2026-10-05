"""Benchmark optimization module: TSP ratios, VRP quality, missing algorithms."""
import json
import math
import random
import time

from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.held_karp import HeldKarp
from src.optimization.local_search import TwoOpt
from src.optimization.three_opt import ThreeOpt
from src.optimization.lin_kernighan import LinKernighan
from src.optimization.simulated_annealing import SimulatedAnnealingTSP
from src.optimization.vrp import VRPInstance, VRPSolver

random.seed(42)


def random_metric_instance(n):
    pts = [(random.random(), random.random()) for _ in range(n)]
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                dist[i][j] = math.sqrt(dx * dx + dy * dy)
    return dist


def tour_cost(tour, dist):
    n = len(tour)
    if n <= 1:
        return 0.0
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


print("=== TSP Approximation Ratio vs Optimal (Held-Karp) ===")
tsp_results = []
for n in [8, 10, 12]:
    dist = random_metric_instance(n)
    cities = [f"c{i}" for i in range(n)]
    instance = TSPInstance(cities=cities, distance_matrix=dist)

    hk = HeldKarp(max_cities=20)
    opt_cost, _ = hk.solve(dist)

    solver = TSPSolver(algorithm="christofides", improve=False)
    c_cost = solver.solve(instance).cost

    solver = TSPSolver(algorithm="nearest_neighbor", improve=False)
    nn_cost = solver.solve(instance).cost

    solver = TSPSolver(algorithm="nearest_neighbor", improve=True)
    nn2_cost = solver.solve(instance).cost

    solver = TSPSolver(algorithm="nearest_neighbor", improve=False)
    result = solver.solve(instance)
    tour_idx = [cities.index(c) for c in result.tour]

    three_opt = ThreeOpt(max_iterations=50)
    three_cost = tour_cost(three_opt.improve(tour_idx, dist), dist)

    lk = LinKernighan(restarts=1, max_iterations=20)
    lk_cost = tour_cost(lk.improve(tour_idx, dist), dist)

    sa = SimulatedAnnealingTSP(max_iterations=2000, seed=42)
    _, sa_cost = sa.solve(dist)

    row = {
        "n": n,
        "optimal": round(opt_cost, 4),
        "christofides": round(c_cost, 4),
        "christofides_ratio": round(c_cost / opt_cost, 4),
        "nn": round(nn_cost, 4),
        "nn_ratio": round(nn_cost / opt_cost, 4),
        "nn_2opt": round(nn2_cost, 4),
        "nn_2opt_ratio": round(nn2_cost / opt_cost, 4),
        "three_opt": round(three_cost, 4),
        "three_opt_ratio": round(three_cost / opt_cost, 4),
        "lk": round(lk_cost, 4),
        "lk_ratio": round(lk_cost / opt_cost, 4),
        "sa": round(sa_cost, 4),
        "sa_ratio": round(sa_cost / opt_cost, 4),
    }
    tsp_results.append(row)
    print(json.dumps(row))

print()
print("=== VRP Solution Quality ===")
# Simple VRP: 10 customers, capacity 30, demands 5-15
n = 10
pts = [(random.random() * 10, random.random() * 10) for _ in range(n)]
depot = (5.0, 5.0)
all_pts = [depot] + pts
dist = [[0.0] * (n + 1) for _ in range(n + 1)]
for i in range(n + 1):
    for j in range(n + 1):
        if i != j:
            dx = all_pts[i][0] - all_pts[j][0]
            dy = all_pts[i][1] - all_pts[j][1]
            dist[i][j] = math.sqrt(dx * dx + dy * dy)

demands = [random.randint(5, 15) for _ in range(n)]
capacity = 30

vrp_instance = VRPInstance(
    depot=depot,
    customers=pts,
    demands=demands,
    vehicle_capacity=capacity,
    distance_matrix=dist,
)

solver = VRPSolver(algorithm="savings")
vrp_result = solver.solve(vrp_instance)
print(f"VRP: {n} customers, capacity={capacity}, demands={demands}")
print(f"  Routes: {vrp_result.num_vehicles}, Total cost: {vrp_result.total_cost:.4f}")
print(f"  Route details: {vrp_result.routes}")

# Check capacity constraints
for i, route in enumerate(vrp_result.routes):
    load = sum(demands[c] for c in route)
    print(f"  Route {i}: load={load}, capacity={capacity}, feasible={load <= capacity}")

# Compare with a simple upper bound: each customer alone
ub_cost = sum(2 * dist[0][c + 1] for c in range(n))
print(f"  Individual-delivery upper bound: {ub_cost:.4f}")
print(f"  Savings / UB ratio: {vrp_result.total_cost / ub_cost:.4f}")

print()
print("=== Missing Algorithms Check ===")
missing = []
# Check for GA
try:
    from src.optimization.genetic_algorithm import GeneticAlgorithmTSP
    print("GA: FOUND")
except ImportError:
    missing.append("Genetic Algorithm (GA) for TSP/VRP")
    print("GA: MISSING")

# Check for ACO
try:
    from src.optimization.aco import ACOSolver
    print("ACO: FOUND")
except ImportError:
    missing.append("Ant Colony Optimization (ACO) for TSP/VRP")
    print("ACO: MISSING")

# Check for ALNS
try:
    from src.optimization.alns import ALNSSolver
    print("ALNS: FOUND")
except ImportError:
    missing.append("Adaptive Large Neighborhood Search (ALNS) for VRP")
    print("ALNS: MISSING")

# Check for VRPTW
try:
    from src.optimization.vrptw import VRPTWSolver
    print("VRPTW: FOUND")
except ImportError:
    missing.append("VRP with Time Windows (VRPTW)")
    print("VRPTW: MISSING")

# Check for multi-depot
try:
    from src.optimization.multi_depot_vrp import MultiDepotVRPSolver
    print("Multi-depot VRP: FOUND")
except ImportError:
    missing.append("Multi-depot VRP")
    print("Multi-depot VRP: MISSING")

# Check for heterogeneous fleet
try:
    from src.optimization.heterogeneous_fleet import HeterogeneousFleetSolver
    print("Heterogeneous fleet: FOUND")
except ImportError:
    missing.append("Heterogeneous fleet VRP")
    print("Heterogeneous fleet: MISSING")

# Check for Edmonds' blossom
try:
    from src.optimization.blossom import min_weight_matching
    print("Edmonds' blossom: FOUND")
except ImportError:
    missing.append("Edmonds' blossom algorithm for optimal matching")
    print("Edmonds' blossom: MISSING")

print()
print("=== Summary ===")
print(f"TSP ratios: Christofides max={max(r['christofides_ratio'] for r in tsp_results):.3f}, "
      f"NN max={max(r['nn_ratio'] for r in tsp_results):.3f}, "
      f"LK max={max(r['lk_ratio'] for r in tsp_results):.3f}")
print(f"VRP: {vrp_result.num_vehicles} vehicles, cost={vrp_result.total_cost:.4f}")
print(f"Missing algorithms: {len(missing)}")
for m in missing:
    print(f"  - {m}")

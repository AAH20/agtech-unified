# Tutorial 02: Optimization Basics

## Overview

Learn how to use TSP, VRP, and GPU-accelerated solvers.

## TSP Solver

```python
from src.optimization.tsp import TSPInstance, TSPSolver

cities = ["A", "B", "C", "D", "E"]
dist = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0],
]

result = TSPSolver("christofides").solve(TSPInstance(cities, dist))
print(f"Tour: {result.tour}, Cost: {result.cost}")
```

## VRP Solver

```python
from src.optimization.vrp import VRPInstance, VRPSolver

depot = (0.0, 0.0)
customers = [(1.0, 2.0), (3.0, 1.0), (2.0, 4.0), (5.0, 3.0)]
demands = [5.0, 8.0, 6.0, 7.0]
dist = [
    [0, 2.2, 3.2, 4.5, 5.8],
    [2.2, 0, 2.8, 3.6, 4.2],
    [3.2, 2.8, 0, 3.2, 3.9],
    [4.5, 3.6, 3.2, 0, 2.5],
    [5.8, 4.2, 3.9, 2.5, 0],
]

result = VRPSolver("savings").solve(VRPInstance(depot, customers, demands, 15.0, dist))
print(f"Routes: {result.routes}, Vehicles: {result.num_vehicles}")
```

## GPU Acceleration

```python
from src.optimization.gpu_tsp import GPUTSPSolver

# Automatically uses GPU if available, falls back to CPU
solver = GPUTSPSolver()
result = solver.solve(dist)
print(f"GPU Tour: {result.tour}, Cost: {result.cost}")
```

## Next Steps

- [Tutorial 03: Swarm Coordination](03-swarm-coordination.md)
- [Tutorial 04: IoT Data Pipeline](04-iot-data-pipeline.md)

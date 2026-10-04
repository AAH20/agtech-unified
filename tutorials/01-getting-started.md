# Tutorial 01: Getting Started

## Overview

This tutorial walks you through installing agtech-unified and running your first optimization.

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
git clone https://github.com/AAH20/agtech-unified.git
cd agtech-unified
pip install -e .
```

## Your First TSP Solve

```python
from src.optimization.tsp import TSPInstance, TSPSolver

# Define cities and distance matrix
cities = ["Farm", "Field-A", "Field-B", "Field-C"]
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]

# Solve
instance = TSPInstance(cities, dist)
solver = TSPSolver(algorithm="christofides")
result = solver.solve(instance)

print(f"Tour: {result.tour}")
print(f"Cost: {result.cost}")
```

## Next Steps

- [Tutorial 02: Optimization Basics](02-optimization-basics.md)
- [Tutorial 03: Swarm Coordination](03-swarm-coordination.md)

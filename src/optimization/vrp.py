"""VRP solver with Clarke-Wright savings algorithm."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VRPInstance:
    """VRP problem instance."""
    depot: Tuple[float, float]
    customers: List[Tuple[float, float]]
    demands: List[float]
    vehicle_capacity: float
    distance_matrix: List[List[float]]

    def __post_init__(self):
        if self.vehicle_capacity <= 0:
            raise ValueError("Capacity must be positive")
        for d in self.demands:
            if d > self.vehicle_capacity:
                raise ValueError("Demand exceeds vehicle capacity")
        if not self.customers:
            return  # Empty instance is valid
        n = len(self.customers) + 1  # +1 for depot
        if len(self.distance_matrix) != n:
            raise ValueError("Distance matrix size must match customers + depot")


@dataclass
class VRPResult:
    """VRP solution result."""
    routes: List[List[int]]
    total_cost: float
    algorithm: str
    num_vehicles: int = 0


class VRPSolver:
    """VRP solver using Clarke-Wright savings algorithm."""

    def __init__(self, algorithm: str = "savings"):
        self.algorithm = algorithm

    def solve(self, instance: VRPInstance) -> VRPResult:
        """Solve VRP instance."""
        n = len(instance.customers)
        if n == 0:
            return VRPResult(routes=[], total_cost=0.0, algorithm=self.algorithm, num_vehicles=0)

        if self.algorithm == "savings":
            return self._savings(instance)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _savings(self, instance: VRPInstance) -> VRPResult:
        """Clarke-Wright savings algorithm."""
        n = len(instance.customers)
        dist = instance.distance_matrix
        demands = instance.demands
        capacity = instance.vehicle_capacity

        # Calculate savings for all pairs
        savings = []
        for i in range(n):
            for j in range(i + 1, n):
                # Saving: s(i,j) = d(0,i) + d(0,j) - d(i,j)
                s = dist[0][i + 1] + dist[0][j + 1] - dist[i + 1][j + 1]
                savings.append((s, i, j))
        savings.sort(reverse=True)

        # Each customer starts in its own route
        routes = [[i] for i in range(n)]
        route_demands = [demands[i] for i in range(n)]
        route_of = list(range(n))  # route_of[customer] = route_index

        for s, i, j in savings:
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue

            # Check if i is at end of its route and j is at end of its route
            route_i = routes[ri]
            route_j = routes[rj]

            # i must be at start or end of route_i
            if route_i[0] != i and route_i[-1] != i:
                continue
            # j must be at start or end of route_j
            if route_j[0] != j and route_j[-1] != j:
                continue

            # Check capacity
            if route_demands[ri] + route_demands[rj] > capacity:
                continue

            # Merge routes
            if route_i[-1] == i and route_j[0] == j:
                # i at end of route_i, j at start of route_j
                new_route = route_i + route_j
            elif route_i[0] == i and route_j[-1] == j:
                # i at start of route_i, j at end of route_j
                new_route = route_j + route_i
            elif route_i[-1] == i and route_j[-1] == j:
                # both at end — reverse route_j
                new_route = route_i + route_j[::-1]
            elif route_i[0] == i and route_j[0] == j:
                # both at start — reverse route_i
                new_route = route_i[::-1] + route_j
            else:
                continue

            # Update routes
            routes[ri] = new_route
            route_demands[ri] += route_demands[rj]
            routes[rj] = []
            for c in new_route:
                route_of[c] = ri

        # Remove empty routes
        routes = [r for r in routes if r]

        # Calculate total cost
        total_cost = 0.0
        for route in routes:
            # depot → first customer
            total_cost += dist[0][route[0] + 1]
            # between customers
            for k in range(len(route) - 1):
                total_cost += dist[route[k] + 1][route[k + 1] + 1]
            # last customer → depot
            total_cost += dist[route[-1] + 1][0]

        return VRPResult(
            routes=routes,
            total_cost=total_cost,
            algorithm="savings",
            num_vehicles=len(routes),
        )

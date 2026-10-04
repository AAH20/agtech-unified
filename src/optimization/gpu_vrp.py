"""GPU-accelerated VRP solver using PyTorch tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUVRPResult:
    """GPU VRP solution result."""

    routes: List[List[int]]
    total_cost: float
    algorithm: str
    device: str


class GPUVRPSolver:
    """GPU-accelerated VRP solver using PyTorch tensors.

    Supports parallel savings computation on GPU.
    """

    def __init__(self, algorithm: str = "savings", device: Optional[str] = None):
        self.algorithm = algorithm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def solve(
        self,
        depot: Tuple[float, float],
        customers: List[Tuple[float, float]],
        demands: List[float],
        vehicle_capacity: float,
        distance_matrix: List[List[float]],
    ) -> GPUVRPResult:
        """Solve VRP instance using GPU acceleration."""
        n = len(customers)
        if n == 0:
            return GPUVRPResult(
                routes=[], total_cost=0.0, algorithm=self.algorithm, device=self.device
            )

        if self.algorithm == "savings":
            routes, total_cost = self._gpu_savings(distance_matrix, demands, vehicle_capacity)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return GPUVRPResult(
            routes=routes, total_cost=total_cost, algorithm=self.algorithm, device=self.device
        )

    def compute_savings_batch(
        self, distance_matrices: List[List[List[float]]]
    ) -> List[torch.Tensor]:
        """Compute savings matrices for multiple VRP instances in batch.

        Args:
            distance_matrices: List of distance matrices (including depot at index 0).

        Returns:
            List of savings tensors, one per instance.
        """
        savings_list = []
        for dist_matrix in distance_matrices:
            dist = torch.tensor(dist_matrix, dtype=torch.float32, device=self.device)
            savings = self._compute_savings_tensor(dist)
            savings_list.append(savings)
        return savings_list

    def _compute_savings_tensor(self, dist: torch.Tensor) -> torch.Tensor:
        """Compute savings matrix using GPU tensor operations.

        s(i,j) = d(0, i+1) + d(0, j+1) - d(i+1, j+1)
        """
        n = dist.shape[0] - 1  # Number of customers (excluding depot)
        if n <= 1:
            return torch.zeros((n, n), dtype=torch.float32, device=self.device)

        # d(0, i) for all customers i (1-indexed in dist matrix)
        d_depot = dist[0, 1:]  # (n,)

        # s(i,j) = d(0,i) + d(0,j) - d(i,j)
        # Using broadcasting: (n, 1) + (1, n) - (n, n)
        savings = d_depot.unsqueeze(1) + d_depot.unsqueeze(0) - dist[1:, 1:]

        # Zero out diagonal (no self-savings)
        mask = torch.eye(n, dtype=torch.bool, device=self.device)
        savings = savings.masked_fill(mask, 0.0)

        return savings

    def _gpu_savings(
        self, distance_matrix: List[List[float]], demands: List[float], capacity: float
    ) -> Tuple[List[List[int]], float]:
        """Clarke-Wright savings algorithm with GPU-computed savings."""
        n = len(demands)
        dist = torch.tensor(distance_matrix, dtype=torch.float32, device=self.device)

        # Compute savings on GPU
        savings = self._compute_savings_tensor(dist)

        # Get sorted savings (descending) - move to CPU for sequential processing
        savings_cpu = savings.cpu().numpy()

        # Create list of (saving, i, j) tuples
        savings_list = []
        for i in range(n):
            for j in range(i + 1, n):
                savings_list.append((savings_cpu[i][j], i, j))
        savings_list.sort(reverse=True)

        # Each customer starts in its own route
        routes = [[i] for i in range(n)]
        route_demands = [demands[i] for i in range(n)]
        route_of = list(range(n))

        for s, i, j in savings_list:
            ri = route_of[i]
            rj = route_of[j]
            if ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check if i is at start or end of its route
            if route_i[0] != i and route_i[-1] != i:
                continue
            # Check if j is at start or end of its route
            if route_j[0] != j and route_j[-1] != j:
                continue

            # Check capacity
            if route_demands[ri] + route_demands[rj] > capacity:
                continue

            # Merge routes
            if route_i[-1] == i and route_j[0] == j:
                new_route = route_i + route_j
            elif route_i[0] == i and route_j[-1] == j:
                new_route = route_j + route_i
            elif route_i[-1] == i and route_j[-1] == j:
                new_route = route_i + route_j[::-1]
            elif route_i[0] == i and route_j[0] == j:
                new_route = route_i[::-1] + route_j
            else:
                continue

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
            total_cost += distance_matrix[0][route[0] + 1]
            for k in range(len(route) - 1):
                total_cost += distance_matrix[route[k] + 1][route[k + 1] + 1]
            total_cost += distance_matrix[route[-1] + 1][0]

        return routes, total_cost

"""GPU-accelerated TSP solver using PyTorch tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUTSPResult:
    """GPU TSP solution result."""

    tour: List[str]
    cost: float
    algorithm: str
    device: str


class GPUTSPSolver:
    """GPU-accelerated TSP solver using PyTorch tensors.

    Supports:
    - Batched distance matrix computation via broadcasting
    - Parallel 2-opt improvement (all moves evaluated simultaneously)
    - GPU-based nearest neighbor heuristic
    """

    def __init__(self, algorithm: str = "nearest_neighbor", device: Optional[str] = None):
        self.algorithm = algorithm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, cities: List[str], distance_matrix: List[List[float]]) -> GPUTSPResult:
        """Solve TSP instance using GPU acceleration."""
        if not cities:
            return GPUTSPResult(tour=[], cost=0.0, algorithm=self.algorithm, device=self.device)

        if len(cities) == 1:
            return GPUTSPResult(
                tour=cities[:], cost=0.0, algorithm=self.algorithm, device=self.device
            )

        dist_tensor = torch.tensor(distance_matrix, dtype=torch.float32, device=self.device)

        if self.algorithm == "nearest_neighbor":
            tour_indices = self._gpu_nearest_neighbor(dist_tensor)
        elif self.algorithm == "two_opt":
            tour_indices = self._gpu_nearest_neighbor(dist_tensor)
            tour_indices = self._gpu_two_opt(dist_tensor, tour_indices)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        cost = self._compute_cost(dist_tensor, tour_indices)
        tour = [cities[i] for i in tour_indices]

        return GPUTSPResult(tour=tour, cost=cost, algorithm=self.algorithm, device=self.device)

    def solve_batch(
        self, instances: List[Tuple[List[str], List[List[float]]]]
    ) -> List[GPUTSPResult]:
        """Solve multiple TSP instances in batch on GPU."""
        return [self.solve(cities, dist) for cities, dist in instances]

    def compute_distance_matrix_batch(
        self, coordinates_batch: List[List[Tuple[float, float]]]
    ) -> torch.Tensor:
        """Compute distance matrices for multiple instances in batch using GPU broadcasting.

        Args:
            coordinates_batch: List of (x, y) coordinate lists, one per instance.

        Returns:
            Tensor of shape (B, N, N) with pairwise Euclidean distances.
        """
        max_n = max(len(coords) for coords in coordinates_batch)
        batch_size = len(coordinates_batch)

        coords_padded = torch.zeros(batch_size, max_n, 2, dtype=torch.float32, device=self.device)
        for i, coords in enumerate(coordinates_batch):
            if coords:
                coords_padded[i, : len(coords)] = torch.tensor(
                    coords, dtype=torch.float32, device=self.device
                )

        # Broadcasting: (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2)
        diff = coords_padded.unsqueeze(2) - coords_padded.unsqueeze(1)
        dist = torch.sqrt((diff**2).sum(dim=-1))

        return dist

    def _gpu_nearest_neighbor(self, dist: torch.Tensor) -> List[int]:
        """GPU-based nearest neighbor heuristic."""
        n = dist.shape[0]
        visited = torch.zeros(n, dtype=torch.bool, device=self.device)
        tour = [0]
        visited[0] = True
        current = 0

        for _ in range(n - 1):
            masked_dist = dist[current].clone()
            masked_dist[visited] = float("inf")
            nearest = torch.argmin(masked_dist).item()
            tour.append(nearest)
            visited[nearest] = True
            current = nearest

        return tour

    def _gpu_two_opt(self, dist: torch.Tensor, tour: List[int]) -> List[int]:
        """Parallel 2-opt improvement on GPU.

        Evaluates all possible 2-opt moves in parallel and applies the best one.
        Repeats until no improving move exists.
        """
        n = len(tour)
        if n <= 3:
            return tour

        tour_tensor = torch.tensor(tour, dtype=torch.long, device=self.device)
        improved = True

        while improved:
            improved = False

            # Generate all (i, j) pairs where i < j
            indices = torch.arange(n, device=self.device)
            i_idx = indices[:-1].unsqueeze(1).expand(n - 1, n - 1)
            j_idx = indices[1:].unsqueeze(0).expand(n - 1, n - 1)
            mask = i_idx < j_idx

            i_vals = i_idx[mask]
            j_vals = j_idx[mask]

            if len(i_vals) == 0:
                break

            # For 2-opt, we replace edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
            # with (tour[i], tour[j]) and (tour[i+1], tour[j+1])
            # This reverses the segment tour[i+1:j+1]
            j_next = (j_vals + 1) % n

            a = tour_tensor[i_vals]
            b = tour_tensor[(i_vals + 1) % n]
            c = tour_tensor[j_vals]
            d = tour_tensor[j_next]

            old_cost = dist[a, b] + dist[c, d]
            new_cost = dist[a, c] + dist[b, d]
            delta = new_cost - old_cost

            best_delta = torch.min(delta).item()

            if best_delta < -1e-6:
                best_idx = torch.argmin(delta).item()
                i_best = i_vals[best_idx].item()
                j_best = j_vals[best_idx].item()

                # Reverse segment [i_best+1, j_best]
                segment = tour_tensor[i_best + 1 : j_best + 1]
                tour_tensor[i_best + 1 : j_best + 1] = segment.flip(0)
                improved = True

        return tour_tensor.tolist()

    def _compute_cost(self, dist: torch.Tensor, tour: List[int]) -> float:
        """Compute total tour cost."""
        if len(tour) <= 1:
            return 0.0

        tour_tensor = torch.tensor(tour, dtype=torch.long, device=self.device)
        next_tensor = torch.roll(tour_tensor, -1)
        cost = dist[tour_tensor, next_tensor].sum().item()
        return cost

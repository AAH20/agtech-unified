"""Simulated annealing solver for the Traveling Salesman Problem."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


class SimulatedAnnealingTSP:
    """Simulated annealing TSP solver.

    Starts from a random tour and iteratively proposes 2-opt moves.
    Worse moves are accepted with probability exp(-delta / T), where T
    decreases geometrically from ``initial_temperature`` to ``final_temperature``.
    """

    def __init__(
        self,
        initial_temperature: float = 100.0,
        final_temperature: float = 0.01,
        cooling_rate: float = 0.9995,
        max_iterations: int = 10000,
        seed: Optional[int] = None,
    ):
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if final_temperature <= 0:
            raise ValueError("final_temperature must be positive")
        if not 0 < cooling_rate < 1:
            raise ValueError("cooling_rate must be in (0, 1)")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.rng = random.Random(seed)

    def solve(self, distance_matrix: List[List[float]]) -> Tuple[List[int], float]:
        """Solve TSP instance. Returns (tour, cost)."""
        n = len(distance_matrix)
        if n == 0:
            return [], 0.0
        if n == 1:
            return [0], 0.0

        # Validate distance matrix
        for i in range(n):
            if len(distance_matrix[i]) != n:
                raise ValueError("Distance matrix must be square")
            for j in range(n):
                if not math.isfinite(distance_matrix[i][j]):
                    raise ValueError(f"Distance matrix contains NaN or Inf at ({i},{j})")
                if distance_matrix[i][j] < 0:
                    raise ValueError(f"Distance matrix contains negative value at ({i},{j})")

        # Start from a random tour
        current_tour = list(range(n))
        self.rng.shuffle(current_tour)
        current_cost = self._tour_cost(current_tour, distance_matrix)

        best_tour = current_tour[:]
        best_cost = current_cost

        temperature = self.initial_temperature

        for _ in range(self.max_iterations):
            if temperature < self.final_temperature:
                break

            # Propose a 2-opt move: pick two positions and reverse the segment
            i = self.rng.randint(0, n - 2)
            j = self.rng.randint(i + 1, n - 1)
            if i == 0 and j == n - 1:
                continue

            # Compute delta for the 2-opt move
            a, b = current_tour[i], current_tour[(i + 1) % n]
            c, d = current_tour[j], current_tour[(j + 1) % n]
            old_cost = distance_matrix[a][b] + distance_matrix[c][d]
            new_cost = distance_matrix[a][c] + distance_matrix[b][d]
            delta = new_cost - old_cost

            if delta < 0 or self.rng.random() < math.exp(-delta / temperature):
                # Accept the move
                current_tour[i + 1 : j + 1] = reversed(current_tour[i + 1 : j + 1])
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = current_tour[:]

            temperature *= self.cooling_rate

        return best_tour, best_cost

    def _tour_cost(self, tour: List[int], dist: List[List[float]]) -> float:
        """Calculate total tour cost."""
        if len(tour) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost += dist[tour[i]][tour[j]]
        return cost

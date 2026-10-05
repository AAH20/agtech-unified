"""Ant Colony Optimization solver for the Traveling Salesman Problem."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


class AntColonyTSP:
    """Ant Colony Optimization TSP solver.

    Pheromone-based constructive metaheuristic where artificial ants
    build tours probabilistically based on pheromone trails and heuristic
    information (inverse distance). Pheromones evaporate over time and
    are reinforced on good tours.
    """

    def __init__(
        self,
        n_ants: int = 20,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        q: float = 100.0,
        seed: Optional[int] = None,
    ):
        if n_ants < 1:
            raise ValueError("n_ants must be at least 1")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if not 0 <= evaporation_rate <= 1:
            raise ValueError("evaporation_rate must be in [0, 1]")
        if q <= 0:
            raise ValueError("q must be positive")

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.rng = random.Random(seed)
        self.pheromone: Optional[List[List[float]]] = None

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

        # Initialize pheromone matrix
        initial_pheromone = 1.0 / (n * n)
        self.pheromone = [[initial_pheromone] * n for _ in range(n)]

        best_tour: List[int] = []
        best_cost = float("inf")

        for _ in range(self.n_iterations):
            all_tours: List[List[int]] = []
            all_costs: List[float] = []

            # Each ant constructs a tour
            for _ in range(self.n_ants):
                tour = self._construct_tour(distance_matrix)
                cost = self._tour_cost(tour, distance_matrix)
                all_tours.append(tour)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour[:]

            # Update pheromones
            self._update_pheromones(all_tours, all_costs, distance_matrix)

        return best_tour, best_cost

    def _construct_tour(self, dist: List[List[float]]) -> List[int]:
        """Construct a tour using probabilistic pheromone-based selection."""
        if self.pheromone is None:
            raise RuntimeError("Pheromone matrix not initialized")
        n = len(dist)
        start = self.rng.randint(0, n - 1)
        tour = [start]
        unvisited = set(range(n)) - {start}

        while unvisited:
            current = tour[-1]
            # Compute selection probabilities
            probabilities: List[Tuple[int, float]] = []
            total = 0.0

            for city in unvisited:
                tau = self.pheromone[current][city] ** self.alpha
                eta = (1.0 / dist[current][city]) ** self.beta if dist[current][city] > 0 else 0.0
                prob = tau * eta
                probabilities.append((city, prob))
                total += prob

            if total == 0:
                # Fallback: uniform random selection
                next_city = self.rng.choice(list(unvisited))
            else:
                # Roulette wheel selection
                r = self.rng.random() * total
                cumulative = 0.0
                next_city = probabilities[-1][0]
                for city, prob in probabilities:
                    cumulative += prob
                    if cumulative >= r:
                        next_city = city
                        break

            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def _update_pheromones(
        self, tours: List[List[int]], costs: List[float], dist: List[List[float]]
    ) -> None:
        """Evaporate and deposit pheromones."""
        n = len(dist)
        if self.pheromone is None:
            return

        # Evaporation
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] *= 1.0 - self.evaporation_rate

        # Deposit new pheromones
        for tour, cost in zip(tours, costs):
            if cost <= 0:
                continue
            deposit = self.q / cost
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                self.pheromone[tour[i]][tour[j]] += deposit
                self.pheromone[tour[j]][tour[i]] += deposit

    def _tour_cost(self, tour: List[int], dist: List[List[float]]) -> float:
        """Calculate total tour cost."""
        if len(tour) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost += dist[tour[i]][tour[j]]
        return cost

"""Genetic Algorithm solver for the Traveling Salesman Problem."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


class GeneticAlgorithmTSP:
    """Genetic Algorithm TSP solver.

    Population-based evolutionary algorithm using:
    - Order Crossover (OX) for recombination
    - Swap mutation for diversification
    - Elitism to preserve the best solutions
    - Tournament selection for parent choice
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.02,
        elitism_count: int = 2,
        seed: Optional[int] = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        if elitism_count < 0:
            raise ValueError("elitism_count must be non-negative")
        if elitism_count >= population_size:
            raise ValueError("elitism_count must be less than population_size")

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.rng = random.Random(seed)
        self.best_cost_history: List[float] = []

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

        # Initialize population with random permutations
        population = [self._random_tour(n) for _ in range(self.population_size)]
        fitness = [self._tour_cost(tour, distance_matrix) for tour in population]

        best_tour = population[fitness.index(min(fitness))][:]
        best_cost = min(fitness)
        self.best_cost_history = [best_cost]

        for _ in range(self.generations):
            new_population: List[List[int]] = []

            # Elitism: carry forward the best individuals
            sorted_indices = sorted(range(self.population_size), key=lambda i: fitness[i])
            for i in range(min(self.elitism_count, self.population_size)):
                new_population.append(population[sorted_indices[i]][:])

            # Fill the rest with offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)

                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self._ox_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            fitness = [self._tour_cost(tour, distance_matrix) for tour in population]

            gen_best = min(fitness)
            if gen_best < best_cost:
                best_cost = gen_best
                best_tour = population[fitness.index(gen_best)][:]
            self.best_cost_history.append(best_cost)

        return best_tour, best_cost

    def _random_tour(self, n: int) -> List[int]:
        """Generate a random permutation of [0, n)."""
        tour = list(range(n))
        self.rng.shuffle(tour)
        return tour

    def _tournament_select(
        self, population: List[List[int]], fitness: List[float], tournament_size: int = 3
    ) -> List[int]:
        """Tournament selection: pick the best from a random subset."""
        candidates = self.rng.sample(range(len(population)), min(tournament_size, len(population)))
        best = min(candidates, key=lambda i: fitness[i])
        return population[best]

    def _ox_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order Crossover (OX): preserves relative order of cities."""
        n = len(parent1)
        if n <= 1:
            return parent1[:], parent2[:]

        # Select two cut points
        cut1 = self.rng.randint(0, n - 2)
        cut2 = self.rng.randint(cut1 + 1, n - 1)

        def ox_single(p1: List[int], p2: List[int]) -> List[int]:
            # Copy segment from p1
            child = [-1] * n
            child[cut1 : cut2 + 1] = p1[cut1 : cut2 + 1]
            used = set(child[cut1 : cut2 + 1])

            # Fill remaining positions with cities from p2 in order
            idx = (cut2 + 1) % n
            for i in range(n):
                p2_idx = (cut2 + 1 + i) % n
                city = p2[p2_idx]
                if city not in used:
                    child[idx] = city
                    used.add(city)
                    idx = (idx + 1) % n
            return child

        return ox_single(parent1, parent2), ox_single(parent2, parent1)

    def _mutate(self, tour: List[int]) -> List[int]:
        """Swap mutation: swap two random positions."""
        if self.rng.random() >= self.mutation_rate:
            return tour
        n = len(tour)
        if n < 2:
            return tour
        i, j = self.rng.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def _tour_cost(self, tour: List[int], dist: List[List[float]]) -> float:
        """Calculate total tour cost."""
        if len(tour) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost += dist[tour[i]][tour[j]]
        return cost

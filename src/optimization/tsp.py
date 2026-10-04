"""TSP solver with Christofides algorithm and nearest-neighbor heuristic."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TSPInstance:
    """TSP problem instance with cities and distance matrix."""

    cities: List[str]
    distance_matrix: List[List[float]]
    validate_metric: bool = False

    def __post_init__(self):
        n = len(self.cities)
        if len(self.distance_matrix) != n:
            raise ValueError("Distance matrix must be square")
        for row in self.distance_matrix:
            if len(row) != n:
                raise ValueError("Distance matrix must be square")
        # Check for NaN/Inf
        for i in range(n):
            for j in range(n):
                if not math.isfinite(self.distance_matrix[i][j]):
                    raise ValueError(f"Distance matrix contains NaN or Inf at ({i},{j})")
        # Check non-negative
        for i in range(n):
            for j in range(n):
                if self.distance_matrix[i][j] < 0:
                    raise ValueError(f"Distance matrix contains negative value at ({i},{j})")
        # Check metric property (triangle inequality) — optional
        if self.validate_metric:
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if (
                            self.distance_matrix[i][j]
                            > self.distance_matrix[i][k] + self.distance_matrix[k][j] + 1e-9
                        ):
                            logger.warning(
                                f"Non-Metric TSP: d({i},{j})={self.distance_matrix[i][j]:.2f} > "
                                f"d({i},{k})+d({k},{j})="
                                f"{self.distance_matrix[i][k] + self.distance_matrix[k][j]:.2f}"
                            )
                            return


@dataclass
class TSPResult:
    """TSP solution result."""

    tour: List[str]
    cost: float
    algorithm: str
    approximation_ratio: Optional[float] = None


class TSPSolver:
    """TSP solver supporting multiple algorithms."""

    def __init__(self, algorithm: str = "christofides", improve: bool = True):
        self.algorithm = algorithm
        self.improve = improve

    def solve(self, instance: TSPInstance) -> TSPResult:
        """Solve TSP instance."""
        # Input validation: check for NaN/Inf and negative distances
        n = len(instance.cities)
        for i in range(n):
            for j in range(n):
                if not math.isfinite(instance.distance_matrix[i][j]):
                    raise ValueError(f"Distance matrix contains NaN or Inf at ({i},{j})")
                if instance.distance_matrix[i][j] < 0:
                    raise ValueError(f"Distance matrix contains negative value at ({i},{j})")

        if not instance.cities:
            return TSPResult(tour=[], cost=0.0, algorithm=self.algorithm)

        if len(instance.cities) == 1:
            return TSPResult(tour=instance.cities[:], cost=0.0, algorithm=self.algorithm)

        if self.algorithm == "christofides":
            result = self._christofides(instance)
        elif self.algorithm == "nearest_neighbor":
            result = self._nearest_neighbor(instance)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if self.improve:
            from src.optimization.local_search import TwoOpt

            tour_indices = [instance.cities.index(c) for c in result.tour]
            improved = TwoOpt().improve(tour_indices, instance.distance_matrix)
            new_cost = self._tour_cost(improved, instance.distance_matrix)
            result = TSPResult(
                tour=[instance.cities[i] for i in improved],
                cost=new_cost,
                algorithm=result.algorithm,
                approximation_ratio=result.approximation_ratio,
            )

        return result

    def _christofides(self, instance: TSPInstance) -> TSPResult:
        """Christofides algorithm: 1.5-approximation for metric TSP.

        Note: Uses greedy matching, not optimal Edmonds' blossom algorithm.
        The 1.5-approximation guarantee requires optimal matching, so
        approximation_ratio is set to None.
        """
        n = len(instance.cities)
        if n <= 2:
            tour = instance.cities[:]
            cost = self._tour_cost(list(range(n)), instance.distance_matrix)
            return TSPResult(tour=tour, cost=cost, algorithm="christofides")

        # Step 1: Minimum Spanning Tree (Prim's algorithm)
        mst_edges = self._prim_mst(instance.distance_matrix, n)

        # Step 2: Find odd-degree vertices in MST
        degree = [0] * n
        for u, v in mst_edges:
            degree[u] += 1
            degree[v] += 1
        odd_vertices = [i for i in range(n) if degree[i] % 2 == 1]

        # Step 3: Minimum-weight perfect matching on odd vertices
        matching = self._min_weight_matching(instance.distance_matrix, odd_vertices)

        # Step 4: Combine MST and matching to form Eulerian multigraph
        multigraph = mst_edges + matching

        # Step 5: Find Eulerian tour
        eulerian_tour = self._eulerian_tour(multigraph, n)

        # Step 6: Shortcut to remove duplicates (make Hamiltonian)
        visited = [False] * n
        tour_indices = []
        for v in eulerian_tour:
            if not visited[v]:
                visited[v] = True
                tour_indices.append(v)

        tour = [instance.cities[i] for i in tour_indices]
        cost = self._tour_cost(tour_indices, instance.distance_matrix)
        return TSPResult(tour=tour, cost=cost, algorithm="christofides", approximation_ratio=None)

    def _nearest_neighbor(self, instance: TSPInstance) -> TSPResult:
        """Nearest neighbor heuristic for TSP."""
        n = len(instance.cities)
        if n <= 2:
            tour = instance.cities[:]
            cost = self._tour_cost(list(range(n)), instance.distance_matrix)
            return TSPResult(tour=tour, cost=cost, algorithm="nearest_neighbor")

        unvisited = set(range(1, n))
        tour_indices = [0]
        current = 0

        while unvisited:
            nearest = min(unvisited, key=lambda j: instance.distance_matrix[current][j])
            tour_indices.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        tour = [instance.cities[i] for i in tour_indices]
        cost = self._tour_cost(tour_indices, instance.distance_matrix)
        return TSPResult(tour=tour, cost=cost, algorithm="nearest_neighbor")

    def _prim_mst(self, dist: List[List[float]], n: int) -> List[Tuple[int, int]]:
        """Prim's algorithm for MST."""
        in_mst = [False] * n
        key = [float("inf")] * n
        parent = [-1] * n
        key[0] = 0.0

        edges = []
        for _ in range(n):
            u = min((i for i in range(n) if not in_mst[i]), key=lambda i: key[i])
            in_mst[u] = True
            if parent[u] != -1:
                edges.append((parent[u], u))

            for v in range(n):
                if not in_mst[v] and dist[u][v] < key[v]:
                    key[v] = dist[u][v]
                    parent[v] = u

        return edges

    def _min_weight_matching(
        self, dist: List[List[float]], odd_vertices: List[int]
    ) -> List[Tuple[int, int]]:
        """Greedy minimum-weight perfect matching (not optimal but fast)."""
        n = len(odd_vertices)
        if n == 0:
            return []

        # Sort all pairs by distance
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((dist[odd_vertices[i]][odd_vertices[j]], i, j))
        pairs.sort()

        matched = [False] * n
        matching = []
        for d, i, j in pairs:
            if not matched[i] and not matched[j]:
                matched[i] = True
                matched[j] = True
                matching.append((odd_vertices[i], odd_vertices[j]))

        return matching

    def _eulerian_tour(self, edges: List[Tuple[int, int]], n: int) -> List[int]:
        """Find Eulerian tour using Hierholzer's algorithm."""
        from collections import defaultdict

        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # Start from the first vertex that has edges
        start = 0
        for v in range(n):
            if graph[v]:
                start = v
                break

        stack = [start]
        tour = []
        while stack:
            v = stack[-1]
            if graph[v]:
                u = graph[v].pop()
                graph[u].remove(v)
                stack.append(u)
            else:
                tour.append(stack.pop())

        return tour[::-1]

    def _tour_cost(self, tour_indices: List[int], dist: List[List[float]]) -> float:
        """Calculate total tour cost from city indices."""
        if len(tour_indices) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(tour_indices)):
            j = (i + 1) % len(tour_indices)
            cost += dist[tour_indices[i]][tour_indices[j]]
        return cost

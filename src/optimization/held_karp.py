"""Held-Karp exact TSP solver: O(n^2 * 2^n) dynamic programming.

Solves the traveling salesman problem exactly using the Held-Karp
subset DP. Suitable for small instances (n <= ~15-20); the exponential
state space makes larger instances infeasible, so ``max_cities`` guards
against accidental blowup.
"""

from __future__ import annotations

import math
from typing import List, Tuple


class HeldKarp:
    """Exact TSP solver via the Held-Karp algorithm.

    dp[mask][j] = minimum cost of a path that starts at city 0, visits
    exactly the cities in ``mask`` (which must contain 0 and j), and ends
    at city j. The optimal tour closes by returning from the last city
    to city 0.
    """

    def __init__(self, max_cities: int = 15):
        self.max_cities = max_cities

    def solve(self, distance_matrix: List[List[float]]) -> Tuple[float, List[int]]:
        """Solve TSP exactly. Returns (optimal_cost, tour).

        The tour starts and ends at city 0 (returned as a path with the
        implicit return edge included in the cost).
        """
        n = len(distance_matrix)
        if n == 0:
            return 0.0, []
        if any(len(row) != n for row in distance_matrix):
            raise ValueError("Distance matrix must be square")
        if n > self.max_cities:
            raise ValueError(
                f"n={n} exceeds max_cities={self.max_cities}: "
                "Held-Karp is O(n^2 * 2^n) and infeasible at this size"
            )
        for i in range(n):
            for j in range(n):
                v = distance_matrix[i][j]
                if not math.isfinite(v):
                    raise ValueError(f"Distance matrix contains NaN or Inf at ({i},{j})")
                if v < 0:
                    raise ValueError(f"Distance matrix contains negative value at ({i},{j})")

        if n == 1:
            return 0.0, [0]
        if n == 2:
            return 2.0 * distance_matrix[0][1], [0, 1]

        size = 1 << n
        full = size - 1
        # dp[mask] maps end-city -> min path cost; only masks containing
        # bit 0 are populated.
        dp: List[dict] = [dict() for _ in range(size)]
        parent: List[dict] = [dict() for _ in range(size)]
        dp[1][0] = 0.0

        for mask in range(1, size):
            if not (mask & 1):
                continue
            d = dp[mask]
            if not d:
                continue
            for j, cost_j in d.items():
                # Extend the path to every city k not yet in mask.
                rem = full & ~mask
                m = rem
                dist_j = distance_matrix[j]
                while m:
                    bit = m & (-m)
                    m ^= bit
                    k = bit.bit_length() - 1
                    new_mask = mask | bit
                    new_cost = cost_j + dist_j[k]
                    dk = dp[new_mask]
                    if new_cost < dk.get(k, math.inf):
                        dk[k] = new_cost
                        parent[new_mask][k] = j

        # Close the tour: return to city 0 from the last city.
        best_cost = math.inf
        last = -1
        for j, cost_j in dp[full].items():
            total = cost_j + distance_matrix[j][0]
            if total < best_cost:
                best_cost = total
                last = j

        # Reconstruct the path 0 -> ... -> last by walking parents back.
        tour = [0] * n
        mask = full
        j = last
        idx = n - 1
        while j != 0:
            tour[idx] = j
            idx -= 1
            pj = parent[mask][j]
            mask ^= 1 << j
            j = pj
        tour[0] = 0
        return best_cost, tour

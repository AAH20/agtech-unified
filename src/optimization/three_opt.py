"""3-opt local search improvement for TSP tours."""

from __future__ import annotations

from typing import List, Tuple


class ThreeOpt:
    """3-opt local search: removes 3 edges and reconnects segments.

    Considers all 7 non-identity reconnections of 3-edge cuts.
    Cases 1-3 (reversing one or two segments) are equivalent to 2-opt moves.
    Cases 4-7 (swapping segments) are the true 3-opt moves that 2-opt cannot reach.
    """

    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = max_iterations

    def improve(self, tour: List[int], distance_matrix: List[List[float]]) -> List[int]:
        """Apply 3-opt improvement. Returns improved tour (closed cycle)."""
        if len(tour) <= 2:
            return tour[:]

        n = len(tour)
        # Validate: no duplicate cities
        if len(set(tour)) != n:
            raise ValueError("Tour contains duplicate cities")
        # Validate: all indices within range
        for idx in tour:
            if idx < 0 or idx >= len(distance_matrix):
                raise ValueError(
                    f"Tour index {idx} exceeds distance matrix size {len(distance_matrix)}"
                )
        # Validate: distance matrix is large enough
        if len(distance_matrix) < n:
            raise ValueError(
                f"Distance matrix size {len(distance_matrix)} does not match tour length {n}"
            )

        improved = True
        iterations = 0
        current = tour[:]

        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        if i == 0 and k == n - 1:
                            continue
                        best_case, best_delta = self._best_move(current, i, j, k, distance_matrix)
                        if best_case > 0:
                            self._apply_move(current, i, j, k, best_case)
                            improved = True

        return current

    def _best_move(
        self,
        tour: List[int],
        i: int,
        j: int,
        k: int,
        dist: List[List[float]],
    ) -> Tuple[int, float]:
        """Return (case_number, delta) for the best 3-opt move, or (0, 0) if none improves."""
        n = len(tour)
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        e, f = tour[k], tour[(k + 1) % n]

        old_cost = dist[a][b] + dist[c][d] + dist[e][f]

        # All 7 non-identity reconnections
        candidates = [
            (1, dist[a][c] + dist[b][d] + dist[e][f]),  # A B' C D
            (2, dist[a][b] + dist[c][e] + dist[d][f]),  # A B C' D
            (3, dist[a][c] + dist[b][e] + dist[d][f]),  # A B' C' D
            (4, dist[a][d] + dist[e][b] + dist[c][f]),  # A C B D
            (5, dist[a][d] + dist[e][c] + dist[b][f]),  # A C B' D
            (6, dist[a][e] + dist[d][b] + dist[c][f]),  # A C' B D
            (7, dist[a][e] + dist[d][c] + dist[b][f]),  # A C' B' D
        ]

        best_case = 0
        best_delta = -1e-9
        for case, new_cost in candidates:
            delta = new_cost - old_cost
            if delta < best_delta:
                best_delta = delta
                best_case = case

        return best_case, best_delta

    def _apply_move(self, tour: List[int], i: int, j: int, k: int, case: int) -> None:
        """Apply a 3-opt move in-place."""
        if case == 1:
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
        elif case == 2:
            tour[j + 1 : k + 1] = reversed(tour[j + 1 : k + 1])
        elif case == 3:
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            tour[j + 1 : k + 1] = reversed(tour[j + 1 : k + 1])
        elif case == 4:
            tour[i + 1 : k + 1] = tour[j + 1 : k + 1] + tour[i + 1 : j + 1]
        elif case == 5:
            tour[i + 1 : k + 1] = tour[j + 1 : k + 1] + list(reversed(tour[i + 1 : j + 1]))
        elif case == 6:
            tour[i + 1 : k + 1] = list(reversed(tour[j + 1 : k + 1])) + tour[i + 1 : j + 1]
        elif case == 7:
            tour[i + 1 : k + 1] = list(reversed(tour[j + 1 : k + 1])) + list(
                reversed(tour[i + 1 : j + 1])
            )

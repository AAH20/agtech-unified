"""2-opt local search improvement for TSP tours."""

from __future__ import annotations

from typing import List


class TwoOpt:
    """2-opt local search: removes crossing edges by reversing tour segments.

    When ``use_dont_look_bits`` is True, a per-city "don't-look" bit skips
    cities for which no improving move was found in the last full scan.
    Only the four endpoints of an accepted move can gain new improving
    moves, so re-activating exactly those keeps the result identical to
    plain 2-opt while avoiding useless scans.
    """

    def __init__(self, max_iterations: int = 1000, use_dont_look_bits: bool = True):
        self.max_iterations = max_iterations
        self.use_dont_look_bits = use_dont_look_bits

    def improve(self, tour: List[int], distance_matrix: List[List[float]]) -> List[int]:
        """Apply 2-opt improvement. Returns improved tour (closed cycle)."""
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

        if self.use_dont_look_bits:
            return self._improve_dont_look(tour, distance_matrix)
        return self._improve_plain(tour, distance_matrix)

    def _improve_plain(self, tour: List[int], distance_matrix: List[List[float]]) -> List[int]:
        """Classic first-improvement 2-opt scan."""
        n = len(tour)
        improved = True
        iterations = 0
        current = tour[:]

        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = current[i], current[(i + 1) % n]
                    c, d = current[j], current[(j + 1) % n]
                    old_cost = distance_matrix[a][b] + distance_matrix[c][d]
                    new_cost = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_cost < old_cost - 1e-9:
                        current[i + 1 : j + 1] = reversed(current[i + 1 : j + 1])
                        improved = True
        return current

    def _improve_dont_look(self, tour: List[int], distance_matrix: List[List[float]]) -> List[int]:
        """2-opt with don't-look bits (identical result, fewer scans)."""
        n = len(tour)
        improved = True
        iterations = 0
        current = tour[:]
        # active[city] == False means: no improving 2-opt move exists with this
        # city as the first endpoint, so skip it until an accepted move touches it.
        active = [True] * n

        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            for i in range(n - 1):
                if not active[current[i]]:
                    continue
                found = False
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = current[i], current[(i + 1) % n]
                    c, d = current[j], current[(j + 1) % n]
                    old_cost = distance_matrix[a][b] + distance_matrix[c][d]
                    new_cost = distance_matrix[a][c] + distance_matrix[b][d]
                    if new_cost < old_cost - 1e-9:
                        current[i + 1 : j + 1] = reversed(current[i + 1 : j + 1])
                        # Only the four endpoints of the removed edges can gain
                        # new improving moves.
                        active[a] = True
                        active[b] = True
                        active[c] = True
                        active[d] = True
                        improved = True
                        found = True
                if not found:
                    active[current[i]] = False
        return current

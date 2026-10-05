"""Lin-Kernighan heuristic for the TSP.

A k-opt local search with dynamic k: starting from a tour, it builds a
sequential alternating chain of edge removals and additions, extending
the chain while the cumulative gain stays positive and keeping the best
closing move found. Because k is chosen dynamically (2-opt, 3-opt, ...
up to whatever depth yields improvement), LK escapes local optima that
fixed-depth 2-opt/3-opt cannot.

To escape local optima that LK itself cannot reach, a double-bridge
4-opt perturbation is applied between restarts.
"""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple


class LinKernighan:
    """Lin-Kernighan TSP heuristic.

    Args:
        candidate_set_size: number of nearest-neighbour candidates per city
            used when choosing edges to add (classic LK uses ~5).
        max_iterations: cap on the total number of improvement passes.
        max_chain_depth: cap on the k-opt chain depth (default: n).
        perturbation: enable double-bridge kicks between restarts.
        restarts: number of perturbation restarts after convergence.
        seed: RNG seed for reproducible perturbations.
    """

    def __init__(
        self,
        candidate_set_size: int = 5,
        max_iterations: int = 100,
        max_chain_depth: Optional[int] = None,
        perturbation: bool = True,
        restarts: int = 3,
        seed: Optional[int] = None,
        max_depth_budget: int = 20000,
    ):
        self.candidate_set_size = candidate_set_size
        self.max_iterations = max_iterations
        self.max_chain_depth = max_chain_depth
        self.perturbation = perturbation
        self.restarts = restarts
        self.max_depth_budget = max_depth_budget
        self.rng = random.Random(seed)

    def improve(self, tour: List[int], distance_matrix: List[List[float]]) -> List[int]:
        """Improve ``tour`` in place-style (returns a new tour).

        The result is always a valid permutation of the input and is never
        worse than the input tour.
        """
        n = len(tour)
        for idx in tour:
            if idx < 0 or idx >= len(distance_matrix):
                raise ValueError(
                    f"Tour index {idx} exceeds distance matrix size {len(distance_matrix)}"
                )
        if n <= 2:
            return tour[:]
        if self.max_iterations <= 0:
            return tour[:]
        if len(set(tour)) != n:
            raise ValueError("Tour contains duplicate cities")
        if len(distance_matrix) < n:
            raise ValueError(
                f"Distance matrix size {len(distance_matrix)} does not match tour length {n}"
            )

        candidates = self._build_candidates(distance_matrix, n)
        max_depth = self.max_chain_depth if self.max_chain_depth is not None else n

        current = tour[:]
        best = tour[:]
        best_cost = self._tour_cost(best, distance_matrix)

        passes = 0
        for restart in range(self.restarts + 1):
            improved = True
            while improved and passes < self.max_iterations:
                improved = False
                move = self._lk_pass(current, distance_matrix, candidates, max_depth)
                while move is not None and passes < self.max_iterations:
                    improved = True
                    passes += 1
                    current = move
                    move = self._lk_pass(
                        current, distance_matrix, candidates, max_depth, one_pass=True
                    )
                if not improved:
                    break
            cost = self._tour_cost(current, distance_matrix)
            if cost < best_cost - 1e-12:
                best, best_cost = current[:], cost
            if self.perturbation and restart < self.restarts:
                current = self._double_bridge(best)
                passes = 0  # each restart gets its own improvement budget
        return best

    # ------------------------------------------------------------------
    # Passes
    # ------------------------------------------------------------------
    def _lk_pass(
        self,
        tour: List[int],
        dist: List[List[float]],
        candidates: List[List[int]],
        max_depth: int,
        one_pass: bool = False,
    ) -> Optional[List[int]]:
        """Run LK improvement. Returns a better tour, or None if local optimum.

        With ``one_pass`` a single sweep over all starting cities is done;
        otherwise sweeps repeat until a full sweep yields no improvement.
        """
        n = len(tour)
        current = tour[:]
        improved_any = False
        while True:
            succ = {current[i]: current[(i + 1) % n] for i in range(n)}
            pred = {current[i]: current[(i - 1) % n] for i in range(n)}
            improved = False
            for t1 in range(n):
                move = self._try_improve_from(t1, succ, pred, candidates, dist, n, max_depth)
                if move is not None:
                    current = move
                    improved = True
                    improved_any = True
                    break  # restart sweep with the new tour
            if one_pass:
                return current if improved else None
            if not improved:
                return current if improved_any else None

    # ------------------------------------------------------------------
    # Chain search
    # ------------------------------------------------------------------
    def _try_improve_from(
        self,
        t1: int,
        succ: dict,
        pred: dict,
        candidates: List[List[int]],
        dist: List[List[float]],
        n: int,
        max_depth: int,
    ) -> Optional[List[int]]:
        """Search all LK chains starting at city ``t1``.

        Returns the improved tour from the best move found, or None.
        """
        best_gain = 0.0
        best_move: Optional[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]] = None
        budget = [self.max_depth_budget]

        def check_closing(end: int, gain: float, removed: set, added: set) -> None:
            nonlocal best_gain, best_move
            close_edge = (end, t1)
            rev = (t1, end)
            if close_edge in removed or rev in removed:
                return
            if close_edge in added or rev in added:
                return
            if succ[end] == t1 or succ[t1] == end:
                return
            close_gain = gain - dist[end][t1]
            if close_gain > best_gain:
                best_gain = close_gain
                best_move = (set(removed), set(added) | {close_edge})

        def explore(end: int, gain: float, removed: set, added: set, depth: int) -> None:
            if budget[0] <= 0:
                return
            budget[0] -= 1
            # Choose the next tour edge to remove: X = (end, t_next).
            for t_next in (succ[end], pred[end]):
                if t_next == t1:
                    continue  # closing must stay feasible
                x = (end, t_next) if succ[end] == t_next else (t_next, end)
                if x in added or (x[1], x[0]) in added:
                    continue  # removed/added must stay disjoint
                new_gain = gain + dist[x[0]][x[1]]
                removed.add(x)
                check_closing(t_next, new_gain, removed, added)
                if depth < max_depth:
                    # Choose the next edge to add: Y = (t_next, t5).
                    for t5 in candidates[t_next]:
                        if t5 == t_next:
                            continue
                        y = (t_next, t5)
                        if y in removed or (y[1], y[0]) in removed:
                            continue
                        if succ[t_next] == t5 or pred[t_next] == t5:
                            continue  # Y must not already be a tour edge
                        g2 = new_gain - dist[t_next][t5]
                        if g2 <= 0:
                            continue  # partial gain must stay positive
                        added.add(y)
                        explore(t5, g2, removed, added, depth + 1)
                        added.discard(y)
                removed.discard(x)

        for t2 in (succ[t1], pred[t1]):
            x1 = (t1, t2) if succ[t1] == t2 else (t2, t1)
            removed = {x1}
            base = dist[x1[0]][x1[1]]
            for t3 in candidates[t2]:
                if t3 == t1:
                    continue
                if succ[t2] == t3 or pred[t2] == t3:
                    continue  # Y1 must not be a tour edge
                g1 = base - dist[t2][t3]
                if g1 <= 0:
                    continue  # partial gain must stay positive
                added = {(t2, t3)}
                explore(t3, g1, removed, added, 1)

        if best_move is None:
            return None
        return self._apply_move(best_move, succ, pred, n)

    def _apply_move(
        self,
        move: Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]],
        succ: dict,
        pred: dict,
        n: int,
    ) -> Optional[List[int]]:
        """Apply a (removed, added) edge exchange and return the new tour.

        Validates that the result is a single Hamiltonian cycle; returns
        None if the move is degenerate (safety net for the chain rules).
        """
        removed, added = move
        new_succ = dict(succ)
        for u, v in removed:
            if new_succ.get(u) == v:
                del new_succ[u]
        for u, v in added:
            new_succ[u] = v
        # Walk the cycle from 0; a valid move visits every city exactly once.
        tour = [0]
        seen = {0}
        cur = 0
        while True:
            nxt = new_succ.get(cur)
            if nxt is None or nxt == 0:
                break
            if nxt in seen:
                return None
            seen.add(nxt)
            tour.append(nxt)
            cur = nxt
        if len(tour) != n or 0 not in new_succ:
            return None
        return tour

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------
    def _double_bridge(self, tour: List[int]) -> List[int]:
        """4-opt double-bridge kick: A B C D -> A D C B.

        This is the smallest k-change that LK cannot decompose into
        improving sequential moves, so it escapes LK local optima.
        """
        n = len(tour)
        if n < 4:
            return tour[:]
        a, b, c = sorted(self.rng.sample(range(1, n), 3))
        return tour[:a] + tour[c:] + tour[b:c] + tour[a:b]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_candidates(self, dist: List[List[float]], n: int) -> List[List[int]]:
        """k nearest neighbours per city, sorted by distance."""
        k = min(self.candidate_set_size, n - 1)
        candidates = []
        for i in range(n):
            order = sorted((j for j in range(n) if j != i), key=lambda j: dist[i][j])
            candidates.append(order[:k])
        return candidates

    @staticmethod
    def _tour_cost(tour: List[int], dist: List[List[float]]) -> float:
        n = len(tour)
        if n <= 1:
            return 0.0
        return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))

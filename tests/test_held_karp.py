"""Tests for the Held-Karp exact TSP solver (O(n^2 * 2^n) DP)."""

import math

import pytest

from src.optimization.held_karp import HeldKarp


def _tour_cost(tour, dist):
    if len(tour) <= 1:
        return 0.0
    return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def _is_valid_tour(tour, n):
    return sorted(tour) == list(range(n))


# ======================================================================
# Edge cases
# ======================================================================
class TestHeldKarpEdgeCases:
    def test_empty(self):
        """Empty instance: cost 0, empty tour."""
        hk = HeldKarp()
        cost, tour = hk.solve([])
        assert cost == 0.0
        assert tour == []

    def test_single_city(self):
        """Single city: cost 0, tour [0]."""
        hk = HeldKarp()
        cost, tour = hk.solve([[0]])
        assert cost == 0.0
        assert tour == [0]

    def test_two_cities(self):
        """Two cities: cost is twice the single edge (round trip)."""
        hk = HeldKarp()
        cost, tour = hk.solve([[0, 5], [5, 0]])
        assert cost == 10.0
        assert _is_valid_tour(tour, 2)

    def test_non_square_matrix_raises(self):
        """Non-square distance matrix raises ValueError."""
        hk = HeldKarp()
        with pytest.raises(ValueError, match="square"):
            hk.solve([[0, 1, 2], [1, 0, 2]])

    def test_negative_distance_raises(self):
        """Negative distances raise ValueError."""
        hk = HeldKarp()
        with pytest.raises(ValueError, match="negative"):
            hk.solve([[0, -1], [-1, 0]])

    def test_nan_distance_raises(self):
        """NaN distances raise ValueError."""
        hk = HeldKarp()
        with pytest.raises(ValueError, match="NaN|Inf"):
            hk.solve([[0, float("nan")], [float("nan"), 0]])

    def test_inf_distance_raises(self):
        """Inf distances raise ValueError."""
        hk = HeldKarp()
        with pytest.raises(ValueError, match="NaN|Inf"):
            hk.solve([[0, float("inf")], [float("inf"), 0]])

    def test_too_many_cities_raises(self):
        """n > max_cities raises ValueError (exponential blowup guard)."""
        hk = HeldKarp(max_cities=10)
        n = 11
        dist = [[0.0] * n for _ in range(n)]
        with pytest.raises(ValueError, match="max_cities|too many|exceeds"):
            hk.solve(dist)


# ======================================================================
# Correctness on known instances
# ======================================================================
class TestHeldKarpCorrectness:
    def test_known_4_city_instance(self):
        """Optimal tour cost on a small known instance."""
        hk = HeldKarp()
        dist = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        cost, tour = hk.solve(dist)
        assert _is_valid_tour(tour, 4)
        assert abs(cost - 80.0) < 1e-9  # 0-1-3-2-0 = 10+25+30+15

    def test_line_metric(self):
        """Cities on a line: optimal tour is 2 * (max - min)."""
        hk = HeldKarp()
        positions = [0.0, 3.0, 7.0, 12.0, 20.0]
        n = len(positions)
        dist = [[abs(positions[i] - positions[j]) for j in range(n)] for i in range(n)]
        cost, tour = hk.solve(dist)
        assert _is_valid_tour(tour, n)
        assert abs(cost - 40.0) < 1e-9  # 2 * (20 - 0)

    def test_equilateral_triangle(self):
        """Three equilateral cities: optimal tour is 3 * side."""
        hk = HeldKarp()
        dist = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        cost, tour = hk.solve(dist)
        assert abs(cost - 3.0) < 1e-9
        assert _is_valid_tour(tour, 3)

    def test_asymmetric_instance(self):
        """Held-Karp works on asymmetric instances (ATSP)."""
        hk = HeldKarp()
        dist = [
            [0, 2, 9],
            [1, 0, 6],
            [15, 7, 0],
        ]
        cost, tour = hk.solve(dist)
        assert _is_valid_tour(tour, 3)
        # best tour: 0->1->2->0 = 2+6+15 = 23
        assert abs(cost - 17.0) < 1e-9

    def test_reported_cost_matches_tour(self):
        """The reported cost equals the actual cost of the returned tour."""
        hk = HeldKarp()
        dist = [
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0],
        ]
        cost, tour = hk.solve(dist)
        assert abs(cost - _tour_cost(tour, dist)) < 1e-9

    def test_optimal_cost_matches_brute_force(self):
        """Held-Karp cost matches brute-force enumeration on random instances."""
        import itertools
        import random

        rng = random.Random(42)
        for n in (5, 6, 7):
            dist = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist[i][j] = rng.uniform(1, 50)
            hk = HeldKarp()
            cost, _ = hk.solve(dist)
            best = min(_tour_cost([0] + list(p), dist) for p in itertools.permutations(range(1, n)))
            assert abs(cost - best) < 1e-6


# ======================================================================
# Configuration
# ======================================================================
class TestHeldKarpConfig:
    def test_max_cities_boundary_allowed(self):
        """n == max_cities is allowed."""
        hk = HeldKarp(max_cities=12)
        n = 12
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = abs(i - j)
        cost, tour = hk.solve(dist)
        assert _is_valid_tour(tour, n)
        assert cost > 0

    def test_default_max_cities(self):
        """Default max_cities is at least 15."""
        hk = HeldKarp()
        assert hk.max_cities >= 15

    def test_zero_distances_on_diagonal(self):
        """Diagonal must be zero (or at least finite); off-diagonal zeros allowed."""
        hk = HeldKarp()
        dist = [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
        cost, tour = hk.solve(dist)
        assert _is_valid_tour(tour, 3)
        assert cost >= 0.0

    def test_large_finite_values(self):
        """Large but finite distances are handled."""
        hk = HeldKarp()
        dist = [[0, 1e6], [1e6, 0]]
        cost, _ = hk.solve(dist)
        assert math.isfinite(cost)
        assert abs(cost - 2e6) < 1e-3

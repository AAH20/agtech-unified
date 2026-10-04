"""Tests for simulated annealing TSP solver."""

import math

import pytest

from src.optimization.simulated_annealing import SimulatedAnnealingTSP


def _euclidean_dist(coords):
    n = len(coords)
    return [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(n)
        ]
        for i in range(n)
    ]


def test_sa_basic_square():
    """SA finds the optimal tour on a 4-city square."""
    sa = SimulatedAnnealingTSP(seed=42, max_iterations=5000)
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    tour, cost = sa.solve(dist)
    assert len(tour) == 4
    assert set(tour) == {0, 1, 2, 3}
    assert cost == pytest.approx(4.0)


def test_sa_valid_tour():
    """SA returns a valid permutation of all cities."""
    sa = SimulatedAnnealingTSP(seed=123, max_iterations=3000)
    coords = [(0, 0), (3, 0), (3, 4), (0, 4), (1, 1)]
    dist = _euclidean_dist(coords)
    tour, cost = sa.solve(dist)
    assert len(tour) == 5
    assert set(tour) == {0, 1, 2, 3, 4}
    assert cost > 0


def test_sa_cost_matches_tour():
    """SA-reported cost matches the actual tour cost."""
    sa = SimulatedAnnealingTSP(seed=99, max_iterations=2000)
    coords = [(0, 0), (5, 0), (2, 3), (7, 1)]
    dist = _euclidean_dist(coords)
    tour, cost = sa.solve(dist)
    actual = sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    assert cost == pytest.approx(actual)


def test_sa_empty_instance():
    """SA with empty distance matrix returns empty tour."""
    sa = SimulatedAnnealingTSP(seed=42)
    tour, cost = sa.solve([])
    assert tour == []
    assert cost == 0.0


def test_sa_single_city():
    """SA with single city returns trivial tour."""
    sa = SimulatedAnnealingTSP(seed=42)
    tour, cost = sa.solve([[0]])
    assert tour == [0]
    assert cost == 0.0


def test_sa_two_cities():
    """SA with two cities returns the only possible tour."""
    sa = SimulatedAnnealingTSP(seed=42, max_iterations=500)
    dist = [[0, 10], [10, 0]]
    tour, cost = sa.solve(dist)
    assert len(tour) == 2
    assert set(tour) == {0, 1}
    assert cost == pytest.approx(20.0)


def test_sa_reproducible_with_seed():
    """Same seed produces the same result."""
    dist = [
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0],
    ]
    sa1 = SimulatedAnnealingTSP(seed=777, max_iterations=3000)
    sa2 = SimulatedAnnealingTSP(seed=777, max_iterations=3000)
    tour1, cost1 = sa1.solve(dist)
    tour2, cost2 = sa2.solve(dist)
    assert tour1 == tour2
    assert cost1 == pytest.approx(cost2)


def test_sa_line_instance():
    """SA on a line instance finds a near-optimal tour."""
    sa = SimulatedAnnealingTSP(seed=42, max_iterations=10000)
    n = 6
    coords = [(i, 0) for i in range(n)]
    dist = _euclidean_dist(coords)
    tour, cost = sa.solve(dist)
    # Optimal for line: 2*(n-1) = 10
    assert cost <= 2.0 * (n - 1) * 1.5  # Allow some slack for SA


def test_sa_better_than_random_tour():
    """SA produces a better tour than a random permutation on structured instance."""
    import random

    random.seed(42)
    coords = [(0, 0), (10, 0), (5, 8), (0, 10), (10, 10)]
    dist = _euclidean_dist(coords)

    # Random tour baseline
    random_tour = list(range(5))
    random.shuffle(random_tour)
    random_cost = sum(dist[random_tour[i]][random_tour[(i + 1) % 5]] for i in range(5))

    sa = SimulatedAnnealingTSP(seed=42, max_iterations=8000)
    _, sa_cost = sa.solve(dist)
    assert sa_cost <= random_cost


def test_sa_improves_with_more_iterations():
    """More iterations generally produce better or equal results."""
    dist = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0],
    ]
    sa_short = SimulatedAnnealingTSP(seed=42, max_iterations=500)
    sa_long = SimulatedAnnealingTSP(seed=42, max_iterations=10000)
    _, cost_short = sa_short.solve(dist)
    _, cost_long = sa_long.solve(dist)
    # More iterations should not make things worse (same seed start)
    assert cost_long <= cost_short + 1e-6

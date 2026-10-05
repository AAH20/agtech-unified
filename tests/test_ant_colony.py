"""Tests for Ant Colony Optimization TSP solver."""

import math

import pytest

from src.optimization.ant_colony import AntColonyTSP


def _euclidean_dist(coords):
    n = len(coords)
    return [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(n)
        ]
        for i in range(n)
    ]


def test_aco_basic_square():
    """ACO finds the optimal tour on a 4-city square."""
    aco = AntColonyTSP(seed=42, n_iterations=100, n_ants=20)
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    tour, cost = aco.solve(dist)
    assert len(tour) == 4
    assert set(tour) == {0, 1, 2, 3}
    assert cost == pytest.approx(4.0)


def test_aco_valid_tour():
    """ACO returns a valid permutation of all cities."""
    aco = AntColonyTSP(seed=123, n_iterations=50, n_ants=15)
    coords = [(0, 0), (3, 0), (3, 4), (0, 4), (1, 1)]
    dist = _euclidean_dist(coords)
    tour, cost = aco.solve(dist)
    assert len(tour) == 5
    assert set(tour) == {0, 1, 2, 3, 4}
    assert cost > 0


def test_aco_cost_matches_tour():
    """ACO-reported cost matches the actual tour cost."""
    aco = AntColonyTSP(seed=99, n_iterations=50, n_ants=15)
    coords = [(0, 0), (5, 0), (2, 3), (7, 1)]
    dist = _euclidean_dist(coords)
    tour, cost = aco.solve(dist)
    actual = sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    assert cost == pytest.approx(actual)


def test_aco_empty_instance():
    """ACO with empty distance matrix returns empty tour."""
    aco = AntColonyTSP(seed=42)
    tour, cost = aco.solve([])
    assert tour == []
    assert cost == 0.0


def test_aco_single_city():
    """ACO with single city returns trivial tour."""
    aco = AntColonyTSP(seed=42)
    tour, cost = aco.solve([[0]])
    assert tour == [0]
    assert cost == 0.0


def test_aco_two_cities():
    """ACO with two cities returns the only possible tour."""
    aco = AntColonyTSP(seed=42, n_iterations=20, n_ants=5)
    dist = [[0, 10], [10, 0]]
    tour, cost = aco.solve(dist)
    assert len(tour) == 2
    assert set(tour) == {0, 1}
    assert cost == pytest.approx(20.0)


def test_aco_reproducible_with_seed():
    """Same seed produces the same result."""
    dist = [
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0],
    ]
    aco1 = AntColonyTSP(seed=777, n_iterations=50, n_ants=15)
    aco2 = AntColonyTSP(seed=777, n_iterations=50, n_ants=15)
    tour1, cost1 = aco1.solve(dist)
    tour2, cost2 = aco2.solve(dist)
    assert tour1 == tour2
    assert cost1 == pytest.approx(cost2)


def test_aco_line_instance():
    """ACO on a line instance finds a near-optimal tour."""
    aco = AntColonyTSP(seed=42, n_iterations=100, n_ants=20)
    n = 6
    coords = [(i, 0) for i in range(n)]
    dist = _euclidean_dist(coords)
    tour, cost = aco.solve(dist)
    # Optimal for line: 2*(n-1) = 10
    assert cost <= 2.0 * (n - 1) * 1.5


def test_aco_better_than_random_tour():
    """ACO produces a better tour than a random permutation on structured instance."""
    import random

    random.seed(42)
    coords = [(0, 0), (10, 0), (5, 8), (0, 10), (10, 10)]
    dist = _euclidean_dist(coords)

    random_tour = list(range(5))
    random.shuffle(random_tour)
    random_cost = sum(dist[random_tour[i]][random_tour[(i + 1) % 5]] for i in range(5))

    aco = AntColonyTSP(seed=42, n_iterations=100, n_ants=20)
    _, aco_cost = aco.solve(dist)
    assert aco_cost <= random_cost


def test_aco_improves_with_more_iterations():
    """More iterations generally produce better or equal results."""
    dist = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0],
    ]
    aco_short = AntColonyTSP(seed=42, n_iterations=10, n_ants=10)
    aco_long = AntColonyTSP(seed=42, n_iterations=100, n_ants=20)
    _, cost_short = aco_short.solve(dist)
    _, cost_long = aco_long.solve(dist)
    assert cost_long <= cost_short + 1e-6


def test_aco_invalid_params():
    """ACO raises ValueError for invalid parameters."""
    with pytest.raises(ValueError):
        AntColonyTSP(n_ants=0)
    with pytest.raises(ValueError):
        AntColonyTSP(n_iterations=0)
    with pytest.raises(ValueError):
        AntColonyTSP(alpha=-0.1)
    with pytest.raises(ValueError):
        AntColonyTSP(beta=-0.1)
    with pytest.raises(ValueError):
        AntColonyTSP(evaporation_rate=1.5)
    with pytest.raises(ValueError):
        AntColonyTSP(evaporation_rate=-0.1)


def test_aco_non_square_matrix():
    """ACO raises ValueError for non-square distance matrix."""
    aco = AntColonyTSP(seed=42)
    with pytest.raises(ValueError):
        aco.solve([[0, 1], [1, 0], [0, 1]])


def test_aco_negative_distance():
    """ACO raises ValueError for negative distances."""
    aco = AntColonyTSP(seed=42)
    with pytest.raises(ValueError):
        aco.solve([[0, -1], [-1, 0]])


def test_aco_larger_instance():
    """ACO handles a larger instance (10 cities) and returns valid tour."""
    aco = AntColonyTSP(seed=42, n_iterations=100, n_ants=20)
    import random

    random.seed(123)
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    dist = _euclidean_dist(coords)
    tour, cost = aco.solve(dist)
    assert len(tour) == 10
    assert set(tour) == set(range(10))
    assert cost > 0


def test_aco_pheromone_matrix_initialized():
    """ACO initializes pheromone matrix with positive values."""
    aco = AntColonyTSP(seed=42, n_ants=10, n_iterations=5)
    dist = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
    aco.solve(dist)
    assert aco.pheromone is not None
    assert len(aco.pheromone) == 3
    for i in range(3):
        for j in range(3):
            assert aco.pheromone[i][j] > 0

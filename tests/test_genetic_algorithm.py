"""Tests for Genetic Algorithm TSP solver."""

import math

import pytest

from src.optimization.genetic_algorithm import GeneticAlgorithmTSP


def _euclidean_dist(coords):
    n = len(coords)
    return [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(n)
        ]
        for i in range(n)
    ]


def test_ga_basic_square():
    """GA finds the optimal tour on a 4-city square."""
    ga = GeneticAlgorithmTSP(seed=42, generations=200, population_size=50)
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    tour, cost = ga.solve(dist)
    assert len(tour) == 4
    assert set(tour) == {0, 1, 2, 3}
    assert cost == pytest.approx(4.0)


def test_ga_valid_tour():
    """GA returns a valid permutation of all cities."""
    ga = GeneticAlgorithmTSP(seed=123, generations=100, population_size=30)
    coords = [(0, 0), (3, 0), (3, 4), (0, 4), (1, 1)]
    dist = _euclidean_dist(coords)
    tour, cost = ga.solve(dist)
    assert len(tour) == 5
    assert set(tour) == {0, 1, 2, 3, 4}
    assert cost > 0


def test_ga_cost_matches_tour():
    """GA-reported cost matches the actual tour cost."""
    ga = GeneticAlgorithmTSP(seed=99, generations=100, population_size=30)
    coords = [(0, 0), (5, 0), (2, 3), (7, 1)]
    dist = _euclidean_dist(coords)
    tour, cost = ga.solve(dist)
    actual = sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    assert cost == pytest.approx(actual)


def test_ga_empty_instance():
    """GA with empty distance matrix returns empty tour."""
    ga = GeneticAlgorithmTSP(seed=42)
    tour, cost = ga.solve([])
    assert tour == []
    assert cost == 0.0


def test_ga_single_city():
    """GA with single city returns trivial tour."""
    ga = GeneticAlgorithmTSP(seed=42)
    tour, cost = ga.solve([[0]])
    assert tour == [0]
    assert cost == 0.0


def test_ga_two_cities():
    """GA with two cities returns the only possible tour."""
    ga = GeneticAlgorithmTSP(seed=42, generations=50, population_size=10)
    dist = [[0, 10], [10, 0]]
    tour, cost = ga.solve(dist)
    assert len(tour) == 2
    assert set(tour) == {0, 1}
    assert cost == pytest.approx(20.0)


def test_ga_reproducible_with_seed():
    """Same seed produces the same result."""
    dist = [
        [0, 2, 9, 10],
        [2, 0, 6, 4],
        [9, 6, 0, 8],
        [10, 4, 8, 0],
    ]
    ga1 = GeneticAlgorithmTSP(seed=777, generations=100, population_size=30)
    ga2 = GeneticAlgorithmTSP(seed=777, generations=100, population_size=30)
    tour1, cost1 = ga1.solve(dist)
    tour2, cost2 = ga2.solve(dist)
    assert tour1 == tour2
    assert cost1 == pytest.approx(cost2)


def test_ga_line_instance():
    """GA on a line instance finds a near-optimal tour."""
    ga = GeneticAlgorithmTSP(seed=42, generations=300, population_size=50)
    n = 6
    coords = [(i, 0) for i in range(n)]
    dist = _euclidean_dist(coords)
    tour, cost = ga.solve(dist)
    # Optimal for line: 2*(n-1) = 10
    assert cost <= 2.0 * (n - 1) * 1.5


def test_ga_better_than_random_tour():
    """GA produces a better tour than a random permutation on structured instance."""
    import random

    random.seed(42)
    coords = [(0, 0), (10, 0), (5, 8), (0, 10), (10, 10)]
    dist = _euclidean_dist(coords)

    random_tour = list(range(5))
    random.shuffle(random_tour)
    random_cost = sum(dist[random_tour[i]][random_tour[(i + 1) % 5]] for i in range(5))

    ga = GeneticAlgorithmTSP(seed=42, generations=200, population_size=50)
    _, ga_cost = ga.solve(dist)
    assert ga_cost <= random_cost


def test_ga_improves_with_more_generations():
    """More generations generally produce better or equal results."""
    dist = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0],
    ]
    ga_short = GeneticAlgorithmTSP(seed=42, generations=20, population_size=20)
    ga_long = GeneticAlgorithmTSP(seed=42, generations=200, population_size=50)
    _, cost_short = ga_short.solve(dist)
    _, cost_long = ga_long.solve(dist)
    assert cost_long <= cost_short + 1e-6


def test_ga_elitism_preserves_best():
    """Elitism ensures the best solution is never lost."""
    ga = GeneticAlgorithmTSP(seed=42, generations=100, population_size=30, elitism_count=2)
    coords = [(0, 0), (3, 0), (3, 4), (0, 4), (1, 1), (2, 2)]
    dist = _euclidean_dist(coords)
    tour, cost = ga.solve(dist)
    # Best cost should be monotonically tracked
    assert cost <= ga.best_cost_history[0] + 1e-9


def test_ga_invalid_params():
    """GA raises ValueError for invalid parameters."""
    with pytest.raises(ValueError):
        GeneticAlgorithmTSP(population_size=0)
    with pytest.raises(ValueError):
        GeneticAlgorithmTSP(generations=0)
    with pytest.raises(ValueError):
        GeneticAlgorithmTSP(crossover_rate=1.5)
    with pytest.raises(ValueError):
        GeneticAlgorithmTSP(mutation_rate=-0.1)
    with pytest.raises(ValueError):
        GeneticAlgorithmTSP(elitism_count=-1)


def test_ga_non_square_matrix():
    """GA raises ValueError for non-square distance matrix."""
    ga = GeneticAlgorithmTSP(seed=42)
    with pytest.raises(ValueError):
        ga.solve([[0, 1], [1, 0], [0, 1]])


def test_ga_negative_distance():
    """GA raises ValueError for negative distances."""
    ga = GeneticAlgorithmTSP(seed=42)
    with pytest.raises(ValueError):
        ga.solve([[0, -1], [-1, 0]])


def test_ga_larger_instance():
    """GA handles a larger instance (10 cities) and returns valid tour."""
    ga = GeneticAlgorithmTSP(seed=42, generations=300, population_size=50)
    import random

    random.seed(123)
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    dist = _euclidean_dist(coords)
    tour, cost = ga.solve(dist)
    assert len(tour) == 10
    assert set(tour) == set(range(10))
    assert cost > 0

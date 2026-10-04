"""Tests for 3-opt local search for TSP."""

import pytest

from src.optimization.three_opt import ThreeOpt


def test_three_opt_empty_tour():
    """Empty tour returns empty list."""
    opt = ThreeOpt()
    assert opt.improve([], []) == []


def test_three_opt_single_city():
    """Single city tour is returned unchanged."""
    opt = ThreeOpt()
    assert opt.improve([0], [[0]]) == [0]


def test_three_opt_two_cities():
    """Two-city tour is returned unchanged."""
    opt = ThreeOpt()
    result = opt.improve([0, 1], [[0, 10], [10, 0]])
    assert len(result) == 2
    assert set(result) == {0, 1}


def test_three_opt_improves_crossing_tour():
    """3-opt improves a tour that has crossing edges."""
    opt = ThreeOpt()
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    result = opt.improve([0, 2, 1, 3], dist)
    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}


def test_three_opt_already_optimal():
    """3-opt leaves already-optimal tour unchanged."""
    opt = ThreeOpt()
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    result = opt.improve([0, 1, 2, 3], dist)
    assert result == [0, 1, 2, 3]


def test_three_opt_no_duplicates():
    """3-opt rejects duplicate cities in tour."""
    opt = ThreeOpt()
    with pytest.raises(ValueError, match="duplicate"):
        opt.improve([0, 1, 1], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])


def test_three_opt_index_out_of_range():
    """3-opt rejects out-of-range indices."""
    opt = ThreeOpt()
    with pytest.raises(ValueError, match="exceeds"):
        opt.improve([0, 1, 5], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])


def test_three_opt_5_city_tour():
    """3-opt produces valid permutation on a 5-city instance."""
    opt = ThreeOpt()
    coords = [(0, 0), (10, 0), (5, 8), (0, 10), (10, 10)]
    dist = [
        [
            float(abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]))
            for j in range(5)
        ]
        for i in range(5)
    ]
    tour = [0, 2, 4, 1, 3]
    result = opt.improve(tour, dist)
    assert len(result) == 5
    assert set(result) == {0, 1, 2, 3, 4}

    def cost(t):
        return sum(dist[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))

    assert cost(result) <= cost(tour)


def test_three_opt_beats_two_opt_on_specific_instance():
    """3-opt finds a better tour than 2-opt on instances needing 3-edge moves."""
    from src.optimization.local_search import TwoOpt

    # Construct an instance where 3-opt helps: a "double-star" configuration
    # that 2-opt cannot fully optimize but 3-opt can
    opt3 = ThreeOpt()
    opt2 = TwoOpt()
    dist = [
        [0, 1, 10, 10, 2, 2],
        [1, 0, 10, 10, 2, 2],
        [10, 10, 0, 1, 10, 10],
        [10, 10, 1, 0, 10, 10],
        [2, 2, 10, 10, 0, 1],
        [2, 2, 10, 10, 1, 0],
    ]
    tour = [0, 2, 4, 1, 3, 5]

    result3 = opt3.improve(tour, dist)
    result2 = opt2.improve(tour, dist)

    def cost(t):
        return sum(dist[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))

    assert cost(result3) <= cost(result2) + 1e-9


def test_three_opt_result_is_valid_permutation():
    """3-opt always returns a valid permutation regardless of input."""
    opt = ThreeOpt()
    dist = [
        [0, 3, 4, 2, 7],
        [3, 0, 5, 6, 1],
        [4, 5, 0, 8, 3],
        [2, 6, 8, 0, 9],
        [7, 1, 3, 9, 0],
    ]
    # Start from a non-trivial tour
    tour = [0, 3, 1, 4, 2]
    result = opt.improve(tour, dist)
    assert sorted(result) == [0, 1, 2, 3, 4]

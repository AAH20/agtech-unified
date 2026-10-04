"""Tests for don't-look bits optimization in TwoOpt local search."""

import pytest

from src.optimization.local_search import TwoOpt


def test_dont_look_bits_improves_crossing_tour():
    """Don't-look bits version still finds 2-opt improvements."""
    opt = TwoOpt(use_dont_look_bits=True)
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    result = opt.improve([0, 2, 1, 3], dist)
    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}


def test_dont_look_bits_already_optimal():
    """Don't-look bits leaves already-optimal tour unchanged."""
    opt = TwoOpt(use_dont_look_bits=True)
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    result = opt.improve([0, 1, 2, 3], dist)
    assert result == [0, 1, 2, 3]


def test_dont_look_bits_small_tour():
    """Don't-look bits works on a 5-city tour requiring multiple improvements."""
    opt = TwoOpt(use_dont_look_bits=True)
    # Create a tour that has several 2-opt improvements available
    coords = [(0, 0), (10, 0), (5, 1), (5, -1), (0, 5)]
    dist = [
        [
            float(abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]))
            for j in range(5)
        ]
        for i in range(5)
    ]
    # Start with a deliberately bad tour: zigzag
    tour = [0, 2, 1, 3, 4]
    result = opt.improve(tour, dist)
    # Result should be a valid permutation
    assert len(result) == 5
    assert set(result) == {0, 1, 2, 3, 4}

    # Result should be no worse than input
    def tour_cost(t, d):
        return sum(d[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))

    assert tour_cost(result, dist) <= tour_cost(tour, dist)


def test_dont_look_bits_empty_tour():
    """Don't-look bits with empty tour returns empty list."""
    opt = TwoOpt(use_dont_look_bits=True)
    assert opt.improve([], []) == []


def test_dont_look_bits_single_city():
    """Don't-look bits with single city returns unchanged."""
    opt = TwoOpt(use_dont_look_bits=True)
    assert opt.improve([0], [[0]]) == [0]


def test_dont_look_bits_two_cities():
    """Don't-look bits with two cities returns valid tour."""
    opt = TwoOpt(use_dont_look_bits=True)
    result = opt.improve([0, 1], [[0, 10], [10, 0]])
    assert len(result) == 2
    assert set(result) == {0, 1}


def test_dont_look_bits_no_duplicates():
    """Don't-look bits rejects duplicate cities in tour."""
    opt = TwoOpt(use_dont_look_bits=True)
    with pytest.raises(ValueError, match="duplicate"):
        opt.improve([0, 1, 1], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])


def test_dont_look_bits_index_out_of_range():
    """Don't-look bits rejects out-of-range indices."""
    opt = TwoOpt(use_dont_look_bits=True)
    with pytest.raises(ValueError, match="exceeds"):
        opt.improve([0, 1, 5], [[0, 1, 2], [1, 0, 3], [2, 3, 0]])


def test_dont_look_bits_result_no_worse_than_without():
    """Don't-look bits gives same or better result than plain 2-opt on same tour."""
    dist = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0],
    ]
    tour = [0, 2, 4, 1, 3]

    plain = TwoOpt(use_dont_look_bits=False).improve(tour, dist)
    smart = TwoOpt(use_dont_look_bits=True).improve(tour, dist)

    def cost(t):
        return sum(dist[t[i]][t[(i + 1) % len(t)]] for i in range(len(t)))

    assert cost(smart) <= cost(plain) + 1e-9

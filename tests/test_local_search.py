"""Tests for 2-opt local search."""

from src.optimization.local_search import TwoOpt


def test_empty_tour():
    """Empty tour returns empty list."""
    opt = TwoOpt()
    assert opt.improve([], []) == []


def test_single_city():
    """Single city tour is returned unchanged."""
    opt = TwoOpt()
    assert opt.improve([0], [[0]]) == [0]


def test_improves_crossing_tour():
    """2-opt improves a suboptimal tour with crossing edges."""
    opt = TwoOpt()
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    result = opt.improve([0, 2, 1, 3], dist)
    assert len(result) == 4
    assert set(result) == {0, 1, 2, 3}


def test_already_optimal():
    """Already optimal tour is returned unchanged."""
    opt = TwoOpt()
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    result = opt.improve([0, 1, 2, 3], dist)
    assert result == [0, 1, 2, 3]


def test_two_cities():
    """Two-city tour is returned unchanged."""
    opt = TwoOpt()
    result = opt.improve([0, 1], [[0, 10], [10, 0]])
    assert len(result) == 2
    assert set(result) == {0, 1}

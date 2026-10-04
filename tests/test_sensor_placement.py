"""Test IoT sensor placement optimization (Set Cover reduction)."""
import pytest
from src.iot.sensor_placement import SensorPlacement, PlacementInstance


def test_placement_empty():
    """No sensors or targets returns empty placement."""
    placement = SensorPlacement()
    instance = PlacementInstance(
        sensor_positions=[],
        target_positions=[],
        coverage_radius=5.0,
    )
    result = placement.optimize(instance)
    assert result.selected_sensors == []
    assert result.coverage_ratio == 0.0


def test_placement_single_sensor_covers_all():
    """One sensor covering all targets."""
    placement = SensorPlacement()
    instance = PlacementInstance(
        sensor_positions=[(0, 0)],
        target_positions=[(1, 0), (2, 0), (3, 0)],
        coverage_radius=5.0,
    )
    result = placement.optimize(instance)
    assert len(result.selected_sensors) == 1
    assert result.coverage_ratio == 1.0


def test_placement_two_sensors_needed():
    """Two sensors needed to cover all targets."""
    placement = SensorPlacement()
    instance = PlacementInstance(
        sensor_positions=[(0, 0), (10, 0)],
        target_positions=[(0, 0), (10, 0)],
        coverage_radius=3.0,
    )
    result = placement.optimize(instance)
    assert len(result.selected_sensors) == 2
    assert result.coverage_ratio == 1.0


def test_placement_greedy_set_cover():
    """Greedy algorithm selects minimal sensor set."""
    placement = SensorPlacement(algorithm="greedy")
    instance = PlacementInstance(
        sensor_positions=[(0, 0), (1, 0), (10, 0)],
        target_positions=[(0, 0), (10, 0)],
        coverage_radius=2.0,
    )
    result = placement.optimize(instance)
    # Sensor at (0,0) covers target at (0,0), sensor at (10,0) covers target at (10,0)
    # Sensor at (1,0) is redundant
    assert len(result.selected_sensors) == 2


def test_placement_partial_coverage():
    """Not all targets can be covered."""
    placement = SensorPlacement()
    instance = PlacementInstance(
        sensor_positions=[(0, 0)],
        target_positions=[(0, 0), (100, 100)],
        coverage_radius=5.0,
    )
    result = placement.optimize(instance)
    assert result.coverage_ratio < 1.0
    assert result.coverage_ratio > 0.0


def test_placement_result_contains_algorithm():
    """PlacementResult includes algorithm name."""
    placement = SensorPlacement(algorithm="greedy")
    instance = PlacementInstance(
        sensor_positions=[(0, 0)],
        target_positions=[(1, 0)],
        coverage_radius=5.0,
    )
    result = placement.optimize(instance)
    assert result.algorithm == "greedy"


def test_placement_zero_radius():
    """Zero coverage radius raises ValueError."""
    placement = SensorPlacement()
    with pytest.raises(ValueError, match="Coverage radius must be positive"):
        PlacementInstance(
            sensor_positions=[(0, 0)],
            target_positions=[(1, 0)],
            coverage_radius=0,
        )


def test_placement_all_targets_covered():
    """All targets within radius are covered."""
    placement = SensorPlacement()
    instance = PlacementInstance(
        sensor_positions=[(0, 0), (5, 0)],
        target_positions=[(1, 0), (4, 0)],
        coverage_radius=2.0,
    )
    result = placement.optimize(instance)
    assert result.coverage_ratio == 1.0


def test_placement_minimizes_sensor_count():
    """Greedy selects fewer sensors when possible."""
    placement = SensorPlacement(algorithm="greedy")
    instance = PlacementInstance(
        sensor_positions=[(0, 0), (1, 0), (2, 0)],
        target_positions=[(0, 0), (1, 0), (2, 0)],
        coverage_radius=1.5,
    )
    result = placement.optimize(instance)
    # One sensor at (1,0) with radius 1.5 covers all three targets
    assert len(result.selected_sensors) <= 2

"""Test digital twin simulation engine for agriculture."""

import pytest

from src.digital_twin.simulator import DigitalTwin, SimulationState


def test_simulation_empty_state():
    """Empty state returns empty result."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=0.0,
        temperature=20.0,
        crop_height=0.0,
        nutrient_level=0.0,
    )
    result = twin.simulate(state, days=0)
    assert result.days_simulated == 0
    assert result.final_state.crop_height == 0.0


def test_simulation_growth():
    """Crop grows over time."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    result = twin.simulate(state, days=10)
    assert result.days_simulated == 10
    assert result.final_state.crop_height > 0.1


def test_simulation_water_stress():
    """Low soil moisture reduces growth."""
    twin = DigitalTwin()
    well_watered = SimulationState(
        soil_moisture=0.8,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    drought = SimulationState(
        soil_moisture=0.1,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    result_good = twin.simulate(well_watered, days=10)
    result_bad = twin.simulate(drought, days=10)
    assert result_good.final_state.crop_height > result_bad.final_state.crop_height


def test_simulation_temperature_effect():
    """Optimal temperature produces more growth."""
    twin = DigitalTwin()
    optimal = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    cold = SimulationState(
        soil_moisture=0.5,
        temperature=5.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    result_opt = twin.simulate(optimal, days=10)
    result_cold = twin.simulate(cold, days=10)
    assert result_opt.final_state.crop_height > result_cold.final_state.crop_height


def test_simulation_nutrient_effect():
    """Higher nutrients produce more growth."""
    twin = DigitalTwin()
    rich = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.9,
    )
    poor = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.1,
    )
    result_rich = twin.simulate(rich, days=10)
    result_poor = twin.simulate(poor, days=10)
    assert result_rich.final_state.crop_height > result_poor.final_state.crop_height


def test_simulation_result_contains_algorithm():
    """SimulationResult includes algorithm name."""
    twin = DigitalTwin(algorithm="logistic_growth")
    state = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    result = twin.simulate(state, days=5)
    assert result.algorithm == "logistic_growth"


def test_simulation_negative_days():
    """Negative days raises ValueError."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    with pytest.raises(ValueError, match="Days must be non-negative"):
        twin.simulate(state, days=-1)


def test_simulation_history():
    """Simulation returns daily history."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=0.5,
        temperature=25.0,
        crop_height=0.1,
        nutrient_level=0.5,
    )
    result = twin.simulate(state, days=5)
    assert len(result.history) == 5


def test_simulation_max_height():
    """Crop height doesn't exceed maximum."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=1.0,
        temperature=30.0,
        crop_height=0.1,
        nutrient_level=1.0,
    )
    result = twin.simulate(state, days=100)
    assert result.final_state.crop_height <= 2.0  # Max crop height

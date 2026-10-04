"""Dedicated unit tests for DigitalTwin (src/digital_twin/simulator.py).

Covers algorithm selection, _step() correctness, _temp_factor()/_water_factor()
boundary values, multi-day simulation accuracy, and edge cases.
"""

from __future__ import annotations

import pytest

from src.digital_twin.simulator import DigitalTwin, SimulationResult, SimulationState


class TestDigitalTwinInit:
    """DigitalTwin initialization tests."""

    def test_default_algorithm(self):
        twin = DigitalTwin()
        assert twin.algorithm == "logistic_growth"

    def test_custom_algorithm_stored(self):
        twin = DigitalTwin(algorithm="custom")
        assert twin.algorithm == "custom"

    def test_default_parameters(self):
        twin = DigitalTwin()
        assert twin.max_crop_height == 2.0
        assert twin.optimal_temp == 25.0
        assert twin.growth_rate == 0.1


class TestSimulateBasic:
    """Basic simulation tests."""

    def test_zero_days_returns_initial_state(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=0)
        assert result.days_simulated == 0
        assert result.final_state.crop_height == 0.1
        assert result.total_growth == 0.0
        assert len(result.history) == 0

    def test_negative_days_raises(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        with pytest.raises(ValueError, match="Days must be non-negative"):
            digital_twin.simulate(state, days=-1)

    def test_one_day_simulation(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=1)
        assert result.days_simulated == 1
        assert len(result.history) == 1
        assert result.final_state.crop_height > 0.1

    def test_simulation_result_fields(self, digital_twin, optimal_sim_state):
        result = digital_twin.simulate(optimal_sim_state, days=5)
        assert isinstance(result, SimulationResult)
        assert result.days_simulated == 5
        assert isinstance(result.final_state, SimulationState)
        assert isinstance(result.history, list)
        assert result.algorithm == "logistic_growth"
        assert isinstance(result.total_growth, float)

    def test_history_length_matches_days(self, digital_twin, optimal_sim_state):
        result = digital_twin.simulate(optimal_sim_state, days=10)
        assert len(result.history) == 10

    def test_total_growth_calculation(self, digital_twin, optimal_sim_state):
        result = digital_twin.simulate(optimal_sim_state, days=5)
        expected = result.final_state.crop_height - optimal_sim_state.crop_height
        assert result.total_growth == pytest.approx(expected)


class TestTempFactor:
    """Tests for _temp_factor() boundary values."""

    def test_optimal_temp_returns_1(self, digital_twin):
        factor = digital_twin._temp_factor(25.0)
        assert factor == pytest.approx(1.0)

    def test_zero_temp_returns_near_zero(self, digital_twin):
        """Temp factor at 0°C is small but not exactly zero (Gaussian)."""
        factor = digital_twin._temp_factor(0.0)
        assert 0.0 < factor < 0.1

    def test_negative_temp_returns_zero(self, digital_twin):
        factor = digital_twin._temp_factor(-5.0)
        assert factor == 0.0

    def test_45_temp_returns_small_value(self, digital_twin):
        """Temp factor at 45°C is small but not exactly zero (Gaussian)."""
        factor = digital_twin._temp_factor(45.0)
        assert 0.0 < factor < 0.2

    def test_above_45_temp_returns_zero(self, digital_twin):
        factor = digital_twin._temp_factor(50.0)
        assert factor == 0.0

    def test_temp_factor_symmetric_around_optimal(self, digital_twin):
        """Temp factor should be symmetric around optimal temp."""
        factor_20 = digital_twin._temp_factor(20.0)
        factor_30 = digital_twin._temp_factor(30.0)
        assert factor_20 == pytest.approx(factor_30)

    def test_temp_factor_decreases_from_optimal(self, digital_twin):
        """Temp factor should decrease as we move away from optimal."""
        factor_25 = digital_twin._temp_factor(25.0)
        factor_20 = digital_twin._temp_factor(20.0)
        factor_15 = digital_twin._temp_factor(15.0)
        assert factor_25 > factor_20 > factor_15

    def test_temp_factor_in_range_01(self, digital_twin):
        """Temp factor should always be in [0, 1]."""
        for temp in range(0, 46):
            factor = digital_twin._temp_factor(float(temp))
            assert 0.0 <= factor <= 1.0


class TestWaterFactor:
    """Tests for _water_factor() boundary values."""

    def test_zero_moisture_returns_zero(self, digital_twin):
        factor = digital_twin._water_factor(0.0)
        assert factor == 0.0

    def test_below_01_moisture_returns_zero(self, digital_twin):
        factor = digital_twin._water_factor(0.05)
        assert factor == 0.0

    def test_01_moisture_returns_02(self, digital_twin):
        factor = digital_twin._water_factor(0.1)
        assert factor == pytest.approx(0.2)

    def test_05_moisture_returns_1(self, digital_twin):
        factor = digital_twin._water_factor(0.5)
        assert factor == pytest.approx(1.0)

    def test_above_05_moisture_capped_at_1(self, digital_twin):
        factor = digital_twin._water_factor(0.8)
        assert factor == pytest.approx(1.0)

    def test_water_factor_monotonically_increasing(self, digital_twin):
        """Water factor should increase with moisture."""
        prev = 0.0
        for moisture in [0.1, 0.2, 0.3, 0.4, 0.5]:
            factor = digital_twin._water_factor(moisture)
            assert factor >= prev
            prev = factor


class TestStepFunction:
    """Tests for _step() correctness."""

    def test_step_increases_height(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        new_state = digital_twin._step(state)
        assert new_state.crop_height > state.crop_height

    def test_step_decreases_moisture(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        new_state = digital_twin._step(state)
        assert new_state.soil_moisture < state.soil_moisture

    def test_step_preserves_temperature(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        new_state = digital_twin._step(state)
        assert new_state.temperature == state.temperature

    def test_step_preserves_nutrient(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        new_state = digital_twin._step(state)
        assert new_state.nutrient_level == state.nutrient_level

    def test_step_moisture_never_negative(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.01,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        new_state = digital_twin._step(state)
        assert new_state.soil_moisture >= 0.0

    def test_step_height_never_exceeds_max(self, digital_twin):
        state = SimulationState(
            soil_moisture=1.0,
            temperature=25.0,
            crop_height=1.9,
            nutrient_level=1.0,
        )
        new_state = digital_twin._step(state)
        assert new_state.crop_height <= digital_twin.max_crop_height


class TestMultiDaySimulation:
    """Tests for multi-day simulation accuracy."""

    def test_growth_increases_over_time(self, digital_twin, optimal_sim_state):
        result_5 = digital_twin.simulate(optimal_sim_state, days=5)
        result_10 = digital_twin.simulate(optimal_sim_state, days=10)
        assert result_10.final_state.crop_height > result_5.final_state.crop_height

    def test_growth_rate_slows_near_max(self, digital_twin):
        """Growth should slow as crop approaches max height."""
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=1.5,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=10)
        # Should approach but not exceed max
        assert result.final_state.crop_height <= 2.0
        assert result.final_state.crop_height > 1.5

    def test_365_day_simulation(self, digital_twin):
        """Very long simulation should complete and converge."""
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=365)
        assert result.days_simulated == 365
        assert result.final_state.crop_height <= 2.0
        assert len(result.history) == 365

    def test_moisture_depletes_over_time(self, digital_twin, optimal_sim_state):
        result = digital_twin.simulate(optimal_sim_state, days=30)
        assert result.final_state.soil_moisture < optimal_sim_state.soil_moisture

    def test_zero_moisture_no_growth(self, digital_twin, drought_sim_state):
        result = digital_twin.simulate(drought_sim_state, days=10)
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_extreme_heat_no_growth(self, digital_twin, extreme_heat_sim_state):
        result = digital_twin.simulate(extreme_heat_sim_state, days=10)
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)


class TestSimulationEdgeCases:
    """Edge case tests for DigitalTwin."""

    def test_zero_crop_height_no_growth(self, digital_twin):
        """Zero crop height means no growth (logistic model)."""
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.0,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=10)
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_max_crop_height_no_growth(self, digital_twin):
        """At max height, growth should be zero."""
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=2.0,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=10)
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_very_high_temperature_no_growth(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=100.0,
            crop_height=0.5,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=5)
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_very_low_temperature_no_growth(self, digital_twin):
        state = SimulationState(
            soil_moisture=0.5,
            temperature=-10.0,
            crop_height=0.5,
            nutrient_level=0.5,
        )
        result = digital_twin.simulate(state, days=5)
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_full_moisture_max_growth(self, digital_twin):
        """Full moisture should produce maximum water factor."""
        state = SimulationState(
            soil_moisture=1.0,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=1.0,
        )
        result = digital_twin.simulate(state, days=10)
        assert result.total_growth > 0.1

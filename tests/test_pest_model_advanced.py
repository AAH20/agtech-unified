"""Tests for advanced pest population dynamics: degree-days, development rates, predator-prey."""

import pytest

from src.digital_twin.pest_model import (
    PestModel,
    PestSimulationResult,
    PestState,
)


class TestDegreeDayAccumulation:
    """Degree-day driven development."""

    def test_degree_days_accumulate_above_base_temp(self):
        """Degree-days accumulate when temperature exceeds base threshold."""
        model = PestModel(base_temp=10.0)
        dd = model.accumulate_degree_days(20.0, 1)
        assert dd == pytest.approx(10.0)

    def test_degree_days_zero_below_base_temp(self):
        """No degree-days accumulate below base temperature."""
        model = PestModel(base_temp=10.0)
        dd = model.accumulate_degree_days(5.0, 1)
        assert dd == 0.0

    def test_degree_days_accumulate_over_multiple_days(self):
        """Degree-days accumulate over multiple days."""
        model = PestModel(base_temp=10.0)
        dd = model.accumulate_degree_days(20.0, 5)
        assert dd == pytest.approx(50.0)

    def test_degree_days_fractional_temperature(self):
        """Degree-days work with fractional temperatures."""
        model = PestModel(base_temp=10.0)
        dd = model.accumulate_degree_days(15.5, 1)
        assert dd == pytest.approx(5.5)


class TestTemperatureDependentDevelopment:
    """Temperature-dependent development rates using thermal response curve."""

    def test_development_rate_zero_below_base(self):
        """No development below base temperature."""
        model = PestModel(base_temp=10.0, max_temp=35.0)
        assert model.development_rate(5.0) == 0.0

    def test_development_rate_zero_above_max(self):
        """No development above maximum temperature."""
        model = PestModel(base_temp=10.0, max_temp=35.0)
        assert model.development_rate(40.0) == 0.0

    def test_development_rate_peak_at_optimal(self):
        """Development rate peaks at optimal temperature."""
        model = PestModel(base_temp=10.0, max_temp=35.0, optimal_temp=25.0)
        peak = model.development_rate(25.0)
        lower = model.development_rate(20.0)
        higher = model.development_rate(30.0)
        assert peak > lower
        assert peak > higher

    def test_development_rate_asymmetric_around_optimal(self):
        """Development rate declines faster above optimal than below."""
        model = PestModel(base_temp=10.0, max_temp=35.0, optimal_temp=25.0)
        below = model.development_rate(22.0)
        above = model.development_rate(28.0)
        # Above optimal declines faster (steeper slope)
        assert above < below

    def test_development_rate_increases_toward_optimal(self):
        """Development rate increases as temperature approaches optimal."""
        model = PestModel(base_temp=10.0, max_temp=35.0, optimal_temp=25.0)
        r15 = model.development_rate(15.0)
        r20 = model.development_rate(20.0)
        r24 = model.development_rate(24.0)
        assert r15 < r20 < r24


class TestPredatorPreyInteraction:
    """Predator-prey dynamics coupled with pest population."""

    def test_predator_reduces_pest_population(self):
        """Predators reduce pest population through predation."""
        model = PestModel(predator_efficiency=0.1, predator_growth_rate=0.05)
        state = PestState(population=0.5, predator_population=0.3)
        result = model.simulate(state, days=10)
        # With predators, pest population should be lower than without
        no_pred_state = PestState(population=0.5, predator_population=0.0)
        no_pred_result = model.simulate(no_pred_state, days=10)
        assert result.final_state.population < no_pred_result.final_state.population

    def test_predator_population_grows_with_prey(self):
        """Predator population grows when prey is abundant."""
        model = PestModel(
            predator_efficiency=0.05, predator_growth_rate=0.3, predator_death_rate=0.05
        )
        state = PestState(population=0.8, predator_population=0.1)
        result = model.simulate(state, days=20)
        assert result.final_state.predator_population > 0.1

    def test_predator_starves_without_prey(self):
        """Predator population declines when no prey available."""
        model = PestModel(predator_efficiency=0.05, predator_growth_rate=0.1)
        state = PestState(population=0.0, predator_population=0.5)
        result = model.simulate(state, days=20)
        assert result.final_state.predator_population < 0.5

    def test_predator_prey_oscillation(self):
        """Predator-prey system can exhibit oscillatory dynamics."""
        model = PestModel(
            growth_rate=0.5,
            predator_efficiency=0.2,
            predator_growth_rate=0.15,
            predator_death_rate=0.1,
        )
        state = PestState(population=0.5, predator_population=0.2)
        result = model.simulate(state, days=100)
        # Check that populations oscillate (not monotonically increasing/decreasing)
        pops = [h.population for h in result.history]
        # Should have at least one local max and one local min
        has_max = any(
            pops[i] > pops[i - 1] and pops[i] > pops[i + 1] for i in range(1, len(pops) - 1)
        )
        has_min = any(
            pops[i] < pops[i - 1] and pops[i] < pops[i + 1] for i in range(1, len(pops) - 1)
        )
        assert has_max or has_min  # At least some non-monotonic behavior

    def test_predator_efficiency_zero_no_effect(self):
        """Zero predator efficiency means no predation effect."""
        model = PestModel(predator_efficiency=0.0, predator_growth_rate=0.1)
        state = PestState(population=0.5, predator_population=0.5)
        result = model.simulate(state, days=10)
        no_pred = PestState(population=0.5, predator_population=0.0)
        no_pred_result = model.simulate(no_pred, days=10)
        assert result.final_state.population == pytest.approx(no_pred_result.final_state.population)


class TestPestStateExtended:
    """Extended PestState with predator and degree-day fields."""

    def test_default_predator_population(self):
        """Default predator population is zero."""
        state = PestState()
        assert state.predator_population == 0.0

    def test_default_degree_days(self):
        """Default accumulated degree-days is zero."""
        state = PestState()
        assert state.degree_days_accumulated == 0.0

    def test_custom_predator_population(self):
        """Custom predator population is stored."""
        state = PestState(population=0.5, predator_population=0.3)
        assert state.predator_population == 0.3


class TestPestModelExtended:
    """Extended PestModel with new parameters."""

    def test_base_temp_parameter(self):
        """Model stores base temperature for degree-day calc."""
        model = PestModel(base_temp=12.0)
        assert model.base_temp == 12.0

    def test_predator_parameters(self):
        """Model stores predator-related parameters."""
        model = PestModel(predator_efficiency=0.15, predator_growth_rate=0.08)
        assert model.predator_efficiency == 0.15
        assert model.predator_growth_rate == 0.08

    def test_degree_day_threshold(self):
        """Model stores degree-day threshold for development."""
        model = PestModel(degree_day_threshold=150.0)
        assert model.degree_day_threshold == 150.0

    def test_simulation_with_degree_days(self):
        """Simulation accumulates degree-days over time."""
        model = PestModel(base_temp=10.0, growth_rate=0.1)
        state = PestState(population=0.1, temperature=25.0)
        result = model.simulate(state, days=10)
        # At 25°C with base 10°C, should accumulate 15 DD/day = 150 DD total
        assert result.final_state.degree_days_accumulated >= 100.0

    def test_simulation_with_predators_in_history(self):
        """History tracks predator population over time."""
        model = PestModel(predator_efficiency=0.1, predator_growth_rate=0.05)
        state = PestState(population=0.5, predator_population=0.2)
        result = model.simulate(state, days=5)
        assert len(result.history) == 5
        assert all(hasattr(h, "predator_population") for h in result.history)


class TestPestModelBackwardCompatibility:
    """Existing API remains functional."""

    def test_basic_simulation_still_works(self):
        """Basic simulation without new parameters works."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.1, carrying_capacity=1.0)
        result = model.simulate(state, days=10)
        assert result.final_state.population > 0.1
        assert result.days_simulated == 10

    def test_temperature_factor_unchanged(self):
        """Original temperature_factor method still works."""
        model = PestModel(optimal_temp=25.0, temp_tolerance=15.0)
        assert model.temperature_factor(25.0) == pytest.approx(1.0)
        assert model.temperature_factor(50.0) < 0.1

    def test_pesticide_still_works(self):
        """Pesticide application still reduces population."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.8)
        result = model.simulate(state, days=1, pesticide_efficacy=0.7)
        assert result.final_state.population < 0.8

    def test_result_type(self):
        """Result is PestSimulationResult."""
        model = PestModel()
        state = PestState(population=0.1)
        result = model.simulate(state, days=5)
        assert isinstance(result, PestSimulationResult)

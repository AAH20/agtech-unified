"""Tests for pest population dynamics model."""

import pytest

from src.digital_twin.pest_model import (
    PestModel,
    PestState,
)


class TestPestState:
    """Test PestState dataclass."""

    def test_default_state(self):
        """Default pest state has zero population."""
        state = PestState()
        assert state.population == 0.0
        assert state.carrying_capacity == 1.0
        assert state.temperature == 25.0

    def test_custom_state(self):
        """Custom values are stored."""
        state = PestState(population=0.3, carrying_capacity=0.8, temperature=28.0)
        assert state.population == 0.3
        assert state.carrying_capacity == 0.8
        assert state.temperature == 28.0


class TestPestModel:
    """Test PestModel population dynamics."""

    def test_logistic_growth(self):
        """Pest population grows logistically toward carrying capacity."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.1, carrying_capacity=1.0)
        result = model.simulate(state, days=20)
        assert result.final_state.population > 0.1
        assert result.final_state.population <= 1.0

    def test_zero_population_stays_zero(self):
        """Zero initial population remains zero."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.0)
        result = model.simulate(state, days=10)
        assert result.final_state.population == 0.0

    def test_carrying_capacity_limit(self):
        """Population never exceeds carrying capacity."""
        model = PestModel(growth_rate=0.5)
        state = PestState(population=0.9, carrying_capacity=0.8)
        result = model.simulate(state, days=30)
        assert result.final_state.population <= 0.8 + 1e-9

    def test_temperature_effect_on_growth(self):
        """Higher temperature (within range) increases growth rate."""
        model = PestModel(growth_rate=0.3)
        cold_state = PestState(population=0.1, temperature=10.0)
        warm_state = PestState(population=0.1, temperature=30.0)
        cold_result = model.simulate(cold_state, days=10)
        warm_result = model.simulate(warm_state, days=10)
        assert warm_result.final_state.population > cold_result.final_state.population

    def test_extreme_temperature_stops_growth(self):
        """Extreme temperatures halt pest growth."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.5, temperature=50.0)
        result = model.simulate(state, days=5)
        assert result.final_state.population == pytest.approx(0.5)

    def test_pesticide_application(self):
        """Pesticide application reduces pest population."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.8)
        result = model.simulate(state, days=1, pesticide_efficacy=0.7)
        assert result.final_state.population < 0.8

    def test_pesticide_full_efficacy(self):
        """100% efficacy eliminates all pests."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.8)
        result = model.simulate(state, days=1, pesticide_efficacy=1.0)
        assert result.final_state.population == pytest.approx(0.0)

    def test_simulation_history(self):
        """Simulation returns daily history."""
        model = PestModel(growth_rate=0.2)
        state = PestState(population=0.1)
        result = model.simulate(state, days=5)
        assert len(result.history) == 5
        assert result.days_simulated == 5

    def test_zero_days(self):
        """Zero-day simulation returns initial state."""
        model = PestModel()
        state = PestState(population=0.5)
        result = model.simulate(state, days=0)
        assert result.days_simulated == 0
        assert result.final_state.population == 0.5
        assert result.history == []

    def test_negative_days_raises(self):
        """Negative days raises ValueError."""
        model = PestModel()
        state = PestState(population=0.5)
        with pytest.raises(ValueError, match="Days must be non-negative"):
            model.simulate(state, days=-1)

    def test_population_extinction_threshold(self):
        """Very low population goes extinct."""
        model = PestModel(growth_rate=0.1, extinction_threshold=0.01)
        state = PestState(population=0.005)
        result = model.simulate(state, days=1)
        assert result.final_state.population == 0.0

    def test_growth_rate_factor(self):
        """Temperature factor is computed correctly."""
        model = PestModel(optimal_temp=25.0, temp_tolerance=15.0)
        # At optimal temperature, factor should be 1.0
        assert model.temperature_factor(25.0) == pytest.approx(1.0)
        # Far from optimal, factor should be near 0
        assert model.temperature_factor(50.0) < 0.1

    def test_result_contains_peak_population(self):
        """Result tracks peak population."""
        model = PestModel(growth_rate=0.4)
        state = PestState(population=0.05)
        result = model.simulate(state, days=30)
        assert result.peak_population >= result.final_state.population
        assert result.peak_population >= 0.05

    def test_multiple_pesticide_applications(self):
        """Multiple pesticide applications compound the effect."""
        model = PestModel(growth_rate=0.3)
        state = PestState(population=0.9)
        # Apply pesticide every day for 3 days
        result = model.simulate(state, days=3, pesticide_efficacy=0.5)
        assert result.final_state.population < 0.9 * 0.5**3 + 0.1

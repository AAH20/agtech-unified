"""Tests for disease progression SEIR model."""

import pytest

from src.digital_twin.disease_model import (
    DiseaseModel,
    DiseaseState,
)


class TestDiseaseState:
    """Test DiseaseState dataclass."""

    def test_default_state(self):
        """Default state is fully susceptible."""
        state = DiseaseState()
        assert state.susceptible == 1.0
        assert state.exposed == 0.0
        assert state.infectious == 0.0
        assert state.recovered == 0.0

    def test_custom_state(self):
        """Custom values are stored."""
        state = DiseaseState(susceptible=0.5, exposed=0.2, infectious=0.1, recovered=0.2)
        assert state.susceptible == 0.5
        assert state.exposed == 0.2
        assert state.infectious == 0.1
        assert state.recovered == 0.2

    def test_total_population(self):
        """Total population sums all compartments."""
        state = DiseaseState(susceptible=0.4, exposed=0.1, infectious=0.2, recovered=0.3)
        assert state.total_population == pytest.approx(1.0)


class TestDiseaseModel:
    """Test DiseaseModel SEIR dynamics."""

    def test_initial_exposure(self):
        """Small initial exposure leads to epidemic growth."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.99, exposed=0.01, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=30)
        assert result.final_state.infectious > 0.0
        assert result.final_state.recovered > 0.0

    def test_no_exposure_no_disease(self):
        """Zero initial exposure means no epidemic."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=1.0, exposed=0.0, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=30)
        assert result.final_state.infectious == 0.0
        assert result.final_state.recovered == 0.0
        assert result.final_state.susceptible == pytest.approx(1.0)

    def test_recovery_increases_over_time(self):
        """Recovered compartment grows during epidemic."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.9, exposed=0.05, infectious=0.05, recovered=0.0)
        result = model.simulate(state, days=50)
        assert result.final_state.recovered > 0.05

    def test_higher_infection_rate_faster_spread(self):
        """Higher infection rate leads to faster epidemic."""
        state = DiseaseState(susceptible=0.99, exposed=0.01, infectious=0.0, recovered=0.0)
        slow_model = DiseaseModel(infection_rate=0.2, recovery_rate=0.1, incubation_rate=0.3)
        fast_model = DiseaseModel(infection_rate=0.8, recovery_rate=0.1, incubation_rate=0.3)
        slow_result = slow_model.simulate(state, days=10)
        fast_result = fast_model.simulate(state, days=10)
        assert fast_result.final_state.infectious > slow_result.final_state.infectious

    def test_simulation_history(self):
        """Simulation returns daily history."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.99, exposed=0.01, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=10)
        assert len(result.history) == 10
        assert result.days_simulated == 10

    def test_zero_days(self):
        """Zero-day simulation returns initial state."""
        model = DiseaseModel()
        state = DiseaseState(susceptible=0.5, exposed=0.2, infectious=0.1, recovered=0.2)
        result = model.simulate(state, days=0)
        assert result.days_simulated == 0
        assert result.final_state.susceptible == 0.5
        assert result.history == []

    def test_negative_days_raises(self):
        """Negative days raises ValueError."""
        model = DiseaseModel()
        state = DiseaseState()
        with pytest.raises(ValueError, match="Days must be non-negative"):
            model.simulate(state, days=-1)

    def test_peak_infection_tracked(self):
        """Result tracks peak infection level."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.99, exposed=0.01, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=50)
        assert result.peak_infection > 0.0
        assert result.peak_infection >= result.final_state.infectious

    def test_total_population_conserved(self):
        """Total population remains constant (no births/deaths)."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.9, exposed=0.05, infectious=0.05, recovered=0.0)
        result = model.simulate(state, days=30)
        total = (
            result.final_state.susceptible
            + result.final_state.exposed
            + result.final_state.infectious
            + result.final_state.recovered
        )
        assert total == pytest.approx(1.0)

    def test_temperature_stress_factor(self):
        """Temperature affects disease transmission."""
        model = DiseaseModel(optimal_temp=25.0, temp_tolerance=10.0)
        # At optimal temperature, factor should be 1.0
        assert model.temperature_stress_factor(25.0) == pytest.approx(1.0)
        # Far from optimal, factor should be low
        assert model.temperature_stress_factor(45.0) < 0.1

    def test_humidity_stress_factor(self):
        """Humidity affects disease transmission."""
        model = DiseaseModel(optimal_humidity=70.0, humidity_tolerance=30.0)
        # At optimal humidity, factor should be 1.0
        assert model.humidity_stress_factor(70.0) == pytest.approx(1.0)
        # Very low humidity should reduce transmission
        assert model.humidity_stress_factor(10.0) < 0.5

    def test_vaccination_reduces_susceptible(self):
        """Vaccination moves individuals to recovered."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.99, exposed=0.01, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=1, vaccination_rate=0.3)
        # Some susceptible should have been vaccinated
        assert result.final_state.recovered > 0.0

    def test_r0_calculation(self):
        """Basic reproduction number is computed correctly."""
        model = DiseaseModel(infection_rate=0.6, recovery_rate=0.2, incubation_rate=0.3)
        r0 = model.compute_r0()
        assert r0 == pytest.approx(3.0)

    def test_r0_below_1_no_epidemic(self):
        """R0 < 1 means disease dies out."""
        model = DiseaseModel(infection_rate=0.1, recovery_rate=0.5, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.99, exposed=0.01, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=30)
        assert result.final_state.infectious < 0.01
        assert result.peak_infection < 0.05

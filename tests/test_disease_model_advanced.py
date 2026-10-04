"""Tests for advanced disease SEIR model: latency, sporulation, environmental triggers."""

import pytest

from src.digital_twin.disease_model import (
    DiseaseModel,
    DiseaseSimulationResult,
    DiseaseState,
)


class TestInfectionLatency:
    """Exposed (latent) compartment dynamics."""

    def test_exposed_becomes_infectious(self):
        """Exposed individuals progress to infectious at incubation rate."""
        model = DiseaseModel(incubation_rate=0.5)
        state = DiseaseState(susceptible=0.0, exposed=1.0, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=5)
        assert result.final_state.infectious > 0.0
        assert result.final_state.exposed < 1.0

    def test_higher_incubation_rate_faster_progression(self):
        """Higher incubation rate means faster E->I transition."""
        fast = DiseaseModel(incubation_rate=0.8)
        slow = DiseaseModel(incubation_rate=0.2)
        state = DiseaseState(susceptible=0.0, exposed=1.0, infectious=0.0, recovered=0.0)
        fast_result = fast.simulate(state, days=3)
        slow_result = slow.simulate(state, days=3)
        assert fast_result.final_state.infectious > slow_result.final_state.infectious

    def test_no_exposed_no_new_infections(self):
        """Without exposed individuals, no new infections arise."""
        model = DiseaseModel()
        state = DiseaseState(susceptible=0.0, exposed=0.0, infectious=0.0, recovered=1.0)
        result = model.simulate(state, days=5)
        assert result.final_state.infectious == 0.0

    def test_latency_period_parameter(self):
        """Model stores mean latency period."""
        model = DiseaseModel(mean_latency_period=4.0)
        assert model.mean_latency_period == 4.0
        # incubation_rate should be 1/mean_latency
        assert model.incubation_rate == pytest.approx(0.25)


class TestSporulation:
    """Sporulation dynamics: pathogen reproduction on infected hosts."""

    def test_sporulation_increases_with_infectious(self):
        """Sporulation rate increases with infectious population."""
        model = DiseaseModel(sporulation_rate=0.5)
        state = DiseaseState(susceptible=0.5, exposed=0.0, infectious=0.5, recovered=0.0)
        result = model.simulate(state, days=5)
        # Sporulation should contribute to disease spread
        assert result.final_state.susceptible < 0.5

    def test_sporulation_rate_parameter(self):
        """Model stores sporulation rate."""
        model = DiseaseModel(sporulation_rate=0.8)
        assert model.sporulation_rate == 0.8

    def test_zero_sporulation_no_spread(self):
        """Zero sporulation means no new infections from sporulation."""
        model = DiseaseModel(sporulation_rate=0.0, infection_rate=0.0)
        state = DiseaseState(susceptible=1.0, exposed=0.0, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=5)
        assert result.final_state.susceptible == pytest.approx(1.0)

    def test_sporulation_driven_infection(self):
        """Sporulation can drive new infections independently."""
        model = DiseaseModel(
            infection_rate=0.0,  # No direct contact transmission
            sporulation_rate=0.3,
            sporulation_infection_efficiency=0.5,
        )
        state = DiseaseState(susceptible=0.8, exposed=0.0, infectious=0.2, recovered=0.0)
        result = model.simulate(state, days=10)
        # Sporulation should still cause some new infections
        assert result.final_state.susceptible < 0.8


class TestEnvironmentalTriggers:
    """Environmental conditions trigger disease outbreaks."""

    def test_leaf_wetness_triggers_infection(self):
        """Leaf wetness duration above threshold triggers infection."""
        model = DiseaseModel(leaf_wetness_threshold=6.0)
        # Start with some infectious individuals for leaf wetness to amplify
        state = DiseaseState(susceptible=0.8, exposed=0.0, infectious=0.2, recovered=0.0)
        result = model.simulate(state, days=5, leaf_wetness_hours=8.0)
        # With sufficient leaf wetness, infection should spread
        assert result.final_state.susceptible < 0.8

    def test_no_leaf_wetness_no_infection(self):
        """Without leaf wetness, no environmental infection."""
        model = DiseaseModel(leaf_wetness_threshold=6.0)
        state = DiseaseState(susceptible=1.0, exposed=0.0, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=5, leaf_wetness_hours=2.0)
        # Below threshold, no environmental infection
        assert result.final_state.susceptible == pytest.approx(1.0)

    def test_temperature_trigger_window(self):
        """Disease transmission only within temperature window."""
        model = DiseaseModel(min_temp_for_infection=15.0, max_temp_for_infection=30.0)
        state = DiseaseState(susceptible=0.5, exposed=0.0, infectious=0.5, recovered=0.0)
        # Within window
        result_in = model.simulate(state, days=5, temperature=22.0)
        # Below window
        result_out = model.simulate(state, days=5, temperature=5.0)
        assert result_in.final_state.susceptible < result_out.final_state.susceptible

    def test_humidity_trigger(self):
        """High humidity increases disease transmission."""
        model = DiseaseModel(humidity_threshold=60.0)
        state = DiseaseState(susceptible=0.5, exposed=0.0, infectious=0.5, recovered=0.0)
        high_humidity = model.simulate(state, days=5, humidity=85.0)
        low_humidity = model.simulate(state, days=5, humidity=30.0)
        assert high_humidity.final_state.susceptible < low_humidity.final_state.susceptible

    def test_combined_environmental_stress(self):
        """Multiple environmental factors combine to increase transmission."""
        model = DiseaseModel(
            min_temp_for_infection=15.0,
            max_temp_for_infection=30.0,
            humidity_threshold=60.0,
            leaf_wetness_threshold=6.0,
        )
        state = DiseaseState(susceptible=0.5, exposed=0.0, infectious=0.5, recovered=0.0)
        # Optimal conditions
        optimal = model.simulate(
            state, days=5, temperature=22.0, humidity=85.0, leaf_wetness_hours=8.0
        )
        # Poor conditions
        poor = model.simulate(state, days=5, temperature=5.0, humidity=30.0, leaf_wetness_hours=1.0)
        assert optimal.final_state.susceptible < poor.final_state.susceptible


class TestHostPathogenDynamics:
    """Host-pathogen interaction dynamics."""

    def test_host_resistance_reduces_infection(self):
        """Host resistance reduces effective infection rate."""
        model = DiseaseModel(infection_rate=0.5)
        state = DiseaseState(susceptible=0.5, exposed=0.0, infectious=0.5, recovered=0.0)
        # With host resistance
        result_resistant = model.simulate(state, days=5, host_resistance=0.7)
        # Without host resistance
        result_susceptible = model.simulate(state, days=5, host_resistance=0.0)
        assert result_resistant.final_state.susceptible > result_susceptible.final_state.susceptible

    def test_pathogen_virulence_increases_transmission(self):
        """Higher pathogen virulence increases transmission rate."""
        model = DiseaseModel(infection_rate=0.3, pathogen_virulence=0.5)
        state = DiseaseState(susceptible=0.5, exposed=0.0, infectious=0.5, recovered=0.0)
        result = model.simulate(state, days=5)
        # Virulence should amplify transmission
        assert result.final_state.susceptible < 0.5

    def test_recovery_with_immunity_wanes(self):
        """Recovered individuals may lose immunity over time."""
        model = DiseaseModel(recovery_rate=0.3, immunity_waning_rate=0.05)
        state = DiseaseState(susceptible=0.0, exposed=0.0, infectious=0.0, recovered=1.0)
        result = model.simulate(state, days=20)
        # Some recovered should become susceptible again
        assert result.final_state.susceptible > 0.0

    def test_no_immunity_waning_by_default(self):
        """By default, immunity does not wane."""
        model = DiseaseModel(recovery_rate=0.3)
        state = DiseaseState(susceptible=0.0, exposed=0.0, infectious=0.0, recovered=1.0)
        result = model.simulate(state, days=20)
        assert result.final_state.susceptible == pytest.approx(0.0)


class TestDiseaseStateExtended:
    """Extended DiseaseState with environmental fields."""

    def test_default_spore_load(self):
        """Default spore load is zero."""
        state = DiseaseState()
        assert state.spore_load == 0.0

    def test_default_lesion_count(self):
        """Default lesion count is zero."""
        state = DiseaseState()
        assert state.lesion_count == 0.0

    def test_custom_spore_load(self):
        """Custom spore load is stored."""
        state = DiseaseState(susceptible=0.5, spore_load=0.3)
        assert state.spore_load == 0.3


class TestDiseaseModelExtended:
    """Extended DiseaseModel with new parameters."""

    def test_sporulation_parameters(self):
        """Model stores sporulation parameters."""
        model = DiseaseModel(
            sporulation_rate=0.6,
            sporulation_infection_efficiency=0.4,
        )
        assert model.sporulation_rate == 0.6
        assert model.sporulation_infection_efficiency == 0.4

    def test_environmental_parameters(self):
        """Model stores environmental trigger parameters."""
        model = DiseaseModel(
            leaf_wetness_threshold=8.0,
            min_temp_for_infection=12.0,
            max_temp_for_infection=32.0,
            humidity_threshold=55.0,
        )
        assert model.leaf_wetness_threshold == 8.0
        assert model.min_temp_for_infection == 12.0
        assert model.max_temp_for_infection == 32.0
        assert model.humidity_threshold == 55.0

    def test_host_pathogen_parameters(self):
        """Model stores host-pathogen interaction parameters."""
        model = DiseaseModel(
            host_resistance_default=0.3,
            pathogen_virulence=0.7,
            immunity_waning_rate=0.02,
        )
        assert model.host_resistance_default == 0.3
        assert model.pathogen_virulence == 0.7
        assert model.immunity_waning_rate == 0.02

    def test_simulation_with_environmental_data(self):
        """Simulation accepts environmental data parameters."""
        model = DiseaseModel()
        state = DiseaseState(susceptible=0.8, exposed=0.0, infectious=0.2, recovered=0.0)
        result = model.simulate(
            state,
            days=10,
            temperature=22.0,
            humidity=80.0,
            leaf_wetness_hours=8.0,
            host_resistance=0.2,
        )
        assert result.days_simulated == 10
        assert len(result.history) == 10

    def test_simulation_with_sporulation(self):
        """Simulation with sporulation produces new infections."""
        model = DiseaseModel(
            infection_rate=0.0,
            sporulation_rate=0.4,
            sporulation_infection_efficiency=0.6,
        )
        state = DiseaseState(susceptible=0.7, exposed=0.0, infectious=0.3, recovered=0.0)
        result = model.simulate(state, days=10)
        assert result.final_state.susceptible < 0.7


class TestDiseaseModelBackwardCompatibility:
    """Existing API remains functional."""

    def test_basic_simulation_still_works(self):
        """Basic SEIR simulation works."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1, incubation_rate=0.3)
        state = DiseaseState(susceptible=0.9, exposed=0.0, infectious=0.1, recovered=0.0)
        result = model.simulate(state, days=20)
        assert result.days_simulated == 20
        assert result.final_state.recovered > 0.0

    def test_compute_r0_unchanged(self):
        """R0 calculation still works."""
        model = DiseaseModel(infection_rate=0.5, recovery_rate=0.1)
        assert model.compute_r0() == pytest.approx(5.0)

    def test_temperature_stress_factor_unchanged(self):
        """Original temperature stress factor still works."""
        model = DiseaseModel(optimal_temp=25.0, temp_tolerance=10.0)
        assert model.temperature_stress_factor(25.0) == pytest.approx(1.0)
        assert model.temperature_stress_factor(50.0) < 0.1

    def test_humidity_stress_factor_unchanged(self):
        """Original humidity stress factor still works."""
        model = DiseaseModel(optimal_humidity=70.0, humidity_tolerance=30.0)
        assert model.humidity_stress_factor(70.0) == pytest.approx(1.0)
        assert model.humidity_stress_factor(0.0) < 0.1

    def test_vaccination_still_works(self):
        """Vaccination still moves susceptible to recovered."""
        model = DiseaseModel()
        state = DiseaseState(susceptible=1.0, exposed=0.0, infectious=0.0, recovered=0.0)
        result = model.simulate(state, days=5, vaccination_rate=0.1)
        assert result.final_state.recovered > 0.0
        assert result.final_state.susceptible < 1.0

    def test_result_type(self):
        """Result is DiseaseSimulationResult."""
        model = DiseaseModel()
        state = DiseaseState(susceptible=1.0)
        result = model.simulate(state, days=5)
        assert isinstance(result, DiseaseSimulationResult)

    def test_negative_days_raises(self):
        """Negative days raises ValueError."""
        model = DiseaseModel()
        state = DiseaseState(susceptible=1.0)
        with pytest.raises(ValueError, match="Days must be non-negative"):
            model.simulate(state, days=-1)

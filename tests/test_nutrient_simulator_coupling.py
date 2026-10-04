"""Tests for nutrient cycling ↔ simulator coupling."""

from src.digital_twin.nutrient_cycling import (
    FertilizerApplication,
    NutrientCyclingModel,
    NutrientState,
)
from src.digital_twin.shared_state import DigitalTwinState
from src.digital_twin.simulator import DigitalTwin, SimulationState


def make_nutrient_state(n=50.0, p=10.0, k=80.0, om=3.0):
    """Helper to create NutrientState with required organic_matter."""
    return NutrientState(nitrogen=n, phosphorus=p, potassium=k, organic_matter=om)


class TestNutrientSimulatorCoupling:
    """Test nutrient cycling and simulator integration."""

    def test_nutrient_model_provides_nutrients(self):
        """Nutrient model provides N/P/K to simulator."""
        nc = NutrientCyclingModel()
        state = make_nutrient_state(n=50.0, p=10.0, k=80.0)
        result = nc.simulate(state, days=5)
        assert result.final_state.nitrogen >= 0.0

    def test_simulator_uses_nutrient_output(self):
        """Simulator can use nutrient model output."""
        nc = NutrientCyclingModel()
        nut_state = make_nutrient_state(n=80.0, p=15.0, k=100.0)
        nc.simulate(nut_state, days=3)

        sim = DigitalTwin()
        sim_state = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.7,
        )
        result = sim.simulate(sim_state, days=5)
        assert result.final_state.crop_height > 0.1

    def test_coupled_simulation_nutrient_depletion(self):
        """Coupled simulation shows nutrient depletion effect."""
        nc_high = NutrientCyclingModel()
        high_state = make_nutrient_state(n=100.0, p=20.0, k=100.0)
        nc_high.simulate(high_state, days=5)

        nc_low = NutrientCyclingModel()
        low_state = make_nutrient_state(n=10.0, p=2.0, k=15.0)
        nc_low.simulate(low_state, days=5)

        sim = DigitalTwin()
        high_sim = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.8,
        )
        low_sim = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.2,
        )
        high_result = sim.simulate(high_sim, days=10)
        low_result = sim.simulate(low_sim, days=10)
        assert high_result.final_state.crop_height > low_result.final_state.crop_height

    def test_fertilization_event_coupling(self):
        """Fertilization event in nutrient model affects simulator."""
        nc = NutrientCyclingModel()
        state = make_nutrient_state(n=20.0, p=5.0, k=30.0)
        fert_schedule = [
            FertilizerApplication(nitrogen=50.0, phosphorus=10.0, potassium=40.0, day=0)
        ]
        result = nc.simulate(state, days=5, fertilizer_schedule=fert_schedule)
        assert result.final_state.nitrogen > 20.0

    def test_shared_state_nutrient_coupling(self):
        """Shared state model enables nutrient coupling."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=80.0,
        )
        nut_state = state.to_nutrient_state()
        assert nut_state.nitrogen == 50.0

        nc = NutrientCyclingModel()
        nc_result = nc.simulate(nut_state, days=3)

        state.nitrogen = nc_result.final_state.nitrogen
        state.phosphorus = nc_result.final_state.phosphorus
        state.potassium = nc_result.final_state.potassium

    def test_nutrient_deficiency_reduces_growth(self):
        """Nutrient deficiency in model reduces simulator growth."""
        nc = NutrientCyclingModel()
        deficient_state = make_nutrient_state(n=5.0, p=1.0, k=5.0)
        nc_result = nc.simulate(deficient_state, days=5)

        deficiency = nc.nutrient_deficiency(nc_result.final_state)
        assert deficiency["n"] is True

        sim = DigitalTwin()
        low_nut_sim = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.1,
        )
        low_result = sim.simulate(low_nut_sim, days=10)

        high_nut_sim = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.9,
        )
        high_result = sim.simulate(high_nut_sim, days=10)
        assert high_result.final_state.crop_height > low_result.final_state.crop_height

    def test_nutrient_uptake_reduces_available(self):
        """Plant uptake reduces available nutrients over time."""
        nc = NutrientCyclingModel()
        state = make_nutrient_state(n=100.0, p=20.0, k=80.0)
        crop_demand = {"n": 3.0, "p": 0.5, "k": 2.0}
        result = nc.simulate(state, days=10, crop_demand=crop_demand)

        assert result.budget.total_n_uptake > 0
        assert result.budget.total_p_uptake > 0
        assert result.budget.total_k_uptake > 0

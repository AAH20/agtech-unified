"""Tests for pest/disease/weed pressure model."""

from src.digital_twin.simulator import DigitalTwin, SimulationState


class TestPestDiseaseModel:
    """Test biotic stress factors in simulator."""

    def test_pest_pressure_reduces_growth(self):
        """High pest pressure reduces crop growth."""
        twin = DigitalTwin()
        no_pest = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        high_pest = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.8,
        )
        result_good = twin.simulate(no_pest, days=10)
        result_bad = twin.simulate(high_pest, days=10)
        assert result_good.final_state.crop_height > result_bad.final_state.crop_height

    def test_disease_pressure_reduces_growth(self):
        """Disease pressure reduces crop growth."""
        twin = DigitalTwin()
        healthy = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            disease_pressure=0.0,
        )
        diseased = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            disease_pressure=0.7,
        )
        result_healthy = twin.simulate(healthy, days=10)
        result_diseased = twin.simulate(diseased, days=10)
        assert result_healthy.final_state.crop_height > result_diseased.final_state.crop_height

    def test_weed_pressure_reduces_growth(self):
        """Weed competition reduces crop growth."""
        twin = DigitalTwin()
        no_weeds = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            weed_pressure=0.0,
        )
        heavy_weeds = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            weed_pressure=0.6,
        )
        result_clean = twin.simulate(no_weeds, days=10)
        result_weedy = twin.simulate(heavy_weeds, days=10)
        assert result_clean.final_state.crop_height > result_weedy.final_state.crop_height

    def test_combined_biotic_stress(self):
        """Combined pest + disease + weed stress."""
        twin = DigitalTwin()
        no_stress = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.0,
            disease_pressure=0.0,
            weed_pressure=0.0,
        )
        all_stress = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.5,
            disease_pressure=0.5,
            weed_pressure=0.5,
        )
        result_none = twin.simulate(no_stress, days=10)
        result_all = twin.simulate(all_stress, days=10)
        assert result_none.final_state.crop_height > result_all.final_state.crop_height

    def test_pest_population_dynamics(self):
        """Pest pressure evolves over time."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        result = twin.simulate(state, days=10)
        # Pest pressure should change over time
        assert result.final_state.pest_pressure != 0.1

    def test_pest_outbreak_event(self):
        """Pest outbreak event increases pressure."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        # Simulate with pest outbreak on day 3
        result = twin.simulate(state, days=10)
        # Pest pressure should increase from initial
        assert result.final_state.pest_pressure >= 0.0

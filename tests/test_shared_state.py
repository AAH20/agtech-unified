"""Tests for shared digital twin state model."""

from src.digital_twin.nutrient_cycling import NutrientState
from src.digital_twin.shared_state import DigitalTwinState, StateSnapshot
from src.digital_twin.simulator import SimulationState
from src.digital_twin.water_balance import WaterBalanceState


class TestDigitalTwinState:
    """Test unified state model."""

    def test_create_minimal_state(self):
        """Create state with minimal required fields."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
        )
        assert state.soil_moisture == 0.3
        assert state.temperature == 25.0
        assert state.crop_height == 0.5
        assert state.nitrogen == 0.0
        assert state.phosphorus == 0.0
        assert state.potassium == 0.0

    def test_create_full_state(self):
        """Create state with all fields."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=80.0,
            organic_matter=3.0,
            pH=6.5,
            pest_pressure=0.1,
            disease_pressure=0.0,
            weed_pressure=0.05,
        )
        assert state.nitrogen == 50.0
        assert state.organic_matter == 3.0
        assert state.pest_pressure == 0.1

    def test_to_simulation_state(self):
        """Convert to SimulationState for backward compat."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=80.0,
        )
        sim_state = state.to_simulation_state()
        assert isinstance(sim_state, SimulationState)
        assert sim_state.soil_moisture == 0.3
        assert sim_state.temperature == 25.0
        assert sim_state.crop_height == 0.5
        # nutrient_level is average of N/P/K normalized
        assert sim_state.nutrient_level > 0.0

    def test_from_simulation_state(self):
        """Create from SimulationState."""
        sim_state = SimulationState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
        )
        state = DigitalTwinState.from_simulation_state(sim_state)
        assert state.soil_moisture == 0.3
        assert state.temperature == 25.0
        assert state.crop_height == 0.5

    def test_to_water_balance_state(self):
        """Convert to WaterBalanceState."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
        )
        wb_state = state.to_water_balance_state()
        assert isinstance(wb_state, WaterBalanceState)
        assert wb_state.soil_moisture == 0.3

    def test_from_water_balance_state(self):
        """Create from WaterBalanceState."""
        wb_state = WaterBalanceState(
            soil_moisture=0.25,
            field_capacity=0.30,
            wilting_point=0.10,
            root_depth=0.5,
        )
        state = DigitalTwinState.from_water_balance_state(wb_state)
        assert state.soil_moisture == 0.25

    def test_to_nutrient_state(self):
        """Convert to NutrientState."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=80.0,
            organic_matter=3.0,
            pH=6.5,
        )
        nut_state = state.to_nutrient_state()
        assert isinstance(nut_state, NutrientState)
        assert nut_state.nitrogen == 50.0
        assert nut_state.phosphorus == 10.0
        assert nut_state.potassium == 80.0

    def test_from_nutrient_state(self):
        """Create from NutrientState."""
        nut_state = NutrientState(
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=80.0,
            organic_matter=3.0,
            pH=6.5,
        )
        state = DigitalTwinState.from_nutrient_state(nut_state)
        assert state.nitrogen == 50.0
        assert state.phosphorus == 10.0
        assert state.potassium == 80.0

    def test_clone(self):
        """Clone creates independent copy."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
        )
        clone = state.clone()
        assert clone.soil_moisture == 0.3
        assert clone.nitrogen == 50.0
        clone.soil_moisture = 0.1
        assert state.soil_moisture == 0.3

    def test_to_dict(self):
        """Serialize to dict."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
        )
        d = state.to_dict()
        assert d["soil_moisture"] == 0.3
        assert d["temperature"] == 25.0
        assert d["nitrogen"] == 50.0

    def test_from_dict(self):
        """Deserialize from dict."""
        d = {
            "soil_moisture": 0.3,
            "temperature": 25.0,
            "crop_height": 0.5,
            "nitrogen": 50.0,
        }
        state = DigitalTwinState.from_dict(d)
        assert state.soil_moisture == 0.3
        assert state.nitrogen == 50.0

    def test_nutrient_level_computed(self):
        """Nutrient level is computed from N/P/K."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=100.0,
            phosphorus=20.0,
            potassium=80.0,
        )
        # nutrient_level should be a normalized aggregate
        level = state.nutrient_level
        assert 0.0 <= level <= 1.0

    def test_stress_factor_combined(self):
        """Combined stress factor from all stressors."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=40.0,
            pest_pressure=0.5,
        )
        stress = state.combined_stress_factor()
        assert 0.0 <= stress <= 1.0
        # Pest pressure should reduce stress factor
        no_pest = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
            nitrogen=50.0,
            phosphorus=10.0,
            potassium=40.0,
            pest_pressure=0.0,
        )
        assert stress < no_pest.combined_stress_factor()


class TestStateSnapshot:
    """Test state snapshot for persistence."""

    def test_create_snapshot(self):
        """Create snapshot with timestamp."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
        )
        snapshot = StateSnapshot(state=state, day=10, label="mid-season")
        assert snapshot.day == 10
        assert snapshot.label == "mid-season"
        assert snapshot.state.soil_moisture == 0.3

    def test_snapshot_to_dict(self):
        """Serialize snapshot."""
        state = DigitalTwinState(
            soil_moisture=0.3,
            temperature=25.0,
            crop_height=0.5,
        )
        snapshot = StateSnapshot(state=state, day=10)
        d = snapshot.to_dict()
        assert d["day"] == 10
        assert d["state"]["soil_moisture"] == 0.3

    def test_snapshot_from_dict(self):
        """Deserialize snapshot."""
        d = {
            "day": 10,
            "label": "test",
            "state": {
                "soil_moisture": 0.3,
                "temperature": 25.0,
                "crop_height": 0.5,
            },
        }
        snapshot = StateSnapshot.from_dict(d)
        assert snapshot.day == 10
        assert snapshot.state.soil_moisture == 0.3

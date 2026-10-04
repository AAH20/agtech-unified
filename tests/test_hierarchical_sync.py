"""Tests for hierarchical multi-twin synchronization."""

import pytest

from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.digital_twin.sync import HierarchicalSynchronizer, TwinLevel


def make_twin(soil_moisture=0.5, temperature=25.0, crop_height=0.1, nutrient_level=0.5):
    """Helper to create a DigitalTwin with a given initial state."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=soil_moisture,
        temperature=temperature,
        crop_height=crop_height,
        nutrient_level=nutrient_level,
    )
    from src.integration.farm_state import FarmState

    twin.ingest_sensor_data(
        FarmState(
            soil_moisture=state.soil_moisture,
            temperature=state.temperature,
            crop_height=state.crop_height,
            nutrient_level=state.nutrient_level,
            pest_pressure=getattr(state, "pest_pressure", 0.0),
        )
    )
    return twin


class TestHierarchicalSynchronizer:
    """Test hierarchical multi-twin synchronization."""

    def test_register_twin(self):
        """Twin can be registered in the hierarchy."""
        sync = HierarchicalSynchronizer()
        twin = make_twin()
        sync.register_twin("field_1", twin, TwinLevel.FIELD, weight=10.0, parent_id="farm_1")
        assert "field_1" in sync._registrations

    def test_register_twin_invalid_weight(self):
        """Registration with non-positive weight raises ValueError."""
        sync = HierarchicalSynchronizer()
        twin = make_twin()
        with pytest.raises(ValueError, match="Weight must be positive"):
            sync.register_twin("field_1", twin, TwinLevel.FIELD, weight=0.0)

    def test_unregister_twin(self):
        """Twin can be unregistered."""
        sync = HierarchicalSynchronizer()
        twin = make_twin()
        sync.register_twin("field_1", twin, TwinLevel.FIELD)
        sync.unregister_twin("field_1")
        assert "field_1" not in sync._registrations

    def test_get_twins_at_level(self):
        """Can filter twins by hierarchy level."""
        sync = HierarchicalSynchronizer()
        sync.register_twin("field_1", make_twin(), TwinLevel.FIELD)
        sync.register_twin("field_2", make_twin(), TwinLevel.FIELD)
        sync.register_twin("farm_1", make_twin(), TwinLevel.FARM)
        sync.register_twin("regional", make_twin(), TwinLevel.REGIONAL)
        assert set(sync.get_twins_at_level(TwinLevel.FIELD)) == {"field_1", "field_2"}
        assert sync.get_twins_at_level(TwinLevel.FARM) == ["farm_1"]
        assert sync.get_twins_at_level(TwinLevel.REGIONAL) == ["regional"]

    def test_get_children(self):
        """Can get children of a parent twin."""
        sync = HierarchicalSynchronizer()
        sync.register_twin("field_1", make_twin(), TwinLevel.FIELD, parent_id="farm_1")
        sync.register_twin("field_2", make_twin(), TwinLevel.FIELD, parent_id="farm_1")
        sync.register_twin("field_3", make_twin(), TwinLevel.FIELD, parent_id="farm_2")
        children = sync.get_children("farm_1")
        assert set(children) == {"field_1", "field_2"}

    def test_sync_hierarchical_empty_raises(self):
        """Sync with no twins raises ValueError."""
        sync = HierarchicalSynchronizer()
        with pytest.raises(ValueError, match="No twins registered"):
            sync.sync_hierarchical()

    def test_sync_hierarchical_field_level(self):
        """Field twins sync within their farm group."""
        sync = HierarchicalSynchronizer(convergence_factor=1.0)
        sync.register_twin(
            "field_1", make_twin(soil_moisture=0.2), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin(
            "field_2", make_twin(soil_moisture=0.8), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin("farm_1", make_twin(soil_moisture=0.5), TwinLevel.FARM)
        result = sync.sync_hierarchical()
        assert "farm:farm_1" in result.level_results
        # After full convergence, both fields should be at the mean
        assert result.regional_mean.soil_moisture == pytest.approx(0.5)

    def test_sync_hierarchical_multi_farm(self):
        """Multiple farms sync toward regional mean."""
        sync = HierarchicalSynchronizer(convergence_factor=1.0)
        sync.register_twin(
            "field_1", make_twin(soil_moisture=0.2), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin(
            "field_2", make_twin(soil_moisture=0.4), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin(
            "field_3", make_twin(soil_moisture=0.6), TwinLevel.FIELD, parent_id="farm_2"
        )
        sync.register_twin(
            "field_4", make_twin(soil_moisture=0.8), TwinLevel.FIELD, parent_id="farm_2"
        )
        sync.register_twin("farm_1", make_twin(soil_moisture=0.3), TwinLevel.FARM)
        sync.register_twin("farm_2", make_twin(soil_moisture=0.7), TwinLevel.FARM)
        result = sync.sync_hierarchical()
        assert "farm:farm_1" in result.level_results
        assert "farm:farm_2" in result.level_results
        assert "regional" in result.level_results
        # Regional mean should be the mean of all field values
        assert result.regional_mean.soil_moisture == pytest.approx(0.5)

    def test_sync_hierarchical_with_regional_twin(self):
        """Regional twin is updated to the regional mean."""
        sync = HierarchicalSynchronizer(convergence_factor=1.0)
        sync.register_twin(
            "field_1", make_twin(soil_moisture=0.2), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin(
            "field_2", make_twin(soil_moisture=0.8), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin("farm_1", make_twin(soil_moisture=0.5), TwinLevel.FARM)
        regional_twin = make_twin(soil_moisture=0.9)
        sync.register_twin("regional", regional_twin, TwinLevel.REGIONAL)
        sync.sync_hierarchical()
        # Regional twin should have moved toward the mean
        regional_state = regional_twin.get_current_state()
        assert regional_state is not None
        assert regional_state.soil_moisture == pytest.approx(0.5)

    def test_merge_states_equal_weights(self):
        """Merge states with equal weights produces simple mean."""
        sync = HierarchicalSynchronizer()
        states = {
            "a": SimulationState(
                soil_moisture=0.2, temperature=20.0, crop_height=0.1, nutrient_level=0.3
            ),
            "b": SimulationState(
                soil_moisture=0.6, temperature=30.0, crop_height=0.3, nutrient_level=0.7
            ),
        }
        merged = sync.merge_states(states)
        assert merged.soil_moisture == pytest.approx(0.4)
        assert merged.temperature == pytest.approx(25.0)
        assert merged.crop_height == pytest.approx(0.2)
        assert merged.nutrient_level == pytest.approx(0.5)

    def test_merge_states_custom_weights(self):
        """Merge states with custom weights produces weighted mean."""
        sync = HierarchicalSynchronizer()
        states = {
            "a": SimulationState(
                soil_moisture=0.2, temperature=20.0, crop_height=0.1, nutrient_level=0.3
            ),
            "b": SimulationState(
                soil_moisture=0.8, temperature=40.0, crop_height=0.5, nutrient_level=0.9
            ),
        }
        weights = {"a": 3.0, "b": 1.0}
        merged = sync.merge_states(states, weights=weights)
        # Weighted mean: (3*0.2 + 1*0.8) / 4 = 0.35
        assert merged.soil_moisture == pytest.approx(0.35)
        assert merged.temperature == pytest.approx(25.0)

    def test_merge_states_empty_raises(self):
        """Merge with no states raises ValueError."""
        sync = HierarchicalSynchronizer()
        with pytest.raises(ValueError, match="No states to merge"):
            sync.merge_states({})

    def test_merge_states_uses_registration_weights(self):
        """Merge uses registration weights when no explicit weights given."""
        sync = HierarchicalSynchronizer()
        sync.register_twin("a", make_twin(), TwinLevel.FIELD, weight=3.0)
        sync.register_twin("b", make_twin(), TwinLevel.FIELD, weight=1.0)
        states = {
            "a": SimulationState(
                soil_moisture=0.2, temperature=20.0, crop_height=0.1, nutrient_level=0.3
            ),
            "b": SimulationState(
                soil_moisture=0.8, temperature=40.0, crop_height=0.5, nutrient_level=0.9
            ),
        }
        merged = sync.merge_states(states)
        assert merged.soil_moisture == pytest.approx(0.35)

    def test_hierarchical_sync_result_structure(self):
        """HierarchicalSyncResult contains expected fields."""
        sync = HierarchicalSynchronizer(convergence_factor=1.0)
        sync.register_twin(
            "field_1", make_twin(soil_moisture=0.3), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin(
            "field_2", make_twin(soil_moisture=0.7), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin("farm_1", make_twin(soil_moisture=0.5), TwinLevel.FARM)
        result = sync.sync_hierarchical()
        assert hasattr(result, "level_results")
        assert hasattr(result, "regional_mean")
        assert hasattr(result, "convergence_rounds")
        assert result.convergence_rounds == 1

    def test_field_only_hierarchy(self):
        """Sync works with only field-level twins (no farm/regional)."""
        sync = HierarchicalSynchronizer(convergence_factor=1.0)
        sync.register_twin("field_1", make_twin(soil_moisture=0.2), TwinLevel.FIELD)
        sync.register_twin("field_2", make_twin(soil_moisture=0.8), TwinLevel.FIELD)
        result = sync.sync_hierarchical()
        assert result.regional_mean.soil_moisture == pytest.approx(0.5)

    def test_convergence_factor_zero_no_change(self):
        """Convergence factor of 0 means no state change."""
        sync = HierarchicalSynchronizer(convergence_factor=0.0)
        sync.register_twin(
            "field_1", make_twin(soil_moisture=0.2), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin(
            "field_2", make_twin(soil_moisture=0.8), TwinLevel.FIELD, parent_id="farm_1"
        )
        sync.register_twin("farm_1", make_twin(soil_moisture=0.5), TwinLevel.FARM)
        result = sync.sync_hierarchical()
        # With factor=0, states should not change
        assert result.regional_mean.soil_moisture == pytest.approx(0.5)

"""Tests for multi-twin synchronization."""

import pytest

from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.digital_twin.sync import TwinSynchronizer


def make_twin(soil_moisture=0.5, temperature=25.0, crop_height=0.1, nutrient_level=0.5):
    """Helper to create a DigitalTwin with a given initial state."""
    twin = DigitalTwin()
    state = SimulationState(
        soil_moisture=soil_moisture,
        temperature=temperature,
        crop_height=crop_height,
        nutrient_level=nutrient_level,
    )
    twin.ingest_sensor_data(_to_farm_state(state))
    return twin, state


def _to_farm_state(state):
    """Convert SimulationState to FarmState for ingestion."""
    from src.integration.farm_state import FarmState

    return FarmState(
        soil_moisture=state.soil_moisture,
        temperature=state.temperature,
        crop_height=state.crop_height,
        nutrient_level=state.nutrient_level,
        pest_pressure=getattr(state, "pest_pressure", 0.0),
    )


class TestTwinSynchronizer:
    """Test multi-twin synchronization."""

    def test_register_twin(self):
        """Twin can be registered with synchronizer."""
        sync = TwinSynchronizer()
        twin, _ = make_twin()
        sync.register_twin("field_1", twin)
        assert "field_1" in sync.twins

    def test_unregister_twin(self):
        """Twin can be unregistered."""
        sync = TwinSynchronizer()
        twin, _ = make_twin()
        sync.register_twin("field_1", twin)
        sync.unregister_twin("field_1")
        assert "field_1" not in sync.twins

    def test_sync_all_converges(self):
        """Synchronization moves all twins toward mean state."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(soil_moisture=0.2)
        twin2, _ = make_twin(soil_moisture=0.8)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all()
        assert result.sync_count == 2
        # After sync, moisture values should be closer together
        m1 = sync.get_twin_state("field_1").soil_moisture
        m2 = sync.get_twin_state("field_2").soil_moisture
        assert abs(m1 - m2) < abs(0.2 - 0.8)

    def test_sync_all_with_initial_states(self):
        """Sync uses provided initial states."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(soil_moisture=0.2)
        twin2, _ = make_twin(soil_moisture=0.8)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        states = {
            "field_1": SimulationState(
                soil_moisture=0.2, temperature=25.0, crop_height=0.1, nutrient_level=0.5
            ),
            "field_2": SimulationState(
                soil_moisture=0.8, temperature=25.0, crop_height=0.1, nutrient_level=0.5
            ),
        }
        result = sync.sync_all(initial_states=states)
        assert result.sync_count == 2

    def test_sync_all_empty_raises(self):
        """Sync with no twins raises ValueError."""
        sync = TwinSynchronizer()
        with pytest.raises(ValueError, match="No twins registered"):
            sync.sync_all()

    def test_sync_convergence_factor(self):
        """Convergence factor controls sync strength."""
        sync_weak = TwinSynchronizer(convergence_factor=0.1)
        sync_strong = TwinSynchronizer(convergence_factor=0.9)
        twin1_w, _ = make_twin(soil_moisture=0.2)
        twin2_w, _ = make_twin(soil_moisture=0.8)
        twin1_s, _ = make_twin(soil_moisture=0.2)
        twin2_s, _ = make_twin(soil_moisture=0.8)
        sync_weak.register_twin("field_1", twin1_w)
        sync_weak.register_twin("field_2", twin2_w)
        sync_strong.register_twin("field_1", twin1_s)
        sync_strong.register_twin("field_2", twin2_s)
        sync_weak.sync_all()
        sync_strong.sync_all()
        # Strong convergence should produce values closer to mean
        m1_w = sync_weak.get_twin_state("field_1").soil_moisture
        m2_w = sync_weak.get_twin_state("field_2").soil_moisture
        m1_s = sync_strong.get_twin_state("field_1").soil_moisture
        m2_s = sync_strong.get_twin_state("field_2").soil_moisture
        assert abs(m1_s - m2_s) < abs(m1_w - m2_w)

    def test_sync_updates_twin_state(self):
        """Sync updates the twin's internal current state."""
        sync = TwinSynchronizer()
        twin, _ = make_twin(soil_moisture=0.2)
        sync.register_twin("field_1", twin)
        other, _ = make_twin(soil_moisture=0.8)
        sync.register_twin("field_2", other)
        sync.sync_all()
        state = sync.get_twin_state("field_1")
        assert state is not None
        assert state.soil_moisture != 0.2  # Should have moved toward mean

    def test_sync_result_contains_mean_state(self):
        """SyncResult includes the computed mean state."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(soil_moisture=0.3, temperature=20.0)
        twin2, _ = make_twin(soil_moisture=0.7, temperature=30.0)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all()
        assert result.mean_state.soil_moisture == pytest.approx(0.5)
        assert result.mean_state.temperature == pytest.approx(25.0)

    def test_sync_with_three_twins(self):
        """Sync works with more than two twins."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(soil_moisture=0.1)
        twin2, _ = make_twin(soil_moisture=0.5)
        twin3, _ = make_twin(soil_moisture=0.9)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        sync.register_twin("field_3", twin3)
        result = sync.sync_all()
        assert result.sync_count == 3
        assert result.mean_state.soil_moisture == pytest.approx(0.5)

    def test_sync_preserves_total_water(self):
        """Weighted average preserves total water across twins."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(soil_moisture=0.2)
        twin2, _ = make_twin(soil_moisture=0.6)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        total_before = 0.2 + 0.6
        sync.sync_all()
        total_after = (
            sync.get_twin_state("field_1").soil_moisture
            + sync.get_twin_state("field_2").soil_moisture
        )
        # Total should be approximately preserved
        assert total_after == pytest.approx(total_before, abs=0.15)

    def test_sync_different_crop_heights(self):
        """Sync handles different crop heights."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(crop_height=0.2)
        twin2, _ = make_twin(crop_height=0.8)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all()
        assert result.mean_state.crop_height == pytest.approx(0.5)

    def test_sync_different_temperatures(self):
        """Sync handles different temperatures."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(temperature=15.0)
        twin2, _ = make_twin(temperature=35.0)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all()
        assert result.mean_state.temperature == pytest.approx(25.0)

    def test_sync_different_nutrient_levels(self):
        """Sync handles different nutrient levels."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(nutrient_level=0.2)
        twin2, _ = make_twin(nutrient_level=0.6)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all()
        assert result.mean_state.nutrient_level == pytest.approx(0.4)

    def test_sync_with_biotic_stress(self):
        """Sync handles biotic stress factors."""
        sync = TwinSynchronizer()
        twin1 = DigitalTwin()
        twin2 = DigitalTwin()
        state1 = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.1,
            disease_pressure=0.2,
            weed_pressure=0.0,
        )
        state2 = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.3,
            disease_pressure=0.0,
            weed_pressure=0.1,
        )
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all(initial_states={"field_1": state1, "field_2": state2})
        assert result.mean_state.pest_pressure == pytest.approx(0.2)
        assert result.mean_state.disease_pressure == pytest.approx(0.1)
        assert result.mean_state.weed_pressure == pytest.approx(0.05)

    def test_sync_result_max_difference(self):
        """SyncResult reports max difference after sync."""
        sync = TwinSynchronizer()
        twin1, _ = make_twin(soil_moisture=0.2)
        twin2, _ = make_twin(soil_moisture=0.8)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        result = sync.sync_all()
        assert result.max_difference >= 0.0
        assert result.max_difference <= 0.6  # Should be less than initial difference

    def test_sync_idempotent(self):
        """Syncing twice with full convergence produces same result as syncing once."""
        sync = TwinSynchronizer(convergence_factor=1.0)
        twin1, _ = make_twin(soil_moisture=0.2)
        twin2, _ = make_twin(soil_moisture=0.8)
        sync.register_twin("field_1", twin1)
        sync.register_twin("field_2", twin2)
        sync.sync_all()
        state1_after_first = sync.get_twin_state("field_1").soil_moisture
        sync.sync_all()
        state1_after_second = sync.get_twin_state("field_1").soil_moisture
        # With full convergence, second sync is a no-op
        assert abs(state1_after_second - state1_after_first) < 1e-9

    def test_get_twin_state_unknown(self):
        """Getting state for unregistered twin returns None."""
        sync = TwinSynchronizer()
        assert sync.get_twin_state("unknown") is None

    def test_sync_with_single_twin(self):
        """Sync with single twin is a no-op."""
        sync = TwinSynchronizer()
        twin, _ = make_twin(soil_moisture=0.5)
        sync.register_twin("field_1", twin)
        result = sync.sync_all()
        assert result.sync_count == 1
        assert result.mean_state.soil_moisture == pytest.approx(0.5)

"""Tests for DigitalTwin ingesting sensor data (INT-015)."""

import pytest

from src.digital_twin.simulator import DigitalTwin
from src.integration.farm_state import FarmState


class TestDigitalTwinIngest:
    """DigitalTwin can ingest FarmState sensor data."""

    def test_ingest_farm_state_updates_simulation(self):
        """Ingesting FarmState updates the twin's internal simulation state."""
        twin = DigitalTwin()
        state = FarmState(
            soil_moisture=0.6,
            temperature=22.0,
            crop_height=0.4,
            nutrient_level=0.7,
            pest_pressure=0.1,
        )

        twin.ingest_sensor_data(state)

        sim_state = twin.get_current_state()
        assert sim_state is not None
        assert sim_state.soil_moisture == pytest.approx(0.6)
        assert sim_state.temperature == pytest.approx(22.0)
        assert sim_state.crop_height == pytest.approx(0.4)
        assert sim_state.nutrient_level == pytest.approx(0.7)

    def test_ingest_then_simulate(self):
        """After ingesting sensor data, simulation starts from that state."""
        twin = DigitalTwin()
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )

        twin.ingest_sensor_data(state)
        result = twin.simulate_from_current(days=5)

        assert result.days_simulated == 5
        assert result.final_state.crop_height > 0.1

    def test_ingest_invalid_state_raises(self):
        """Ingesting invalid sensor data raises ValueError."""
        twin = DigitalTwin()
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
            pest_pressure=0.0,
        )
        state.soil_moisture = 1.5  # Corrupt

        with pytest.raises(ValueError):
            twin.ingest_sensor_data(state)

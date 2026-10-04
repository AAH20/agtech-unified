"""Tests for IoT orchestrator: sensor data → event_bus → FarmState."""

import pytest

from src.integration.event_bus import EventBus, EventType
from src.iot.iot_orchestrator import IoTOrchestrator


class TestIoTOrchestrator:
    """IoTOrchestrator bridges sensor data to the event bus."""

    def test_process_reading_publishes_event(self):
        """Processing a sensor reading publishes SENSOR_READING_RECEIVED."""
        bus = EventBus()
        orch = IoTOrchestrator(bus=bus)
        reading = {"sensor_id": "s1", "value": 0.35, "unit": "ratio", "metric": "soil_moisture"}

        orch.process_reading(reading)

        events = bus.get_history(EventType.SENSOR_READING_RECEIVED)
        assert len(events) == 1
        assert events[0].source == "iot.orchestrator"
        assert events[0].payload["sensor_id"] == "s1"

    def test_process_reading_stores_farm_state(self):
        """Processing a reading updates the orchestrator's FarmState."""
        bus = EventBus()
        orch = IoTOrchestrator(bus=bus)
        reading = {"sensor_id": "s1", "value": 0.35, "unit": "ratio", "metric": "soil_moisture"}

        orch.process_reading(reading)

        state = orch.get_farm_state()
        assert state is not None
        assert state.soil_moisture == pytest.approx(0.35)

    def test_invalid_reading_raises(self):
        """An out-of-range reading raises ValueError from FarmState validation."""
        bus = EventBus()
        orch = IoTOrchestrator(bus=bus)
        reading = {"sensor_id": "s1", "value": 1.5, "unit": "ratio", "metric": "soil_moisture"}

        with pytest.raises(ValueError):
            orch.process_reading(reading)

    def test_subscribe_to_bus(self):
        """Orchestrator can subscribe a handler to an event type."""
        bus = EventBus()
        orch = IoTOrchestrator(bus=bus)
        received = []
        orch.subscribe(EventType.SENSOR_READING_RECEIVED, received.append)

        orch.process_reading(
            {"sensor_id": "s1", "value": 0.5, "unit": "", "metric": "soil_moisture"}
        )

        assert len(received) == 1

    def test_multiple_readings_update_state(self):
        """Multiple readings for the same metric overwrite the FarmState value."""
        bus = EventBus()
        orch = IoTOrchestrator(bus=bus)

        orch.process_reading(
            {"sensor_id": "s1", "value": 0.3, "unit": "", "metric": "soil_moisture"}
        )
        orch.process_reading(
            {"sensor_id": "s1", "value": 0.6, "unit": "", "metric": "soil_moisture"}
        )

        assert orch.get_farm_state().soil_moisture == pytest.approx(0.6)

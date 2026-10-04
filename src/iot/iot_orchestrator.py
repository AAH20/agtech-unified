"""IoT orchestrator: bridges sensor data to the event bus.

Subscribes to data_pipeline events, validates readings through FarmState,
and publishes SENSOR_READING_RECEIVED events for downstream consumers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.integration.farm_state import FarmState

logger = logging.getLogger(__name__)


class IoTOrchestrator:
    """Orchestrates IoT sensor data flow through the event bus.

    Accepts sensor readings, validates them via FarmState, and publishes
    SENSOR_READING_RECEIVED events so decision_support and digital_twin
    can react without direct coupling to the IoT module.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._farm_state: Optional[FarmState] = None

    @property
    def bus(self) -> EventBus:
        return self._bus

    def process_reading(self, reading: Dict[str, Any]) -> FarmState:
        """Validate a sensor reading and publish it to the event bus.

        Args:
            reading: Dict with keys: sensor_id, value, unit, metric

        Returns:
            The updated FarmState after incorporating this reading.

        Raises:
            ValueError: If the reading produces an invalid FarmState.
        """
        metric = reading.get("metric", "soil_moisture")
        value = reading.get("value", 0.0)

        # Build updated state
        state_dict = {
            "soil_moisture": 0.5,
            "temperature": 20.0,
            "crop_height": 0.5,
            "nutrient_level": 0.5,
            "pest_pressure": 0.0,
        }
        if self._farm_state is not None:
            state_dict = {
                "soil_moisture": self._farm_state.soil_moisture,
                "temperature": self._farm_state.temperature,
                "crop_height": self._farm_state.crop_height,
                "nutrient_level": self._farm_state.nutrient_level,
                "pest_pressure": self._farm_state.pest_pressure,
            }
        state_dict[metric] = value

        new_state = FarmState(**state_dict)
        self._farm_state = new_state

        event = DomainEvent(
            event_type=EventType.SENSOR_READING_RECEIVED,
            source="iot.orchestrator",
            payload={
                "sensor_id": reading.get("sensor_id", "unknown"),
                "metric": metric,
                "value": value,
                "unit": reading.get("unit", ""),
            },
        )
        self._bus.publish(event)
        return new_state

    def get_farm_state(self) -> Optional[FarmState]:
        """Return the current FarmState, or None if no readings processed."""
        return self._farm_state

    def subscribe(self, event_type: str, handler) -> None:
        """Subscribe a handler to an event type on the bus."""
        self._bus.subscribe(event_type, handler)

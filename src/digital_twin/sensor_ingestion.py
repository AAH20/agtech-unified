"""Sensor data ingestion for agricultural digital twin.

Provides interfaces for real-time sensor data (soil moisture probes,
weather stations, drones) to update the digital twin state through
MQTT/WebSocket ingestion and state reconciliation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of agricultural sensors."""

    SOIL_MOISTURE = "soil_moisture"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    NITROGEN = "nitrogen"
    PHOSPHORUS = "phosphorus"
    POTASSIUM = "potassium"
    PH = "ph"
    RAINFALL = "rainfall"
    WIND_SPEED = "wind_speed"
    SOLAR_RADIATION = "solar_radiation"
    LEAF_WETNESS = "leaf_wetness"
    UNKNOWN = "unknown"


@dataclass
class SensorReading:
    """A single sensor reading."""

    sensor_id: str
    sensor_type: SensorType
    value: float
    timestamp: float
    unit: str = ""
    field_id: Optional[str] = None
    confidence: float = 1.0  # 0-1, for weighted reconciliation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reading."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "unit": self.unit,
            "field_id": self.field_id,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SensorReading:
        """Deserialize reading."""
        return cls(
            sensor_id=d["sensor_id"],
            sensor_type=SensorType(d.get("sensor_type", "unknown")),
            value=d["value"],
            timestamp=d["timestamp"],
            unit=d.get("unit", ""),
            field_id=d.get("field_id"),
            confidence=d.get("confidence", 1.0),
            metadata=dict(d.get("metadata", {})),
        )


class SensorDataIngestion:
    """Ingest and manage sensor data readings.

    Stores readings by sensor ID and type, provides latest-value
    queries, and supports batch ingestion.
    """

    def __init__(self):
        self._readings: List[SensorReading] = []
        self._by_sensor: Dict[str, List[SensorReading]] = {}
        self._by_type: Dict[SensorType, List[SensorReading]] = {}

    def ingest(self, reading: SensorReading) -> None:
        """Ingest a single sensor reading."""
        self._readings.append(reading)
        self._by_sensor.setdefault(reading.sensor_id, []).append(reading)
        self._by_type.setdefault(reading.sensor_type, []).append(reading)

    def ingest_batch(self, readings: List[SensorReading]) -> None:
        """Ingest multiple readings."""
        for reading in readings:
            self.ingest(reading)

    def get_latest(self, sensor_type: SensorType) -> Optional[SensorReading]:
        """Get latest reading by sensor type."""
        readings = self._by_type.get(sensor_type, [])
        if not readings:
            return None
        return max(readings, key=lambda r: r.timestamp)

    def get_readings_by_sensor(self, sensor_id: str) -> List[SensorReading]:
        """Get all readings from a specific sensor."""
        return list(self._by_sensor.get(sensor_id, []))

    def get_readings_by_type(self, sensor_type: SensorType) -> List[SensorReading]:
        """Get all readings of a specific type."""
        return list(self._by_type.get(sensor_type, []))

    def reading_count(self) -> int:
        """Total number of readings."""
        return len(self._readings)

    def clear(self) -> None:
        """Clear all readings."""
        self._readings.clear()
        self._by_sensor.clear()
        self._by_type.clear()


class StateReconciler:
    """Reconcile sensor readings with digital twin state.

    Updates state dict from sensor readings, with optional
    confidence-weighted blending.
    """

    # Mapping from sensor type to state field
    _FIELD_MAP: Dict[SensorType, str] = {
        SensorType.SOIL_MOISTURE: "soil_moisture",
        SensorType.TEMPERATURE: "temperature",
        SensorType.NITROGEN: "nitrogen",
        SensorType.PHOSPHORUS: "phosphorus",
        SensorType.POTASSIUM: "potassium",
        SensorType.PH: "pH",
        SensorType.RAINFALL: "precipitation",
        SensorType.WIND_SPEED: "wind_speed",
        SensorType.SOLAR_RADIATION: "solar_radiation",
    }

    def reconcile(self, state: Dict[str, Any], reading: SensorReading) -> Dict[str, Any]:
        """Reconcile state with a sensor reading.

        Uses confidence-weighted blending when confidence < 1.0.
        """
        field = self._FIELD_MAP.get(reading.sensor_type)
        if field is None:
            return state

        if reading.confidence >= 1.0:
            state[field] = reading.value
        else:
            # Weighted blend
            old_value = state.get(field, reading.value)
            state[field] = old_value * (1 - reading.confidence) + reading.value * reading.confidence

        return state

    def reconcile_batch(
        self, state: Dict[str, Any], readings: List[SensorReading]
    ) -> Dict[str, Any]:
        """Reconcile state with multiple readings."""
        for reading in readings:
            state = self.reconcile(state, reading)
        return state

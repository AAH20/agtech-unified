"""IoT module: MQTT/Kafka/TimescaleDB pipeline, sensor placement, context broker,
and smart models."""

from src.iot.context_broker import NGSILDBroker
from src.iot.data_pipeline import (
    KafkaStream,
    MQTTClient,
    SensorReading,
    TimescaleDBStorage,
    TimeSeriesQuery,
)
from src.iot.data_validation import RangeChecker, Schema, SchemaValidator, ValidationResult
from src.iot.device_management import Device, DeviceRegistry, DeviceStatus, HealthStatus
from src.iot.sensor_placement import PlacementInstance, PlacementResult, SensorPlacement
from src.iot.smart_models import AgriculturalSmartModels

__all__ = [
    "SensorReading",
    "TimeSeriesQuery",
    "MQTTClient",
    "KafkaStream",
    "TimescaleDBStorage",
    "PlacementInstance",
    "PlacementResult",
    "SensorPlacement",
    "NGSILDBroker",
    "ValidationResult",
    "Schema",
    "SchemaValidator",
    "RangeChecker",
    "DeviceStatus",
    "HealthStatus",
    "Device",
    "DeviceRegistry",
    "AgriculturalSmartModels",
]

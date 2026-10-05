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
from src.iot.data_validation import (
    RangeChecker,
    RateOfChangeValidator,
    Schema,
    SchemaValidator,
    ValidationResult,
)
from src.iot.device_management import Device, DeviceRegistry, DeviceStatus, HealthStatus
from src.iot.edge_computing import EdgeNode, EdgeScheduler, EdgeTask, TaskAssignment
from src.iot.mqtt_qos import DeliveryState, MQTTMessage, MQTTQoSClient, QoSLevel
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
    "RateOfChangeValidator",
    "DeviceStatus",
    "HealthStatus",
    "Device",
    "DeviceRegistry",
    "AgriculturalSmartModels",
    "EdgeNode",
    "EdgeScheduler",
    "EdgeTask",
    "TaskAssignment",
    "DeliveryState",
    "MQTTMessage",
    "MQTTQoSClient",
    "QoSLevel",
]

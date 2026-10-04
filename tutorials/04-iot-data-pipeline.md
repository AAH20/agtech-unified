# Tutorial 04: IoT Data Pipeline

## Overview

Learn how to ingest, validate, and store IoT sensor data.

## Sensor Data Ingestion

```python
from src.iot.data_pipeline import SensorReading, MQTTClient

# Create sensor reading
reading = SensorReading(
    sensor_id="soil-01",
    timestamp="2026-10-04T12:00:00Z",
    temperature=25.3,
    soil_moisture=0.65,
    ph=6.8,
)

# Publish via MQTT
client = MQTTClient(broker_url="localhost")
client.publish("sensors/soil-01", reading.to_json())
```

## Data Validation

```python
from src.iot.data_validation import SchemaValidator, RangeChecker

# Define schema
schema = {
    "temperature": {"type": "float", "min": -40, "max": 80},
    "soil_moisture": {"type": "float", "min": 0, "max": 1},
    "ph": {"type": "float", "min": 0, "max": 14},
}

validator = SchemaValidator(schema)
result = validator.validate({"temperature": 25.3, "soil_moisture": 0.65, "ph": 6.8})
print(f"Valid: {result.is_valid}, Errors: {result.errors}")
```

## Sensor Placement Optimization

```python
from src.iot.sensor_placement import PlacementInstance, SensorPlacement

sensors = [(0, 0), (5, 5), (10, 0), (5, 10)]
targets = [(1, 1), (4, 4), (9, 1), (5, 0), (2, 8)]
radius = 3.0

result = SensorPlacement("greedy").optimize(PlacementInstance(sensors, targets, radius))
print(f"Selected: {result.selected_sensors}, Coverage: {result.coverage_ratio:.0%}")
```

## Next Steps

- [Tutorial 05: Digital Twin Simulation](05-digital-twin-simulation.md)
- [Tutorial 06: Decision Support](06-decision-support.md)

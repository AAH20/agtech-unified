"""IoT pipeline example: ingest, validate, and store sensor data."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.iot.data_pipeline import SensorReading, MQTTClient
from src.iot.data_validation import SchemaValidator


def main():
    reading = SensorReading(
        sensor_id="soil-01",
        timestamp="2026-10-04T12:00:00Z",
        temperature=25.3,
        soil_moisture=0.65,
        ph=6.8,
    )
    print(f"Sensor: {reading.sensor_id}")
    print(f"Temperature: {reading.temperature}C")
    print(f"Soil Moisture: {reading.soil_moisture}")

    schema = {
        "temperature": {"type": "float", "min": -40, "max": 80},
        "soil_moisture": {"type": "float", "min": 0, "max": 1},
    }
    validator = SchemaValidator(schema)
    result = validator.validate({"temperature": 25.3, "soil_moisture": 0.65})
    print(f"Valid: {result.is_valid}")


if __name__ == "__main__":
    main()

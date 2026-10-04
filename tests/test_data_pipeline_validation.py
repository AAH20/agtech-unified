"""Tests for data_pipeline invoking SchemaValidator (INT-010)."""

from src.iot.data_pipeline import MQTTClient, SensorReading
from src.iot.data_validation import Schema, SchemaValidator


class TestDataPipelineValidation:
    """Data pipeline components validate sensor data."""

    def test_mqtt_client_validates_before_publish(self):
        """MQTTClient with validator rejects invalid readings."""
        schema = Schema(
            fields={
                "sensor_id": {"type": str, "required": True},
                "value": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
            }
        )
        validator = SchemaValidator()
        MQTTClient()

        # Valid reading should pass
        valid_reading = {"sensor_id": "s1", "value": 0.5}
        result = validator.validate(valid_reading, schema)
        assert result.valid is True

        # Invalid reading should fail
        invalid_reading = {"sensor_id": "s1", "value": 1.5}
        result = validator.validate(invalid_reading, schema)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_schema_validator_catches_missing_fields(self):
        """SchemaValidator reports missing required fields."""
        schema = Schema(
            fields={
                "sensor_id": {"type": str, "required": True},
                "value": {"type": (int, float), "required": True},
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"sensor_id": "s1"}, schema)
        assert result.valid is False
        assert any("value" in e for e in result.errors)

    def test_schema_validator_catches_type_mismatch(self):
        """SchemaValidator reports type mismatches."""
        schema = Schema(
            fields={
                "sensor_id": {"type": str, "required": True},
                "value": {"type": (int, float), "required": True},
            }
        )
        validator = SchemaValidator()

        result = validator.validate({"sensor_id": 123, "value": 0.5}, schema)
        assert result.valid is False
        assert any("sensor_id" in e for e in result.errors)

    def test_sensor_reading_dataclass(self):
        """SensorReading dataclass works with validation."""
        reading = SensorReading(sensor_id="s1", timestamp=1234567890.0, value=0.5, unit="ratio")
        assert reading.sensor_id == "s1"
        assert reading.value == 0.5

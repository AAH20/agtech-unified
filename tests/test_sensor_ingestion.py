"""Tests for sensor data ingestion module."""

from src.digital_twin.sensor_ingestion import (
    SensorDataIngestion,
    SensorReading,
    SensorType,
    StateReconciler,
)


class TestSensorReading:
    """Test sensor reading data structure."""

    def test_create_reading(self):
        """Create a sensor reading."""
        reading = SensorReading(
            sensor_id="soil_moisture_01",
            sensor_type=SensorType.SOIL_MOISTURE,
            value=0.35,
            timestamp=1696156800.0,
            unit="m³/m³",
            field_id="field_01",
        )
        assert reading.sensor_id == "soil_moisture_01"
        assert reading.value == 0.35
        assert reading.unit == "m³/m³"

    def test_to_dict(self):
        """Serialize reading."""
        reading = SensorReading(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            value=25.0,
            timestamp=1696156800.0,
            unit="°C",
        )
        d = reading.to_dict()
        assert d["sensor_id"] == "temp_01"
        assert d["value"] == 25.0

    def test_from_dict(self):
        """Deserialize reading."""
        d = {
            "sensor_id": "temp_01",
            "sensor_type": "temperature",
            "value": 25.0,
            "timestamp": 1696156800.0,
            "unit": "°C",
        }
        reading = SensorReading.from_dict(d)
        assert reading.sensor_id == "temp_01"
        assert reading.value == 25.0


class TestSensorDataIngestion:
    """Test sensor data ingestion."""

    def test_ingest_reading(self):
        """Ingest a sensor reading."""
        ingestion = SensorDataIngestion()
        reading = SensorReading(
            sensor_id="soil_01",
            sensor_type=SensorType.SOIL_MOISTURE,
            value=0.35,
            timestamp=1696156800.0,
        )
        ingestion.ingest(reading)
        assert ingestion.reading_count() == 1

    def test_ingest_multiple(self):
        """Ingest multiple readings."""
        ingestion = SensorDataIngestion()
        for i in range(5):
            ingestion.ingest(
                SensorReading(
                    sensor_id=f"sensor_{i}",
                    sensor_type=SensorType.SOIL_MOISTURE,
                    value=0.3 + i * 0.01,
                    timestamp=1696156800.0 + i * 3600,
                )
            )
        assert ingestion.reading_count() == 5

    def test_get_latest_by_type(self):
        """Get latest reading by sensor type."""
        ingestion = SensorDataIngestion()
        ingestion.ingest(
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.30,
                timestamp=1696156800.0,
            )
        )
        ingestion.ingest(
            SensorReading(
                sensor_id="s2",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.35,
                timestamp=1696160400.0,
            )
        )
        latest = ingestion.get_latest(SensorType.SOIL_MOISTURE)
        assert latest.value == 0.35

    def test_get_latest_empty(self):
        """No readings returns None."""
        ingestion = SensorDataIngestion()
        assert ingestion.get_latest(SensorType.SOIL_MOISTURE) is None

    def test_get_readings_by_sensor(self):
        """Get all readings from a specific sensor."""
        ingestion = SensorDataIngestion()
        ingestion.ingest(
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.30,
                timestamp=1696156800.0,
            )
        )
        ingestion.ingest(
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.32,
                timestamp=1696160400.0,
            )
        )
        ingestion.ingest(
            SensorReading(
                sensor_id="s2",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.35,
                timestamp=1696160400.0,
            )
        )
        s1_readings = ingestion.get_readings_by_sensor("s1")
        assert len(s1_readings) == 2

    def test_clear(self):
        """Clear all readings."""
        ingestion = SensorDataIngestion()
        ingestion.ingest(
            SensorReading(
                sensor_id="s1",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.30,
                timestamp=1696156800.0,
            )
        )
        ingestion.clear()
        assert ingestion.reading_count() == 0

    def test_ingest_batch(self):
        """Ingest batch of readings."""
        ingestion = SensorDataIngestion()
        readings = [
            SensorReading(
                sensor_id=f"s{i}",
                sensor_type=SensorType.SOIL_MOISTURE,
                value=0.3 + i * 0.01,
                timestamp=1696156800.0 + i * 60,
            )
            for i in range(10)
        ]
        ingestion.ingest_batch(readings)
        assert ingestion.reading_count() == 10


class TestStateReconciler:
    """Test state reconciliation with sensor data."""

    def test_reconcile_moisture(self):
        """Reconcile soil moisture from sensor."""
        reconciler = StateReconciler()
        reading = SensorReading(
            sensor_id="soil_01",
            sensor_type=SensorType.SOIL_MOISTURE,
            value=0.35,
            timestamp=1696156800.0,
        )
        state = {"soil_moisture": 0.25}
        reconciler.reconcile(state, reading)
        assert state["soil_moisture"] == 0.35

    def test_reconcile_temperature(self):
        """Reconcile temperature from sensor."""
        reconciler = StateReconciler()
        reading = SensorReading(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            value=28.0,
            timestamp=1696156800.0,
        )
        state = {"temperature": 22.0}
        reconciler.reconcile(state, reading)
        assert state["temperature"] == 28.0

    def test_reconcile_nitrogen(self):
        """Reconcile nitrogen from sensor."""
        reconciler = StateReconciler()
        reading = SensorReading(
            sensor_id="n_01",
            sensor_type=SensorType.NITROGEN,
            value=45.0,
            timestamp=1696156800.0,
        )
        state = {"nitrogen": 30.0}
        reconciler.reconcile(state, reading)
        assert state["nitrogen"] == 45.0

    def test_reconcile_unknown_type(self):
        """Unknown sensor type is ignored."""
        reconciler = StateReconciler()
        reading = SensorReading(
            sensor_id="unknown_01",
            sensor_type=SensorType.UNKNOWN,
            value=999.0,
            timestamp=1696156800.0,
        )
        state = {"soil_moisture": 0.3}
        reconciler.reconcile(state, reading)
        assert state["soil_moisture"] == 0.3

    def test_reconcile_with_confidence(self):
        """Reconcile with confidence weighting."""
        reconciler = StateReconciler()
        reading = SensorReading(
            sensor_id="soil_01",
            sensor_type=SensorType.SOIL_MOISTURE,
            value=0.40,
            timestamp=1696156800.0,
            confidence=0.8,
        )
        state = {"soil_moisture": 0.30}
        reconciler.reconcile(state, reading)
        # Weighted average: 0.3 * 0.2 + 0.4 * 0.8 = 0.38
        assert abs(state["soil_moisture"] - 0.38) < 0.01

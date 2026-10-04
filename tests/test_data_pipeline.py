"""Test agricultural IoT data pipeline: MQTT, Kafka, TimescaleDB."""
import pytest
import time
import json
from src.iot.data_pipeline import (
    MQTTClient,
    KafkaStream,
    TimescaleDBStorage,
    SensorReading,
    TimeSeriesQuery,
)


# ===========================================================================
# MQTT Client Tests
# ===========================================================================


class TestMQTTClient:
    """Tests for MQTTClient pub/sub functionality."""

    def test_mqtt_connect_disconnect(self):
        """MQTT client connects and disconnects."""
        client = MQTTClient(host="localhost", port=1883)
        assert client.connect() is True
        assert client.is_connected is True
        assert client.disconnect() is True
        assert client.is_connected is False

    def test_mqtt_publish_subscribe(self):
        """MQTT publish delivers message to subscriber."""
        client = MQTTClient()
        client.connect()

        received = []
        client.subscribe("sensors/temperature", lambda t, p: received.append((t, p)))

        client.publish("sensors/temperature", {"temp": 25.5})
        assert len(received) == 1
        assert received[0][0] == "sensors/temperature"
        assert json.loads(received[0][1].decode()) == {"temp": 25.5}

    def test_mqtt_multiple_subscribers(self):
        """Multiple subscribers on same topic all receive messages."""
        client = MQTTClient()
        client.connect()

        received_a = []
        received_b = []
        client.subscribe("sensors/humidity", lambda t, p: received_a.append(p))
        client.subscribe("sensors/humidity", lambda t, p: received_b.append(p))

        client.publish("sensors/humidity", "60%")
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_mqtt_unsubscribe(self):
        """Unsubscribed callback no longer receives messages."""
        client = MQTTClient()
        client.connect()

        received = []
        cb = lambda t, p: received.append(p)
        client.subscribe("sensors/soil", cb)
        client.publish("sensors/soil", "data1")
        assert len(received) == 1

        client.unsubscribe("sensors/soil", cb)
        client.publish("sensors/soil", "data2")
        assert len(received) == 1  # No new message

    def test_mqtt_topic_isolation(self):
        """Messages on one topic don't reach subscribers of another."""
        client = MQTTClient()
        client.connect()

        received = []
        client.subscribe("sensors/temperature", lambda t, p: received.append(p))

        client.publish("sensors/humidity", "60%")
        assert len(received) == 0


# ===========================================================================
# Kafka Stream Tests
# ===========================================================================


class TestKafkaStream:
    """Tests for KafkaStream producer/consumer functionality."""

    def test_kafka_start_stop(self):
        """Kafka stream starts and stops."""
        stream = KafkaStream()
        assert stream.start() is True
        assert stream.is_running is True
        assert stream.stop() is True
        assert stream.is_running is False

    def test_kafka_produce_consume(self):
        """Produced messages can be consumed."""
        stream = KafkaStream()
        stream.start()

        stream.produce("sensor-data", {"sensor": "temp-1", "value": 22.5})
        messages = stream.consume("sensor-data", timeout=0.5)
        assert len(messages) == 1
        assert json.loads(messages[0]["value"].decode()) == {"sensor": "temp-1", "value": 22.5}

    def test_kafka_topic_creation(self):
        """Topics are created and listed."""
        stream = KafkaStream()
        stream.create_topic("agri-events", partitions=3)
        assert "agri-events" in stream.list_topics()
        assert stream.get_partition_count("agri-events") == 3

    def test_kafka_multiple_messages_ordering(self):
        """Messages are consumed in FIFO order."""
        stream = KafkaStream()
        stream.start()

        for i in range(5):
            stream.produce("events", {"seq": i})

        messages = stream.consume("events", timeout=0.5, max_messages=5)
        assert len(messages) == 5
        for i, msg in enumerate(messages):
            assert json.loads(msg["value"].decode())["seq"] == i

    def test_kafka_consume_empty_topic(self):
        """Consuming from empty topic returns empty list."""
        stream = KafkaStream()
        stream.create_topic("empty-topic")
        messages = stream.consume("empty-topic", timeout=0.1)
        assert messages == []

    def test_kafka_delete_topic(self):
        """Topic deletion removes topic."""
        stream = KafkaStream()
        stream.create_topic("temp-topic")
        assert "temp-topic" in stream.list_topics()
        assert stream.delete_topic("temp-topic") is True
        assert "temp-topic" not in stream.list_topics()
        assert stream.delete_topic("temp-topic") is False


# ===========================================================================
# TimescaleDB Storage Tests
# ===========================================================================


class TestTimescaleDBStorage:
    """Tests for TimescaleDBStorage time-series functionality."""

    def test_timescale_connect_disconnect(self):
        """Storage connects and disconnects."""
        storage = TimescaleDBStorage()
        assert storage.connect() is True
        assert storage.is_connected is True
        assert storage.disconnect() is True
        assert storage.is_connected is False

    def test_timescale_insert_and_query(self):
        """Insert and query sensor readings."""
        storage = TimescaleDBStorage()
        storage.connect()

        reading = SensorReading(
            sensor_id="temp-1",
            timestamp=time.time(),
            value=25.5,
            unit="celsius",
        )
        assert storage.insert_reading(reading) is True

        results = storage.query("sensor_readings", sensor_id="temp-1")
        assert len(results) == 1
        assert results[0]["value"] == 25.5
        assert results[0]["unit"] == "celsius"

    def test_timescale_time_range_query(self):
        """Query filters by time range."""
        storage = TimescaleDBStorage()
        storage.connect()

        now = time.time()
        storage.insert("sensor_readings", {
            "time": now - 3600, "sensor_id": "s1", "value": 10.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": now - 1800, "sensor_id": "s1", "value": 20.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": now, "sensor_id": "s1", "value": 30.0, "unit": "c"
        })

        results = storage.query(
            "sensor_readings", sensor_id="s1", start_time=now - 2000
        )
        assert len(results) == 2
        assert results[0]["value"] == 20.0
        assert results[1]["value"] == 30.0

    def test_timescale_aggregation_avg(self):
        """Average aggregation computes correctly."""
        storage = TimescaleDBStorage()
        storage.connect()

        for val in [10.0, 20.0, 30.0]:
            storage.insert("sensor_readings", {
                "time": time.time(), "sensor_id": "s1", "value": val, "unit": "c"
            })

        avg = storage.aggregate("sensor_readings", "s1", "avg")
        assert avg == 20.0

    def test_timescale_aggregation_min_max(self):
        """Min and max aggregations work correctly."""
        storage = TimescaleDBStorage()
        storage.connect()

        for val in [15.0, 5.0, 25.0]:
            storage.insert("sensor_readings", {
                "time": time.time(), "sensor_id": "s1", "value": val, "unit": "c"
            })

        assert storage.aggregate("sensor_readings", "s1", "min") == 5.0
        assert storage.aggregate("sensor_readings", "s1", "max") == 25.0

    def test_timescale_hypertable_creation(self):
        """Hypertable creation and detection."""
        storage = TimescaleDBStorage()
        storage.connect()

        assert storage.is_hypertable("sensor_data") is False
        storage.create_hypertable("sensor_data")
        assert storage.is_hypertable("sensor_data") is True

    def test_timescale_batch_insert(self):
        """Batch insert adds multiple records."""
        storage = TimescaleDBStorage()
        storage.connect()

        records = [
            {"time": time.time() + i, "sensor_id": "s1", "value": float(i), "unit": "c"}
            for i in range(10)
        ]
        count = storage.insert_batch("sensor_readings", records)
        assert count == 10
        assert storage.get_table_size("sensor_readings") == 10

    def test_timescale_query_latest(self):
        """Query latest returns most recent record."""
        storage = TimescaleDBStorage()
        storage.connect()

        now = time.time()
        storage.insert("sensor_readings", {
            "time": now - 100, "sensor_id": "s1", "value": 10.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": now, "sensor_id": "s1", "value": 20.0, "unit": "c"
        })

        latest = storage.query_latest("sensor_readings", "s1")
        assert latest is not None
        assert latest["value"] == 20.0

    def test_timescale_delete_old_data(self):
        """Delete old data removes records before cutoff."""
        storage = TimescaleDBStorage()
        storage.connect()

        now = time.time()
        storage.insert("sensor_readings", {
            "time": now - 7200, "sensor_id": "s1", "value": 1.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": now - 3600, "sensor_id": "s1", "value": 2.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": now, "sensor_id": "s1", "value": 3.0, "unit": "c"
        })

        deleted = storage.delete_old_data("sensor_readings", now - 5000)
        assert deleted == 1
        assert storage.get_table_size("sensor_readings") == 2

    def test_timescale_get_sensor_ids(self):
        """Get sensor IDs returns unique IDs."""
        storage = TimescaleDBStorage()
        storage.connect()

        storage.insert("sensor_readings", {
            "time": time.time(), "sensor_id": "s1", "value": 1.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": time.time(), "sensor_id": "s2", "value": 2.0, "unit": "c"
        })
        storage.insert("sensor_readings", {
            "time": time.time(), "sensor_id": "s1", "value": 3.0, "unit": "c"
        })

        ids = storage.get_sensor_ids("sensor_readings")
        assert set(ids) == {"s1", "s2"}

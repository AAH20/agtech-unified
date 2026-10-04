"""Test agricultural IoT streaming pipeline integration."""

import json
import time

from src.iot.data_pipeline import (
    KafkaStream,
    MQTTClient,
    TimescaleDBStorage,
)


class TestMQTTStreaming:
    """Tests for MQTT streaming integration patterns."""

    def test_mqtt_sensor_data_stream(self):
        """Stream sensor data through MQTT pub/sub."""
        client = MQTTClient()
        client.connect()

        readings = []
        client.subscribe("farm/sensors/#", lambda t, p: readings.append((t, p)))

        for i in range(5):
            client.publish("farm/sensors/temperature", {"value": 20.0 + i, "unit": "c"})

        assert len(readings) == 5
        for i, (topic, payload) in enumerate(readings):
            data = json.loads(payload.decode())
            assert data["value"] == 20.0 + i

    def test_mqtt_qos_levels(self):
        """Publish with different QoS levels."""
        client = MQTTClient()
        client.connect()

        received = []
        client.subscribe("test/qos", lambda t, p: received.append(p))

        assert client.publish("test/qos", "msg1", qos=0) is True
        assert client.publish("test/qos", "msg2", qos=1) is True
        assert client.publish("test/qos", "msg3", qos=2) is True

        assert len(received) == 3

    def test_mqtt_binary_payload(self):
        """Publish binary payload."""
        client = MQTTClient()
        client.connect()

        received = []
        client.subscribe("binary/topic", lambda t, p: received.append(p))

        client.publish("binary/topic", b"\x00\x01\x02\x03")
        assert len(received) == 1
        assert received[0] == b"\x00\x01\x02\x03"

    def test_mqtt_clear_subscribers(self):
        """Clear all subscribers."""
        client = MQTTClient()
        client.connect()

        received = []
        client.subscribe("test/topic", lambda t, p: received.append(p))
        client.clear_subscribers()

        client.publish("test/topic", "data")
        assert len(received) == 0


class TestKafkaStreaming:
    """Tests for Kafka streaming integration patterns."""

    def test_kafka_sensor_event_stream(self):
        """Stream sensor events through Kafka."""
        stream = KafkaStream()
        stream.start()

        for i in range(10):
            stream.produce(
                "sensor-events",
                {
                    "sensor_id": f"sensor-{i % 3}",
                    "value": float(i),
                    "timestamp": time.time(),
                },
            )

        assert stream.get_topic_size("sensor-events") == 10

        messages = stream.consume("sensor-events", timeout=0.5, max_messages=10)
        assert len(messages) == 10

    def test_kafka_topic_with_key(self):
        """Produce messages with keys."""
        stream = KafkaStream()
        stream.start()

        stream.produce("keyed-topic", {"data": "value1"}, key="sensor-1")
        stream.produce("keyed-topic", {"data": "value2"}, key="sensor-2")

        messages = stream.consume("keyed-topic", timeout=0.5, max_messages=2)
        assert messages[0]["key"] == "sensor-1"
        assert messages[1]["key"] == "sensor-2"

    def test_kafka_consume_all(self):
        """Consume all messages from topic."""
        stream = KafkaStream()
        stream.start()

        for i in range(5):
            stream.produce("bulk-topic", {"seq": i})

        messages = stream.consume_all("bulk-topic")
        assert len(messages) == 5
        assert stream.get_topic_size("bulk-topic") == 0

    def test_kafka_multiple_topics(self):
        """Multiple topics operate independently."""
        stream = KafkaStream()
        stream.start()

        stream.produce("topic-a", {"source": "a"})
        stream.produce("topic-b", {"source": "b"})

        assert stream.get_topic_size("topic-a") == 1
        assert stream.get_topic_size("topic-b") == 1

        msg_a = stream.consume("topic-a", timeout=0.5)
        assert json.loads(msg_a[0]["value"].decode())["source"] == "a"


class TestEndToEndPipeline:
    """End-to-end integration tests for the full data pipeline."""

    def test_mqtt_to_kafka_bridge(self):
        """Bridge MQTT messages to Kafka stream."""
        mqtt = MQTTClient()
        mqtt.connect()
        kafka = KafkaStream()
        kafka.start()

        # Bridge: MQTT subscriber produces to Kafka
        def bridge(topic, payload):
            kafka.produce("bridged-events", json.loads(payload.decode()))

        mqtt.subscribe("sensors/#", bridge)
        mqtt.publish("sensors/temperature", {"value": 25.0, "unit": "c"})

        messages = kafka.consume("bridged-events", timeout=0.5)
        assert len(messages) == 1
        assert json.loads(messages[0]["value"].decode())["value"] == 25.0

    def test_kafka_to_timescale_bridge(self):
        """Bridge Kafka messages to TimescaleDB storage."""
        kafka = KafkaStream()
        kafka.start()
        storage = TimescaleDBStorage()
        storage.connect()
        storage.create_hypertable("sensor_readings")

        # Produce sensor readings to Kafka
        for i in range(5):
            kafka.produce(
                "sensor-readings",
                {
                    "sensor_id": "temp-1",
                    "value": 20.0 + i,
                    "unit": "celsius",
                    "timestamp": time.time() + i,
                },
            )

        # Consume and store
        messages = kafka.consume("sensor-readings", timeout=0.5, max_messages=5)
        for msg in messages:
            data = json.loads(msg["value"].decode())
            storage.insert("sensor_readings", data)

        results = storage.query("sensor_readings", sensor_id="temp-1")
        assert len(results) == 5

    def test_full_pipeline_sensor_data(self):
        """Full pipeline: MQTT -> Kafka -> TimescaleDB."""
        mqtt = MQTTClient()
        mqtt.connect()
        kafka = KafkaStream()
        kafka.start()
        storage = TimescaleDBStorage()
        storage.connect()
        storage.create_hypertable("sensor_readings")

        # Stage 1: MQTT subscriber bridges to Kafka
        def mqtt_to_kafka(topic, payload):
            kafka.produce("pipeline", json.loads(payload.decode()))

        mqtt.subscribe("farm/sensors/#", mqtt_to_kafka)

        # Stage 2: Publish sensor data via MQTT
        for i in range(3):
            mqtt.publish(
                "farm/sensors/soil",
                {
                    "sensor_id": "soil-1",
                    "value": 30.0 + i,
                    "unit": "percent",
                    "timestamp": time.time() + i,
                },
            )

        # Stage 3: Consume from Kafka and store in TimescaleDB
        messages = kafka.consume("pipeline", timeout=0.5, max_messages=3)
        for msg in messages:
            data = json.loads(msg["value"].decode())
            storage.insert("sensor_readings", data)

        # Verify data in storage
        results = storage.query("sensor_readings", sensor_id="soil-1")
        assert len(results) == 3
        assert results[0]["value"] == 30.0
        assert results[2]["value"] == 32.0

    def test_pipeline_with_aggregation(self):
        """Full pipeline with aggregation query."""
        mqtt = MQTTClient()
        mqtt.connect()
        kafka = KafkaStream()
        kafka.start()
        storage = TimescaleDBStorage()
        storage.connect()

        def mqtt_to_kafka(topic, payload):
            kafka.produce("agg-pipeline", json.loads(payload.decode()))

        mqtt.subscribe("sensors/#", mqtt_to_kafka)

        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for val in values:
            mqtt.publish(
                "sensors/temp",
                {
                    "sensor_id": "temp-agg",
                    "value": val,
                    "unit": "c",
                    "timestamp": time.time(),
                },
            )

        messages = kafka.consume("agg-pipeline", timeout=0.5, max_messages=5)
        for msg in messages:
            data = json.loads(msg["value"].decode())
            storage.insert("sensor_readings", data)

        avg = storage.aggregate("sensor_readings", "temp-agg", "avg")
        assert avg == 30.0
        assert storage.aggregate("sensor_readings", "temp-agg", "min") == 10.0
        assert storage.aggregate("sensor_readings", "temp-agg", "max") == 50.0

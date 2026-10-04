"""Agricultural IoT data pipeline: MQTT, Kafka, and TimescaleDB integration.

Provides MQTTClient for pub/sub messaging, KafkaStream for event streaming,
and TimescaleDBStorage for time-series data persistence. All components work
in-memory for testing and can be extended with real backends.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SensorReading:
    """Single sensor reading from an agricultural IoT device."""

    sensor_id: str
    timestamp: float
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesQuery:
    """Time-series query parameters."""

    sensor_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    limit: Optional[int] = None
    aggregation: Optional[str] = None  # 'avg', 'min', 'max', 'sum', 'count'


# ---------------------------------------------------------------------------
# MQTT Client
# ---------------------------------------------------------------------------


class MQTTClient:
    """MQTT client for IoT sensor data pub/sub.

    Supports topic-based publish/subscribe with callback dispatch.
    Works in-memory; can be extended with paho-mqtt for real brokers.
    """

    def __init__(self, host: str = "localhost", port: int = 1883, client_id: str = ""):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._connected = False
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Disconnect from MQTT broker."""
        self._connected = False
        return True

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def _topic_matches(self, subscription: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern.

        Supports MQTT wildcards: '+' matches one level, '#' matches all remaining.
        """
        if subscription == topic:
            return True
        if subscription.endswith("/#"):
            prefix = subscription[:-2]
            return topic == prefix or topic.startswith(prefix + "/")
        sub_parts = subscription.split("/")
        topic_parts = topic.split("/")
        for i, part in enumerate(sub_parts):
            if part == "+":
                if i >= len(topic_parts):
                    return False
            elif part == "#":
                return True
            else:
                if i >= len(topic_parts) or topic_parts[i] != part:
                    return False
        return len(sub_parts) == len(topic_parts)

    def publish(self, topic: str, payload: Union[str, bytes, dict], qos: int = 0) -> bool:
        """Publish message to a topic.

        Args:
            topic: MQTT topic string
            payload: Message payload (str, bytes, or dict)
            qos: Quality of service level (0, 1, or 2)

        Returns:
            True if published successfully
        """
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        with self._lock:
            for sub_topic, callbacks in self._subscribers.items():
                if self._topic_matches(sub_topic, topic):
                    for callback in callbacks:
                        try:
                            callback(topic, payload)
                        except Exception as e:
                            logger.error(f"Error in MQTT callback for topic {topic}: {e}")
        return True

    def subscribe(self, topic: str, callback: Callable) -> bool:
        """Subscribe to a topic with a callback function.

        Args:
            topic: MQTT topic string
            callback: Function called with (topic, payload) on message

        Returns:
            True if subscribed successfully
        """
        with self._lock:
            self._subscribers[topic].append(callback)
        return True

    def unsubscribe(self, topic: str, callback: Optional[Callable] = None) -> bool:
        """Unsubscribe from a topic.

        Args:
            topic: MQTT topic string
            callback: Specific callback to remove, or None to remove all

        Returns:
            True if unsubscribed successfully
        """
        with self._lock:
            if callback is None:
                self._subscribers.pop(topic, None)
            else:
                self._subscribers[topic] = [
                    cb for cb in self._subscribers.get(topic, []) if cb != callback
                ]
        return True

    def on_message(self, topic: str, payload: bytes) -> None:
        """Handle incoming message (internal handler)."""
        self._message_queue.put((topic, payload))

    def get_message(self, timeout: float = 1.0) -> Optional[Tuple[str, bytes]]:
        """Get a message from the internal queue.

        Args:
            timeout: Maximum time to wait for a message

        Returns:
            Tuple of (topic, payload) or None if timeout
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_subscribers(self) -> None:
        """Remove all subscribers."""
        with self._lock:
            self._subscribers.clear()


# ---------------------------------------------------------------------------
# Kafka Stream
# ---------------------------------------------------------------------------


class KafkaStream:
    """Kafka producer/consumer for event streaming.

    Supports topic-based message production and consumption with
    partitioning. Works in-memory; can be extended with kafka-python.
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._topics: Dict[str, queue.Queue] = defaultdict(queue.Queue)
        self._topic_partitions: Dict[str, int] = {}
        self._consumer_offsets: Dict[str, int] = defaultdict(int)
        self._running = False

    def start(self) -> bool:
        """Start the Kafka stream processor."""
        self._running = True
        return True

    def stop(self) -> bool:
        """Stop the Kafka stream processor."""
        self._running = False
        return True

    @property
    def is_running(self) -> bool:
        """Check if stream processor is running."""
        return self._running

    def create_topic(self, topic: str, partitions: int = 1) -> bool:
        """Create a Kafka topic.

        Args:
            topic: Topic name
            partitions: Number of partitions

        Returns:
            True if created or already exists
        """
        if topic not in self._topics:
            self._topics[topic] = queue.Queue()
            self._topic_partitions[topic] = partitions
        return True

    def list_topics(self) -> List[str]:
        """List all topics."""
        return list(self._topics.keys())

    def produce(
        self,
        topic: str,
        message: Union[str, bytes, dict],
        key: Optional[str] = None,
        partition: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Produce a message to a topic.

        Args:
            topic: Topic name
            message: Message payload
            key: Optional message key
            partition: Optional partition number

        Returns:
            Metadata about the produced message
        """
        if topic not in self._topics:
            self.create_topic(topic)

        if isinstance(message, dict):
            message = json.dumps(message)
        if isinstance(message, str):
            message = message.encode("utf-8")

        offset = self._topics[topic].qsize()
        record = {
            "key": key,
            "value": message,
            "timestamp": time.time(),
            "partition": partition or 0,
            "offset": offset,
        }
        self._topics[topic].put(record)
        return record

    def consume(
        self, topic: str, timeout: float = 1.0, max_messages: int = 1
    ) -> List[Dict[str, Any]]:
        """Consume messages from a topic.

        Args:
            topic: Topic name
            timeout: Maximum time to wait per message
            max_messages: Maximum number of messages to consume

        Returns:
            List of message records
        """
        messages = []
        for _ in range(max_messages):
            try:
                msg = self._topics[topic].get(timeout=timeout)
                messages.append(msg)
            except queue.Empty:
                break
        return messages

    def consume_all(self, topic: str) -> List[Dict[str, Any]]:
        """Consume all available messages from a topic.

        Args:
            topic: Topic name

        Returns:
            List of all available message records
        """
        messages = []
        while not self._topics[topic].empty():
            try:
                messages.append(self._topics[topic].get_nowait())
            except queue.Empty:
                break
        return messages

    def get_topic_size(self, topic: str) -> int:
        """Get the number of messages in a topic.

        Args:
            topic: Topic name

        Returns:
            Number of messages
        """
        return self._topics[topic].qsize()

    def get_partition_count(self, topic: str) -> int:
        """Get the number of partitions for a topic.

        Args:
            topic: Topic name

        Returns:
            Number of partitions
        """
        return self._topic_partitions.get(topic, 1)

    def delete_topic(self, topic: str) -> bool:
        """Delete a topic.

        Args:
            topic: Topic name

        Returns:
            True if deleted, False if topic didn't exist
        """
        if topic in self._topics:
            del self._topics[topic]
            self._topic_partitions.pop(topic, None)
            return True
        return False


# ---------------------------------------------------------------------------
# TimescaleDB Storage
# ---------------------------------------------------------------------------


class TimescaleDBStorage:
    """TimescaleDB storage for time-series data.

    Supports hypertable creation, data insertion, time-range queries,
    and aggregations. Works in-memory; can be extended with psycopg2.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "iot",
        user: str = "postgres",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._hypertables: set = set()
        self._connected = False

    def connect(self) -> bool:
        """Connect to TimescaleDB."""
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Disconnect from TimescaleDB."""
        self._connected = False
        return True

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected

    def create_hypertable(self, table_name: str, time_column: str = "time") -> bool:
        """Create a hypertable for time-series data.

        Args:
            table_name: Name of the table
            time_column: Name of the timestamp column

        Returns:
            True if created successfully
        """
        self._hypertables.add(table_name)
        return True

    def is_hypertable(self, table_name: str) -> bool:
        """Check if a table is a hypertable.

        Args:
            table_name: Name of the table

        Returns:
            True if table is a hypertable
        """
        return table_name in self._hypertables

    def insert(self, table_name: str, data: Dict[str, Any]) -> bool:
        """Insert data into a table.

        Args:
            table_name: Name of the table
            data: Dictionary of column names to values

        Returns:
            True if inserted successfully
        """
        record = dict(data)
        if "time" not in record:
            record["time"] = time.time()
        self._data[table_name].append(record)
        return True

    def insert_reading(self, reading: SensorReading) -> bool:
        """Insert a sensor reading.

        Args:
            reading: SensorReading dataclass instance

        Returns:
            True if inserted successfully
        """
        return self.insert(
            "sensor_readings",
            {
                "time": reading.timestamp,
                "sensor_id": reading.sensor_id,
                "value": reading.value,
                "unit": reading.unit,
                "metadata": reading.metadata,
            },
        )

    def insert_batch(self, table_name: str, records: List[Dict[str, Any]]) -> int:
        """Insert multiple records into a table.

        Args:
            table_name: Name of the table
            records: List of record dictionaries

        Returns:
            Number of records inserted
        """
        count = 0
        for record in records:
            if self.insert(table_name, record):
                count += 1
        return count

    def query(
        self,
        table_name: str,
        sensor_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query data from a table.

        Args:
            table_name: Name of the table
            sensor_id: Filter by sensor ID
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum number of results

        Returns:
            List of matching records
        """
        results = self._data.get(table_name, [])

        if sensor_id is not None:
            results = [r for r in results if r.get("sensor_id") == sensor_id]
        if start_time is not None:
            results = [r for r in results if r.get("time", 0) >= start_time]
        if end_time is not None:
            results = [r for r in results if r.get("time", 0) <= end_time]

        results = sorted(results, key=lambda r: r.get("time", 0))

        if limit is not None:
            results = results[:limit]

        return results

    def query_latest(self, table_name: str, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest reading for a sensor.

        Args:
            table_name: Name of the table
            sensor_id: Sensor ID

        Returns:
            Latest record or None if no data
        """
        results = self.query(table_name, sensor_id=sensor_id)
        return results[-1] if results else None

    def aggregate(
        self,
        table_name: str,
        sensor_id: str,
        aggregation: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[float]:
        """Aggregate data for a sensor.

        Args:
            table_name: Name of the table
            sensor_id: Sensor ID
            aggregation: Type of aggregation ('avg', 'min', 'max', 'sum', 'count')
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp

        Returns:
            Aggregated value or None if no data
        """
        results = self.query(
            table_name, sensor_id=sensor_id, start_time=start_time, end_time=end_time
        )

        if not results:
            return None

        values = [r["value"] for r in results if "value" in r]

        if not values:
            return None

        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "count":
            return float(len(values))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def get_sensor_ids(self, table_name: str) -> List[str]:
        """Get all sensor IDs in a table.

        Args:
            table_name: Name of the table

        Returns:
            List of unique sensor IDs
        """
        results = self._data.get(table_name, [])
        return list({r["sensor_id"] for r in results if r.get("sensor_id") is not None})

    def delete_old_data(self, table_name: str, before_timestamp: float) -> int:
        """Delete data older than a timestamp.

        Args:
            table_name: Name of the table
            before_timestamp: Delete records before this time

        Returns:
            Number of records deleted
        """
        if table_name not in self._data:
            return 0

        original_len = len(self._data[table_name])
        self._data[table_name] = [
            r for r in self._data[table_name] if r.get("time", 0) >= before_timestamp
        ]
        return original_len - len(self._data[table_name])

    def get_table_size(self, table_name: str) -> int:
        """Get the number of records in a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of records
        """
        return len(self._data.get(table_name, []))

    def create_continuous_aggregate(
        self, view_name: str, table_name: str, bucket_interval: str = "1 hour"
    ) -> bool:
        """Create a continuous aggregate view.

        Args:
            view_name: Name of the view
            table_name: Source table name
            bucket_interval: Time bucket interval

        Returns:
            True if created successfully
        """
        return True

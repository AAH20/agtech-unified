"""Tests for event bus persistence, delivery guarantees, and schema validation."""

import os
import tempfile
import time

from src.integration.event_bus import (
    DeliverySemantics,
    DomainEvent,
    EventBus,
    EventSchema,
    EventSchemaRegistry,
    FileEventStore,
    InMemoryEventStore,
)


class TestEventSchemaRegistry:
    """Event schema registration and validation."""

    def test_register_and_validate(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id", "value"},
            field_types={"sensor_id": str, "value": float},
        )
        registry.register(schema)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1", "value": 42.0},
        )
        assert registry.validate(event) is True

    def test_validate_missing_required_field(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id", "value"},
            field_types={"sensor_id": str, "value": float},
        )
        registry.register(schema)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1"},  # missing value
        )
        assert registry.validate(event) is False

    def test_validate_wrong_type(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id", "value"},
            field_types={"sensor_id": str, "value": float},
        )
        registry.register(schema)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1", "value": "not_a_float"},
        )
        assert registry.validate(event) is False

    def test_validate_unknown_event_type(self):
        registry = EventSchemaRegistry()
        event = DomainEvent(
            event_type="unknown.type",
            source="test",
            payload={},
        )
        # Unknown types pass validation (no schema registered)
        assert registry.validate(event) is True

    def test_schema_versioning(self):
        registry = EventSchemaRegistry()
        schema_v1 = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id"},
            field_types={"sensor_id": str},
        )
        schema_v2 = EventSchema(
            event_type="sensor.reading",
            version=2,
            required_fields={"sensor_id", "value"},
            field_types={"sensor_id": str, "value": float},
        )
        registry.register(schema_v1)
        registry.register(schema_v2)
        # v2 event should validate against v2 schema
        event_v2 = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1", "value": 42.0},
        )
        assert registry.validate(event_v2) is True

    def test_get_schema(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id"},
            field_types={"sensor_id": str},
        )
        registry.register(schema)
        retrieved = registry.get_schema("sensor.reading")
        assert retrieved is not None
        assert retrieved.version == 1


class TestInMemoryEventStore:
    """In-memory event store backend."""

    def test_append_and_get(self):
        store = InMemoryEventStore()
        event = DomainEvent(event_type="test", source="test", payload={})
        store.append(event)
        events = store.get_all()
        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_get_by_type(self):
        store = InMemoryEventStore()
        e1 = DomainEvent(event_type="a", source="test", payload={})
        e2 = DomainEvent(event_type="b", source="test", payload={})
        e3 = DomainEvent(event_type="a", source="test", payload={})
        store.append(e1)
        store.append(e2)
        store.append(e3)
        a_events = store.get_by_type("a")
        assert len(a_events) == 2

    def test_get_by_time_range(self):
        store = InMemoryEventStore()
        e1 = DomainEvent(event_type="a", source="test", payload={})
        time.sleep(0.01)
        e2 = DomainEvent(event_type="a", source="test", payload={})
        store.append(e1)
        store.append(e2)
        events = store.get_by_time_range(e1.timestamp, e2.timestamp)
        assert len(events) == 2

    def test_replay(self):
        store = InMemoryEventStore()
        for i in range(5):
            store.append(DomainEvent(event_type="test", source="test", payload={"i": i}))
        replayed = store.replay()
        assert len(replayed) == 5
        assert replayed[0].payload["i"] == 0
        assert replayed[4].payload["i"] == 4


class TestFileEventStore:
    """File-based event store for persistence."""

    def test_append_and_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(os.path.join(tmpdir, "events.jsonl"))
            event = DomainEvent(event_type="test", source="test", payload={"key": "value"})
            store.append(event)
            # Read back from file
            events = store.get_all()
            assert len(events) == 1
            assert events[0].payload["key"] == "value"

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.jsonl")
            store1 = FileEventStore(path)
            event = DomainEvent(event_type="test", source="test", payload={"key": "value"})
            store1.append(event)
            # New instance reads from same file
            store2 = FileEventStore(path)
            events = store2.get_all()
            assert len(events) == 1
            assert events[0].event_id == event.event_id

    def test_get_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(os.path.join(tmpdir, "events.jsonl"))
            store.append(DomainEvent(event_type="a", source="test", payload={}))
            store.append(DomainEvent(event_type="b", source="test", payload={}))
            store.append(DomainEvent(event_type="a", source="test", payload={}))
            a_events = store.get_by_type("a")
            assert len(a_events) == 2

    def test_replay_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(os.path.join(tmpdir, "events.jsonl"))
            for i in range(3):
                store.append(DomainEvent(event_type="test", source="test", payload={"i": i}))
            replayed = store.replay()
            assert len(replayed) == 3


class TestEventBusPersistence:
    """Event bus with persistence backend."""

    def test_publish_persists_to_store(self):
        store = InMemoryEventStore()
        bus = EventBus(event_store=store)
        event = DomainEvent(event_type="test", source="test", payload={})
        bus.publish(event)
        assert len(store.get_all()) == 1

    def test_publish_persists_to_file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileEventStore(os.path.join(tmpdir, "events.jsonl"))
            bus = EventBus(event_store=store)
            event = DomainEvent(event_type="test", source="test", payload={"key": "value"})
            bus.publish(event)
            events = store.get_all()
            assert len(events) == 1
            assert events[0].payload["key"] == "value"

    def test_replay_from_store(self):
        store = InMemoryEventStore()
        bus = EventBus(event_store=store)
        for i in range(3):
            bus.publish(DomainEvent(event_type="test", source="test", payload={"i": i}))
        replayed = store.replay()
        assert len(replayed) == 3

    def test_history_still_works_with_store(self):
        store = InMemoryEventStore()
        bus = EventBus(event_store=store)
        bus.publish(DomainEvent(event_type="a", source="test", payload={}))
        bus.publish(DomainEvent(event_type="b", source="test", payload={}))
        assert len(bus.get_history()) == 2


class TestDeliveryGuarantees:
    """At-least-once delivery with retry and dead letter queue."""

    def test_at_least_once_retries_failed_handler(self):
        bus = EventBus(delivery_semantics=DeliverySemantics.AT_LEAST_ONCE)
        call_count = [0]

        def failing_handler(event):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient failure")

        bus.subscribe("test", failing_handler)
        bus.publish(DomainEvent(event_type="test", source="test", payload={}))
        assert call_count[0] == 3

    def test_dead_letter_queue_after_max_retries(self):
        bus = EventBus(
            delivery_semantics=DeliverySemantics.AT_LEAST_ONCE,
            max_retries=2,
        )
        call_count = [0]

        def always_failing(event):
            call_count[0] += 1
            raise RuntimeError("permanent failure")

        bus.subscribe("test", always_failing)
        bus.publish(DomainEvent(event_type="test", source="test", payload={}))
        assert call_count[0] == 3  # initial + 2 retries
        dlq = bus.get_dead_letter_queue()
        assert len(dlq) == 1

    def test_at_most_once_no_retry(self):
        bus = EventBus(delivery_semantics=DeliverySemantics.AT_MOST_ONCE)
        call_count = [0]

        def failing_handler(event):
            call_count[0] += 1
            raise RuntimeError("failure")

        bus.subscribe("test", failing_handler)
        bus.publish(DomainEvent(event_type="test", source="test", payload={}))
        assert call_count[0] == 1  # no retry

    def test_exactly_once_deduplication(self):
        bus = EventBus(delivery_semantics=DeliverySemantics.EXACTLY_ONCE)
        call_count = [0]

        def handler(event):
            call_count[0] += 1

        bus.subscribe("test", handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        bus.publish(event)
        bus.publish(event)  # same event_id
        assert call_count[0] == 1  # deduplicated

    def test_retry_with_exponential_backoff(self):
        bus = EventBus(
            delivery_semantics=DeliverySemantics.AT_LEAST_ONCE,
            max_retries=3,
            base_retry_delay=0.01,
        )
        timestamps = []

        def failing_handler(event):
            timestamps.append(time.time())
            if len(timestamps) < 3:
                raise RuntimeError("transient")

        bus.subscribe("test", failing_handler)
        bus.publish(DomainEvent(event_type="test", source="test", payload={}))
        assert len(timestamps) == 3
        # Second call should be delayed more than first
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        assert delay2 >= delay1

    def test_clear_dead_letter_queue(self):
        bus = EventBus(
            delivery_semantics=DeliverySemantics.AT_LEAST_ONCE,
            max_retries=1,
        )

        def always_failing(event):
            raise RuntimeError("failure")

        bus.subscribe("test", always_failing)
        bus.publish(DomainEvent(event_type="test", source="test", payload={}))
        assert len(bus.get_dead_letter_queue()) == 1
        bus.clear_dead_letter_queue()
        assert len(bus.get_dead_letter_queue()) == 0


class TestSchemaValidationIntegration:
    """Schema validation integrated into event bus."""

    def test_publish_validates_against_schema(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id", "value"},
            field_types={"sensor_id": str, "value": float},
        )
        registry.register(schema)
        bus = EventBus(schema_registry=registry)
        received = []
        bus.subscribe("sensor.reading", received.append)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1", "value": 42.0},
        )
        bus.publish(event)
        assert len(received) == 1

    def test_publish_rejects_invalid_payload(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id", "value"},
            field_types={"sensor_id": str, "value": float},
        )
        registry.register(schema)
        bus = EventBus(schema_registry=registry)
        received = []
        bus.subscribe("sensor.reading", received.append)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1"},  # missing value
        )
        bus.publish(event)
        assert len(received) == 0  # rejected

    def test_publish_without_schema_passes(self):
        registry = EventSchemaRegistry()
        bus = EventBus(schema_registry=registry)
        received = []
        bus.subscribe("unknown.type", received.append)
        event = DomainEvent(
            event_type="unknown.type",
            source="test",
            payload={"anything": "goes"},
        )
        bus.publish(event)
        assert len(received) == 1


class TestEventVersioning:
    """Event versioning and evolution."""

    def test_event_has_version_field(self):
        event = DomainEvent(event_type="test", source="test", payload={})
        assert hasattr(event, "version")
        assert event.version == 1

    def test_event_version_can_be_set(self):
        event = DomainEvent(event_type="test", source="test", payload={}, version=2)
        assert event.version == 2

    def test_schema_deprecation_warning(self):
        registry = EventSchemaRegistry()
        schema = EventSchema(
            event_type="sensor.reading",
            version=1,
            required_fields={"sensor_id"},
            field_types={"sensor_id": str},
            deprecated=True,
        )
        registry.register(schema)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1"},
        )
        # Should still validate but be flagged as deprecated
        assert registry.validate(event) is True
        assert registry.is_deprecated("sensor.reading") is True

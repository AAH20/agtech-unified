"""Tests for event batching: micro-batching for high-throughput handlers.

TDD-style: written before implementation in integration/event_bus.py.
"""

import asyncio
import time

import pytest

from src.integration.event_bus import DomainEvent, EventBus, EventType


class TestEventBatchingBasics:
    """Batching config and enable/disable."""

    def test_batching_disabled_by_default(self):
        bus = EventBus()
        assert bus._batch_size == 1
        assert bus._batch_timeout == 0.0

    def test_batching_enabled_with_config(self):
        bus = EventBus(batch_size=10, batch_timeout=0.05)
        assert bus._batch_size == 10
        assert bus._batch_timeout == 0.05

    def test_batching_disabled_with_size_1(self):
        bus = EventBus(batch_size=1, batch_timeout=0.05)
        assert bus._batch_size == 1

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            EventBus(batch_size=0)

    def test_invalid_batch_timeout_raises(self):
        with pytest.raises(ValueError):
            EventBus(batch_timeout=-1.0)


class TestEventBatchingSync:
    """Sync handlers receive batched events."""

    def test_single_event_not_batched(self):
        """With batch_size=1, each event is delivered individually."""
        bus = EventBus(batch_size=1, batch_timeout=0.05)
        received = []
        bus.subscribe(EventType.SENSOR_READING_RECEIVED, lambda e: received.append(e))
        event = DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test")
        bus.publish(event)
        assert len(received) == 1
        assert received[0].event_id == event.event_id

    def test_multiple_events_batched(self):
        """With batch_size=5, up to 5 events are delivered together."""
        bus = EventBus(batch_size=5, batch_timeout=0.05)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        for i in range(3):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        # Wait for batch to flush
        time.sleep(0.1)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_batch_flushes_on_timeout(self):
        """Batch flushes when timeout expires even if not full."""
        bus = EventBus(batch_size=10, batch_timeout=0.05)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        bus.publish(DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test"))
        time.sleep(0.1)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batch_flushes_when_full(self):
        """Batch flushes immediately when it reaches batch_size."""
        bus = EventBus(batch_size=3, batch_timeout=10.0)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        for i in range(3):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        # Should flush immediately without waiting for timeout
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_multiple_batches(self):
        """Events exceeding batch_size are split into multiple batches."""
        bus = EventBus(batch_size=2, batch_timeout=0.05)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        for i in range(5):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        time.sleep(0.1)
        assert len(batches) == 3  # 2 + 2 + 1
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_different_event_types_batched_separately(self):
        """Batches are per-event-type."""
        bus = EventBus(batch_size=5, batch_timeout=0.05)
        sensor_batches = []
        alert_batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: sensor_batches.append(list(events))
        )
        bus.subscribe(EventType.ALERT_TRIGGERED, lambda events: alert_batches.append(list(events)))
        bus.publish(DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test"))
        bus.publish(DomainEvent(event_type=EventType.ALERT_TRIGGERED, source="test"))
        time.sleep(0.1)
        assert len(sensor_batches) == 1
        assert len(alert_batches) == 1
        assert sensor_batches[0][0].event_type == EventType.SENSOR_READING_RECEIVED
        assert alert_batches[0][0].event_type == EventType.ALERT_TRIGGERED


class TestEventBatchingAsync:
    """Async handlers receive batched events."""

    def test_async_handler_receives_batch(self):
        bus = EventBus(batch_size=3, batch_timeout=0.05)
        batches = []

        async def handler(events):
            batches.append(list(events))

        bus.subscribe(EventType.SENSOR_READING_RECEIVED, handler)
        for i in range(3):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        time.sleep(0.1)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_publish_async_with_batching(self):
        bus = EventBus(batch_size=3, batch_timeout=0.05)
        batches = []

        async def handler(events):
            batches.append(list(events))

        bus.subscribe(EventType.SENSOR_READING_RECEIVED, handler)
        for i in range(3):
            asyncio.run(
                bus.publish_async(
                    DomainEvent(
                        event_type=EventType.SENSOR_READING_RECEIVED,
                        source="test",
                        payload={"i": i},
                    )
                )
            )
        time.sleep(0.1)
        assert len(batches) == 1
        assert len(batches[0]) == 3


class TestEventBatchingWithPersistence:
    """Batching works with event store persistence."""

    def test_batched_events_persisted(self, tmp_path):
        store_path = str(tmp_path / "events.jsonl")
        bus = EventBus(batch_size=5, batch_timeout=0.05, persist=store_path)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        for i in range(3):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        time.sleep(0.1)
        # Events should be in history
        history = bus.get_history(EventType.SENSOR_READING_RECEIVED)
        assert len(history) == 3
        # And in the store
        store_events = bus.replay()
        assert len(store_events) == 3


class TestEventBatchingWithSchemaValidation:
    """Batching respects schema validation."""

    def test_invalid_events_not_batched(self):
        from src.integration.event_bus import EventSchema, EventSchemaRegistry

        registry = EventSchemaRegistry()
        registry.register(
            EventSchema(
                event_type=EventType.SENSOR_READING_RECEIVED,
                version=1,
                required_fields={"sensor_id"},
            )
        )
        bus = EventBus(batch_size=5, batch_timeout=0.05, schema_registry=registry)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        # Invalid event (missing required field)
        bus.publish(
            DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={})
        )
        time.sleep(0.1)
        assert len(batches) == 0


class TestEventBatchingHighThroughput:
    """Batching improves throughput for high-volume publishing."""

    def test_high_volume_batching(self):
        """1000 events with batch_size=50 should produce ~20 batches."""
        bus = EventBus(batch_size=50, batch_timeout=1.0)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        for i in range(1000):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        # Should have flushed in batches of 50
        assert len(batches) == 20
        total = sum(len(b) for b in batches)
        assert total == 1000

    def test_batching_preserves_order(self):
        """Events within a batch preserve publish order."""
        bus = EventBus(batch_size=10, batch_timeout=1.0)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        for i in range(10):
            bus.publish(
                DomainEvent(
                    event_type=EventType.SENSOR_READING_RECEIVED, source="test", payload={"i": i}
                )
            )
        assert len(batches) == 1
        assert [e.payload["i"] for e in batches[0]] == list(range(10))


class TestEventBatchingUnsubscribe:
    """Unsubscribing stops batch delivery."""

    def test_unsubscribe_stops_batches(self):
        bus = EventBus(batch_size=5, batch_timeout=0.05)
        batches = []

        def handler(events):
            batches.append(list(events))

        bus.subscribe(EventType.SENSOR_READING_RECEIVED, handler)
        bus.publish(DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test"))
        time.sleep(0.1)
        assert len(batches) == 1
        bus.unsubscribe(EventType.SENSOR_READING_RECEIVED, handler)
        bus.publish(DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test"))
        time.sleep(0.1)
        assert len(batches) == 1  # no new batch


class TestEventBatchingClear:
    """Clearing the bus clears pending batches."""

    def test_clear_clears_pending_batches(self):
        bus = EventBus(batch_size=10, batch_timeout=10.0)
        batches = []
        bus.subscribe(
            EventType.SENSOR_READING_RECEIVED, lambda events: batches.append(list(events))
        )
        bus.publish(DomainEvent(event_type=EventType.SENSOR_READING_RECEIVED, source="test"))
        bus.clear()
        time.sleep(0.1)
        assert len(batches) == 0

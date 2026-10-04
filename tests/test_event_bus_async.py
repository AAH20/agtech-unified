"""Tests for async event handler support in the event bus."""

import asyncio
import time

import pytest

from src.integration.event_bus import DeliverySemantics, DomainEvent, EventBus


class TestAsyncHandlerInvocation:
    """Async handlers are properly detected and scheduled."""

    @pytest.mark.asyncio
    async def test_async_handler_called(self):
        bus = EventBus()
        received = []

        async def async_handler(event):
            received.append(event)

        bus.subscribe("test", async_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert len(received) == 1
        assert received[0].event_id == event.event_id

    @pytest.mark.asyncio
    async def test_sync_handler_still_works_with_publish_async(self):
        bus = EventBus()
        received = []

        def sync_handler(event):
            received.append(event)

        bus.subscribe("test", sync_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_handlers(self):
        bus = EventBus()
        sync_received = []
        async_received = []

        def sync_handler(event):
            sync_received.append(event)

        async def async_handler(event):
            async_received.append(event)

        bus.subscribe("test", sync_handler)
        bus.subscribe("test", async_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert len(sync_received) == 1
        assert len(async_received) == 1

    @pytest.mark.asyncio
    async def test_multiple_async_handlers_all_called(self):
        bus = EventBus()
        received = []

        async def handler1(event):
            received.append(("h1", event))

        async def handler2(event):
            received.append(("h2", event))

        bus.subscribe("test", handler1)
        bus.subscribe("test", handler2)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert len(received) == 2
        names = [r[0] for r in received]
        assert "h1" in names
        assert "h2" in names


class TestAsyncHandlerIsolation:
    """Async handler exceptions don't break the bus."""

    @pytest.mark.asyncio
    async def test_async_handler_exception_isolated(self):
        bus = EventBus()
        received = []

        async def failing_handler(event):
            raise RuntimeError("async failure")

        async def good_handler(event):
            received.append(event)

        bus.subscribe("test", failing_handler)
        bus.subscribe("test", good_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        # Should not raise
        await bus.publish_async(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_handler_exception_logged(self, caplog):
        bus = EventBus()

        async def failing_handler(event):
            raise RuntimeError("async boom")

        bus.subscribe("test", failing_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        with caplog.at_level("ERROR"):
            await bus.publish_async(event)
        assert "async boom" in caplog.text


class TestAsyncDeliverySemantics:
    """Delivery guarantees work with async handlers."""

    @pytest.mark.asyncio
    async def test_at_least_once_retries_async_handler(self):
        bus = EventBus(delivery_semantics=DeliverySemantics.AT_LEAST_ONCE)
        call_count = [0]

        async def failing_handler(event):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient")

        bus.subscribe("test", failing_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_at_most_once_no_retry_async(self):
        bus = EventBus(delivery_semantics=DeliverySemantics.AT_MOST_ONCE)
        call_count = [0]

        async def failing_handler(event):
            call_count[0] += 1
            raise RuntimeError("failure")

        bus.subscribe("test", failing_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_exactly_once_deduplication_async(self):
        bus = EventBus(delivery_semantics=DeliverySemantics.EXACTLY_ONCE)
        call_count = [0]

        async def handler(event):
            call_count[0] += 1

        bus.subscribe("test", handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        await bus.publish_async(event)
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_dead_letter_queue_after_max_retries_async(self):
        bus = EventBus(
            delivery_semantics=DeliverySemantics.AT_LEAST_ONCE,
            max_retries=2,
        )
        call_count = [0]

        async def always_failing(event):
            call_count[0] += 1
            raise RuntimeError("permanent")

        bus.subscribe("test", always_failing)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert call_count[0] == 3
        dlq = bus.get_dead_letter_queue()
        assert len(dlq) == 1


class TestAsyncNonBlocking:
    """Sync publish schedules async handlers without blocking."""

    @pytest.mark.asyncio
    async def test_publish_schedules_async_handlers(self):
        """Sync publish should schedule async handlers and return immediately."""
        bus = EventBus()
        received = []

        async def async_handler(event):
            await asyncio.sleep(0.05)
            received.append(event)

        bus.subscribe("test", async_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        start = time.monotonic()
        bus.publish(event)
        elapsed = time.monotonic() - start
        # Should return quickly (non-blocking) since we're in an async context
        assert elapsed < 0.04
        # Handler was scheduled but hasn't completed yet
        await asyncio.sleep(0.1)
        assert len(received) == 1

    def test_publish_returns_handler_count_with_async(self):
        bus = EventBus()

        async def async_handler(event):
            pass

        def sync_handler(event):
            pass

        bus.subscribe("test", async_handler)
        bus.subscribe("test", sync_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        count = bus.publish(event)
        assert count == 2


class TestAsyncSchemaValidation:
    """Schema validation works with async handlers."""

    @pytest.mark.asyncio
    async def test_async_publish_validates_schema(self):
        from src.integration.event_bus import EventSchema, EventSchemaRegistry

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

        async def async_handler(event):
            received.append(event)

        bus.subscribe("sensor.reading", async_handler)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1", "value": 42.0},
        )
        await bus.publish_async(event)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_publish_rejects_invalid_event(self):
        from src.integration.event_bus import EventSchema, EventSchemaRegistry

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

        async def async_handler(event):
            received.append(event)

        bus.subscribe("sensor.reading", async_handler)
        event = DomainEvent(
            event_type="sensor.reading",
            source="test",
            payload={"sensor_id": "s1"},  # missing value
        )
        await bus.publish_async(event)
        assert len(received) == 0


class TestAsyncEventBusIntegration:
    """Integration tests for async event bus with other features."""

    @pytest.mark.asyncio
    async def test_async_handler_receives_correct_event_data(self):
        bus = EventBus()
        received = []

        async def async_handler(event):
            received.append(event)

        bus.subscribe("test", async_handler)
        event = DomainEvent(
            event_type="test",
            source="test_module",
            payload={"key": "value", "number": 42},
        )
        await bus.publish_async(event)
        assert len(received) == 1
        assert received[0].event_type == "test"
        assert received[0].source == "test_module"
        assert received[0].payload["key"] == "value"
        assert received[0].payload["number"] == 42

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe_async_handler(self):
        bus = EventBus()
        received = []

        async def async_handler(event):
            received.append(event)

        bus.subscribe("test", async_handler)
        bus.unsubscribe("test", async_handler)
        event = DomainEvent(event_type="test", source="test", payload={})
        await bus.publish_async(event)
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_async_handler_with_persistence(self):
        """Async handlers work with event store persistence."""
        from src.integration.event_bus import InMemoryEventStore

        store = InMemoryEventStore()
        bus = EventBus(event_store=store)
        received = []

        async def async_handler(event):
            received.append(event)

        bus.subscribe("test", async_handler)
        event = DomainEvent(event_type="test", source="test", payload={"data": 123})
        await bus.publish_async(event)
        assert len(received) == 1
        assert len(store.get_all()) == 1
        assert store.get_all()[0].payload["data"] == 123

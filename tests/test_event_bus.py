"""Test cross-module event bus: pub/sub, domain events, handlers."""

from src.integration.event_bus import DomainEvent, EventBus, EventType


def _make_event(event_type="reading", source="sensor", **payload):
    return DomainEvent(event_type=event_type, source=source, payload=payload)


class TestPubSub:
    """Basic publish/subscribe behavior."""

    def test_subscribe_and_publish(self):
        """A subscribed handler is invoked when a matching event is published."""
        bus = EventBus()
        received = []
        bus.subscribe("reading", received.append)

        bus.publish(_make_event("reading", value=1))

        assert len(received) == 1
        assert received[0].payload["value"] == 1

    def test_unsubscribe_stops_delivery(self):
        """An unsubscribed handler no longer receives events."""
        bus = EventBus()
        received = []
        bus.subscribe("reading", received.append)
        bus.unsubscribe("reading", received.append)

        bus.publish(_make_event("reading"))

        assert received == []

    def test_multiple_subscribers_all_called(self):
        """Every handler subscribed to an event type is invoked."""
        bus = EventBus()
        first, second = [], []
        bus.subscribe("alert", first.append)
        bus.subscribe("alert", second.append)

        bus.publish(_make_event("alert"))

        assert len(first) == 1
        assert len(second) == 1

    def test_publish_without_subscribers_is_noop(self):
        """Publishing to a topic with no subscribers returns 0 and does not raise."""
        bus = EventBus()

        handled = bus.publish(_make_event("nobody_listening"))

        assert handled == 0
        assert bus.get_history() != []


class TestDomainEvents:
    """DomainEvent construction and identity."""

    def test_event_has_unique_id_and_timestamp(self):
        """Each event gets a distinct id and a monotonic timestamp."""
        e1 = _make_event()
        e2 = _make_event()

        assert e1.event_id != e2.event_id
        assert e2.timestamp >= e1.timestamp

    def test_event_carries_type_source_payload(self):
        """Event fields round-trip exactly as constructed."""
        event = _make_event("alert", source="recommender", severity="high")

        assert event.event_type == "alert"
        assert event.source == "recommender"
        assert event.payload == {"severity": "high"}

    def test_predefined_event_types_exist(self):
        """The EventType catalog exposes the domain events named in the gap analysis."""
        for name in (
            "SENSOR_READING_RECEIVED",
            "ALERT_TRIGGERED",
            "SIMULATION_COMPLETED",
            "TASK_ASSIGNED",
            "TASK_COMPLETED",
        ):
            assert isinstance(getattr(EventType, name), str)


class TestHistoryAndHandlers:
    """History tracking and handler isolation."""

    def test_history_records_all_published_events(self):
        """Every published event is retained in history."""
        bus = EventBus()
        bus.publish(_make_event("a"))
        bus.publish(_make_event("b"))

        assert len(bus.get_history()) == 2

    def test_history_filterable_by_event_type(self):
        """get_history(event_type) returns only matching events."""
        bus = EventBus()
        bus.publish(_make_event("a"))
        bus.publish(_make_event("b"))
        bus.publish(_make_event("a"))

        assert len(bus.get_history("a")) == 2
        assert len(bus.get_history("b")) == 1

    def test_clear_history(self):
        """clear() empties the recorded history."""
        bus = EventBus()
        bus.publish(_make_event("a"))
        bus.clear()

        assert bus.get_history() == []

    def test_handler_exception_does_not_break_bus(self):
        """A failing handler is isolated; other handlers still run."""
        bus = EventBus()
        received = []

        def boom(_event):
            raise RuntimeError("handler failure")

        bus.subscribe("x", boom)
        bus.subscribe("x", received.append)

        bus.publish(_make_event("x"))

        assert len(received) == 1

    def test_subscriber_count(self):
        """subscriber_count reflects active registrations per event type."""
        bus = EventBus()
        handler = lambda _e: None  # noqa: E731
        bus.subscribe("x", handler)
        bus.subscribe("x", handler)  # duplicate registration ignored

        assert bus.subscriber_count("x") == 1
        assert bus.subscriber_count("y") == 0

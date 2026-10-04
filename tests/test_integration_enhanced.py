"""Test enhanced event bus persistence, delivery guarantees, and FarmState serialization."""

import json
import os
import tempfile

from src.integration.event_bus import DomainEvent, EventBus
from src.integration.farm_state import FarmState


class TestEventBusPersistence:
    """Test event bus persistence to JSONL file."""

    def test_persist_creates_jsonl_file(self):
        """When persist is set, events are written to a JSONL file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            filepath = f.name
        try:
            bus = EventBus(persist=filepath)
            event = DomainEvent(event_type="test", source="test", payload={"key": "value"})
            bus.publish(event)

            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["event_type"] == "test"
            assert data["source"] == "test"
            assert data["payload"] == {"key": "value"}
        finally:
            os.unlink(filepath)

    def test_persist_multiple_events(self):
        """Multiple events are appended to the JSONL file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            filepath = f.name
        try:
            bus = EventBus(persist=filepath)
            bus.publish(DomainEvent(event_type="test1", source="test"))
            bus.publish(DomainEvent(event_type="test2", source="test"))

            with open(filepath, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
        finally:
            os.unlink(filepath)


class TestDeliveryGuarantee:
    """Test delivery guarantee with retries."""

    def test_delivery_guarantee_retries_on_failure(self):
        """A failing handler is retried up to 3 times."""
        bus = EventBus()
        call_count = [0]

        def failing_handler(event):
            call_count[0] += 1
            raise RuntimeError("always fails")

        bus.subscribe("test", failing_handler)
        bus.publish(
            DomainEvent(event_type="test", source="test"), delivery_guarantee="at_least_once"
        )

        # 1 initial + 3 retries = 4 calls
        assert call_count[0] == 4

    def test_delivery_guarantee_succeeds_after_retry(self):
        """A handler that fails once then succeeds is retried and succeeds."""
        bus = EventBus()
        call_count = [0]
        received = []

        def flaky_handler(event):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first attempt fails")
            received.append(event)

        bus.subscribe("test", flaky_handler)
        bus.publish(
            DomainEvent(event_type="test", source="test"), delivery_guarantee="at_least_once"
        )

        assert call_count[0] == 2
        assert len(received) == 1

    def test_delivery_guarantee_adds_to_dlq_after_max_retries(self):
        """After max retries, the event is added to the dead letter queue."""
        bus = EventBus()

        def always_fails(event):
            raise RuntimeError("always fails")

        bus.subscribe("test", always_fails)
        bus.publish(
            DomainEvent(event_type="test", source="test"), delivery_guarantee="at_least_once"
        )

        dlq = bus.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0].event_type == "test"

    def test_no_delivery_guarantee_no_retry(self):
        """Without delivery_guarantee, failing handlers are not retried."""
        bus = EventBus()
        call_count = [0]

        def failing_handler(event):
            call_count[0] += 1
            raise RuntimeError("fails")

        bus.subscribe("test", failing_handler)
        bus.publish(DomainEvent(event_type="test", source="test"))

        assert call_count[0] == 1


class TestFarmStateSerialization:
    """Test FarmState to_dict and from_dict."""

    def test_to_dict_contains_all_fields(self):
        """to_dict returns all FarmState fields."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
            timestamp=1234567890.0,
        )
        d = state.to_dict()
        assert d["soil_moisture"] == 0.5
        assert d["temperature"] == 25.0
        assert d["crop_height"] == 0.3
        assert d["nutrient_level"] == 0.6
        assert d["pest_pressure"] == 0.2
        assert d["timestamp"] == 1234567890.0

    def test_from_dict_round_trip(self):
        """from_dict(to_dict()) preserves all field values."""
        original = FarmState(
            soil_moisture=0.42,
            temperature=18.5,
            crop_height=1.2,
            nutrient_level=0.75,
            pest_pressure=0.05,
            timestamp=9999999.0,
        )
        restored = FarmState.from_dict(original.to_dict())
        assert restored.soil_moisture == original.soil_moisture
        assert restored.temperature == original.temperature
        assert restored.crop_height == original.crop_height
        assert restored.nutrient_level == original.nutrient_level
        assert restored.pest_pressure == original.pest_pressure
        assert restored.timestamp == original.timestamp

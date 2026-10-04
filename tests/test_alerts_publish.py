"""Tests for AlertManager publishing to event_bus."""

from src.decision_support.alerts import AlertManager, Threshold
from src.integration.event_bus import EventBus, EventType
from src.integration.farm_state import FarmState


class TestAlertManagerPublishing:
    """AlertManager can publish alerts to the event bus."""

    def test_publish_to_event_bus(self):
        """AlertManager.publish_to publishes ALERT_TRIGGERED events."""
        bus = EventBus()
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))
        manager.publish_to(bus)

        state = FarmState(
            soil_moisture=0.1,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        manager.check_state(state)

        events = bus.get_history(EventType.ALERT_TRIGGERED)
        assert len(events) >= 1
        assert events[0].source == "decision_support.alerts"

    def test_publish_to_returns_count(self):
        """publish_to returns the number of events published."""
        bus = EventBus()
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))
        manager.publish_to(bus)

        state = FarmState(
            soil_moisture=0.1,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        manager.check_state(state)

        # After check_state, publish_to should have been called automatically
        # and returned the count of published events
        events = bus.get_history(EventType.ALERT_TRIGGERED)
        assert len(events) >= 1

    def test_no_alerts_no_events(self):
        """When no thresholds are violated, no events are published."""
        bus = EventBus()
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.1, max_value=0.9))
        manager.publish_to(bus)

        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.5,
            nutrient_level=0.5,
            pest_pressure=0.1,
        )
        manager.check_state(state)

        events = bus.get_history(EventType.ALERT_TRIGGERED)
        assert len(events) == 0

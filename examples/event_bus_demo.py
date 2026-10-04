"""Event bus demo: cross-module pub/sub messaging."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.integration.event_bus import EventBus, EventType, DomainEvent


def main():
    bus = EventBus()

    def handler(event):
        print(f"Received: {event.event_type} - {event.payload}")

    bus.subscribe(EventType.SENSOR_READING, handler)
    event = DomainEvent(
        event_type=EventType.SENSOR_READING,
        source="soil-01",
        payload={"temperature": 25.3},
    )
    bus.publish(event)
    print("Event published successfully")


if __name__ == "__main__":
    main()

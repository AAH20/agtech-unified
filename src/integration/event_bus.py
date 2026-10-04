"""Cross-module event bus with pub/sub and domain events.

Provides the central event-driven backbone requested by GAP-008: modules
publish domain events, interested modules subscribe, and no module needs
a direct import of another to react to its state changes.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType:
    """Catalog of domain events exchanged between modules."""

    SENSOR_READING_RECEIVED = "sensor_reading_received"
    ALERT_TRIGGERED = "alert_triggered"
    SIMULATION_COMPLETED = "simulation_completed"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    RECOMMENDATION_PRODUCED = "recommendation_produced"
    ROUTE_PLANNED = "route_planned"


@dataclass
class DomainEvent:
    """A domain event published to the bus.

    Attributes:
        event_type: One of the EventType constants.
        source: Name of the publishing module (e.g. "iot.data_pipeline").
        payload: Event-specific data.
        event_id: Unique identifier (auto-generated).
        timestamp: Unix timestamp (auto-generated).
    """
    event_type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


EventHandler = Callable[[DomainEvent], None]


class EventBus:
    """In-process publish/subscribe event bus.

    Handlers are invoked synchronously in subscription order. A handler
    that raises is isolated: the error is logged and remaining handlers
    still run, so one bad subscriber cannot break the bus.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._history: List[DomainEvent] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register handler for events of the given type.

        Duplicate registrations of the same handler are ignored.
        """
        handlers = self._subscribers[event_type]
        if handler not in handlers:
            handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously registered handler. No-op if not registered."""
        handlers = self._subscribers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def publish(self, event: DomainEvent) -> int:
        """Publish an event to all subscribers of its type.

        Returns the number of handlers invoked.
        """
        self._history.append(event)
        handlers = list(self._subscribers.get(event.event_type, ()))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Event handler %r failed for event %s",
                    getattr(handler, "__name__", handler),
                    event.event_id,
                )
        return len(handlers)

    def get_history(self, event_type: Optional[str] = None) -> List[DomainEvent]:
        """Return recorded events, optionally filtered by event type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]

    def clear(self) -> None:
        """Drop all recorded history. Subscriptions are kept."""
        self._history.clear()

    def subscriber_count(self, event_type: str) -> int:
        """Number of handlers registered for the given event type."""
        return len(self._subscribers.get(event_type, ()))

"""Cross-module event bus with pub/sub, domain events, persistence, and delivery guarantees.

Provides the central event-driven backbone requested by GAP-008: modules
publish domain events, interested modules subscribe, and no module needs
a direct import of another to react to its state changes.

Enhanced with:
- Pluggable event store backends (in-memory, file-based)
- Configurable delivery semantics (at-most-once, at-least-once, exactly-once)
- Dead letter queue for failed events
- Event schema validation and versioning
- Event replay for recovery
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
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


class DeliverySemantics(str, Enum):
    """Delivery guarantee levels."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class DomainEvent:
    """A domain event published to the bus.

    Attributes:
        event_type: One of the EventType constants.
        source: Name of the publishing module (e.g. "iot.data_pipeline").
        payload: Event-specific data.
        event_id: Unique identifier (auto-generated).
        timestamp: Unix timestamp (auto-generated).
        version: Schema version (default 1).
    """

    event_type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: int = 1


EventHandler = Callable[[DomainEvent], None]


@dataclass
class EventSchema:
    """Schema definition for an event type."""

    event_type: str
    version: int
    required_fields: set = field(default_factory=set)
    field_types: Dict[str, type] = field(default_factory=dict)
    deprecated: bool = False


class EventSchemaRegistry:
    """Registry for event schemas with validation."""

    def __init__(self) -> None:
        self._schemas: Dict[str, List[EventSchema]] = defaultdict(list)

    def register(self, schema: EventSchema) -> None:
        """Register a schema for an event type."""
        self._schemas[schema.event_type].append(schema)

    def validate(self, event: DomainEvent) -> bool:
        """Validate an event against its registered schema.

        Returns True if valid or no schema is registered.
        """
        schemas = self._schemas.get(event.event_type)
        if not schemas:
            return True

        # Use the latest schema version
        schema = max(schemas, key=lambda s: s.version)

        # Check required fields
        for field_name in schema.required_fields:
            if field_name not in event.payload:
                return False

        # Check field types
        for field_name, expected_type in schema.field_types.items():
            if field_name in event.payload:
                value = event.payload[field_name]
                if not isinstance(value, expected_type):
                    return False

        return True

    def get_schema(self, event_type: str) -> Optional[EventSchema]:
        """Get the latest schema for an event type."""
        schemas = self._schemas.get(event_type)
        if not schemas:
            return None
        return max(schemas, key=lambda s: s.version)

    def is_deprecated(self, event_type: str) -> bool:
        """Check if an event type is deprecated."""
        schema = self.get_schema(event_type)
        return schema.deprecated if schema else False


class InMemoryEventStore:
    """In-memory event store backend."""

    def __init__(self) -> None:
        self._events: List[DomainEvent] = []

    def append(self, event: DomainEvent) -> None:
        """Append an event to the store."""
        self._events.append(event)

    def get_all(self) -> List[DomainEvent]:
        """Return all events in the store."""
        return list(self._events)

    def get_by_type(self, event_type: str) -> List[DomainEvent]:
        """Return events filtered by type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_by_time_range(self, start: float, end: float) -> List[DomainEvent]:
        """Return events within a time range."""
        return [e for e in self._events if start <= e.timestamp <= end]

    def replay(self) -> List[DomainEvent]:
        """Return all events for replay."""
        return list(self._events)

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()


class FileEventStore:
    """File-based event store for persistence.

    Stores events as JSON Lines for append-only durability.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Ensure the file exists."""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        if not os.path.exists(self._filepath):
            with open(self._filepath, "w"):
                pass

    def append(self, event: DomainEvent) -> None:
        """Append an event to the file."""
        with open(self._filepath, "a") as f:
            f.write(
                json.dumps(
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "source": event.source,
                        "payload": event.payload,
                        "timestamp": event.timestamp,
                        "version": event.version,
                    }
                )
                + "\n"
            )

    def get_all(self) -> List[DomainEvent]:
        """Read all events from the file."""
        events = []
        if not os.path.exists(self._filepath):
            return events
        with open(self._filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                events.append(
                    DomainEvent(
                        event_id=data["event_id"],
                        event_type=data["event_type"],
                        source=data["source"],
                        payload=data["payload"],
                        timestamp=data["timestamp"],
                        version=data.get("version", 1),
                    )
                )
        return events

    def get_by_type(self, event_type: str) -> List[DomainEvent]:
        """Return events filtered by type."""
        return [e for e in self.get_all() if e.event_type == event_type]

    def get_by_time_range(self, start: float, end: float) -> List[DomainEvent]:
        """Return events within a time range."""
        return [e for e in self.get_all() if start <= e.timestamp <= end]

    def replay(self) -> List[DomainEvent]:
        """Return all events for replay."""
        return self.get_all()

    def clear(self) -> None:
        """Clear all events."""
        with open(self._filepath, "w"):
            pass


class EventBus:
    """In-process publish/subscribe event bus.

    Handlers are invoked synchronously in subscription order. A handler
    that raises is isolated: the error is logged and remaining handlers
    still run, so one bad subscriber cannot break the bus.

    Enhanced with:
    - Pluggable event store for persistence
    - Configurable delivery semantics
    - Dead letter queue for failed events
    - Schema validation
    """

    def __init__(
        self,
        event_store: Optional[Any] = None,
        delivery_semantics: DeliverySemantics = DeliverySemantics.AT_MOST_ONCE,
        max_retries: int = 3,
        base_retry_delay: float = 0.1,
        schema_registry: Optional[EventSchemaRegistry] = None,
        persist: Optional[str] = None,
    ) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._history: List[DomainEvent] = []
        self._event_store = event_store
        self._delivery_semantics = delivery_semantics
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._schema_registry = schema_registry
        self._dead_letter_queue: List[DomainEvent] = []
        self._processed_event_ids: set = set()
        if persist:
            self._event_store = FileEventStore(persist)

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

    def publish(self, event: DomainEvent, delivery_guarantee: Optional[str] = None) -> int:
        """Publish an event to all subscribers of its type.

        Args:
            event: The event to publish.
            delivery_guarantee: If "at_least_once", failed handler calls are retried
                up to max_retries times. If None, uses the bus's default delivery semantics.

        Returns the number of handlers invoked.
        """
        # Schema validation
        if self._schema_registry and not self._schema_registry.validate(event):
            logger.warning(
                "Event %s failed schema validation, skipping",
                event.event_id,
            )
            return 0

        # Deduplication for exactly-once
        if self._delivery_semantics == DeliverySemantics.EXACTLY_ONCE:
            if event.event_id in self._processed_event_ids:
                logger.debug(
                    "Event %s already processed, skipping (exactly-once)",
                    event.event_id,
                )
                return 0
            self._processed_event_ids.add(event.event_id)

        # Persist to event store
        if self._event_store:
            self._event_store.append(event)

        self._history.append(event)
        handlers = list(self._subscribers.get(event.event_type, ()))

        if delivery_guarantee == "at_least_once":
            return self._publish_at_least_once(event, handlers)
        elif self._delivery_semantics == DeliverySemantics.AT_MOST_ONCE:
            return self._publish_at_most_once(event, handlers)
        elif self._delivery_semantics == DeliverySemantics.AT_LEAST_ONCE:
            return self._publish_at_least_once(event, handlers)
        else:
            return self._publish_at_most_once(event, handlers)

    def _publish_at_most_once(self, event: DomainEvent, handlers: List[EventHandler]) -> int:
        """Publish with at-most-once semantics (no retry)."""
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

    def _publish_at_least_once(self, event: DomainEvent, handlers: List[EventHandler]) -> int:
        """Publish with at-least-once semantics (retry on failure)."""
        for handler in handlers:
            self._invoke_with_retry(event, handler)
        return len(handlers)

    def _invoke_with_retry(self, event: DomainEvent, handler: EventHandler) -> None:
        """Invoke a handler with retry logic."""
        for attempt in range(self._max_retries + 1):
            try:
                handler(event)
                return
            except Exception:
                if attempt < self._max_retries:
                    delay = self._base_retry_delay * (2**attempt)
                    logger.warning(
                        "Event handler %r failed for event %s (attempt %d/%d), retrying in %.2fs",
                        getattr(handler, "__name__", handler),
                        event.event_id,
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.exception(
                        "Event handler %r failed for event %s after %d attempts, moving to DLQ",
                        getattr(handler, "__name__", handler),
                        event.event_id,
                        self._max_retries + 1,
                    )
                    self._dead_letter_queue.append(event)

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

    def get_dead_letter_queue(self) -> List[DomainEvent]:
        """Return events that exceeded max retries."""
        return list(self._dead_letter_queue)

    def clear_dead_letter_queue(self) -> None:
        """Clear the dead letter queue."""
        self._dead_letter_queue.clear()

    def replay(self) -> List[DomainEvent]:
        """Replay all events from the event store."""
        if self._event_store:
            return self._event_store.replay()
        return list(self._history)

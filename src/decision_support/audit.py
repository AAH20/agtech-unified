"""Audit logging for API access tracking and compliance."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Categories of auditable events."""

    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    ACTUATOR_COMMAND = "actuator_command"
    ALERT_STATE_CHANGE = "alert_state_change"
    USER_ACTION = "user_action"
    SPOOFING_DETECTED = "spoofing_detected"
    RECOMMENDATION = "recommendation"


@dataclass
class AuditEvent:
    """A single immutable audit event."""

    event_id: str
    timestamp: float
    event_type: AuditEventType
    actor: str
    action: str
    resource: str
    tenant_id: Optional[str] = None
    request_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    status: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to a JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
        }

    def to_json(self) -> str:
        """Serialize the event to a canonical JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


class AuditLogger:
    """Thread-safe, tamper-evident audit logger for API access tracking.

    Events are hash-chained: each event records the hash of the previous
    event, so any modification or deletion of historical entries breaks the
    chain and is detectable via verify_chain().
    """

    def __init__(self, actor: str = "system"):
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._actor = actor
        self._last_hash: Optional[str] = None
        self._request_index: Dict[str, List[AuditEvent]] = {}

    @property
    def event_count(self) -> int:
        """Total number of recorded events."""
        return len(self._events)

    def _compute_hash(self, event: AuditEvent) -> str:
        """Compute the tamper-evident hash for an event."""
        payload = event.to_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str,
        actor: Optional[str] = None,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> AuditEvent:
        """Record an audit event and return it."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp if timestamp is not None else time.time(),
            event_type=event_type,
            actor=actor or self._actor,
            action=action,
            resource=resource,
            tenant_id=tenant_id,
            request_id=request_id,
            source_ip=source_ip,
            user_agent=user_agent,
            status=status,
            latency_ms=latency_ms,
            metadata=metadata or {},
            previous_hash=self._last_hash,
        )
        with self._lock:
            self._last_hash = self._compute_hash(event)
            self._events.append(event)
            if request_id is not None:
                self._request_index.setdefault(request_id, []).append(event)
        logger.debug("Audit: %s %s %s", event_type.value, action, resource)
        return event

    def log_api_request(
        self,
        method: str,
        endpoint: str,
        tenant_id: Optional[str] = None,
        actor: str = "anonymous",
        request_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: Optional[str] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Convenience method for API request/response audit entries."""
        return self.log(
            event_type=AuditEventType.API_REQUEST,
            action=method.upper(),
            resource=endpoint,
            actor=actor,
            tenant_id=tenant_id,
            request_id=request_id,
            source_ip=source_ip,
            user_agent=user_agent,
            status=status,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        tenant_id: Optional[str] = None,
        actor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Query recorded events with optional filters."""
        with self._lock:
            events = list(self._events)
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        if tenant_id is not None:
            events = [e for e in events if e.tenant_id == tenant_id]
        if actor is not None:
            events = [e for e in events if e.actor == actor]
        events.sort(key=lambda e: e.timestamp)
        if limit is not None:
            events = events[-limit:]
        return events

    def get_request_trace(self, request_id: str) -> List[AuditEvent]:
        """Get all events belonging to a single request id."""
        with self._lock:
            return list(self._request_index.get(request_id, []))

    def verify_chain(self) -> bool:
        """Verify the tamper-evident hash chain. Returns True if intact."""
        with self._lock:
            events = list(self._events)
        previous_hash: Optional[str] = None
        for event in events:
            if event.previous_hash != previous_hash:
                return False
            # Recompute hash over the event content as stored.
            recomputed = hashlib.sha256(
                event.to_json().encode("utf-8")
            ).hexdigest()
            previous_hash = recomputed
        return True

    def export_json(self) -> str:
        """Export all events as a JSON array string."""
        with self._lock:
            events = list(self._events)
        return json.dumps([e.to_dict() for e in events], indent=2, default=str)

    def clear(self) -> None:
        """Clear all recorded events (mainly for testing)."""
        with self._lock:
            self._events.clear()
            self._request_index.clear()
            self._last_hash = None

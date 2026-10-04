"""Alert management system for farm monitoring."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.decision_support.recommender import FarmState
from src.integration.event_bus import DomainEvent, EventBus, EventType

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert."""

    id: str
    timestamp: float
    severity: AlertSeverity
    metric: str
    message: str
    value: float
    threshold: float
    acknowledged: bool = False
    resolved: bool = False
    tenant_id: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "metric": self.metric,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "tenant_id": self.tenant_id,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
        }


@dataclass
class Threshold:
    """Threshold configuration for a metric."""

    metric: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    severity: AlertSeverity = AlertSeverity.WARNING
    tenant_id: Optional[str] = None


@dataclass
class NotificationChannel:
    """Notification channel configuration."""

    name: str
    channel_type: str
    target: str
    min_severity: AlertSeverity = AlertSeverity.INFO
    tenant_id: Optional[str] = None


@dataclass
class EscalationPolicy:
    """Escalation policy for unacknowledged alerts."""

    name: str
    timeout_seconds: float
    escalation_targets: List[str]
    severity_trigger: AlertSeverity = AlertSeverity.CRITICAL
    tenant_id: Optional[str] = None


class AlertManager:
    """Manages alerts based on configurable thresholds with multi-tenant isolation."""

    def __init__(self, tenant_id: Optional[str] = None, audit_logger: Any = None):
        self._thresholds: List[Threshold] = []
        self._channels: List[NotificationChannel] = []
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        self._handlers: List[Callable] = []
        self._bus: Optional[EventBus] = None
        self._tenant_id = tenant_id
        self._audit_logger = audit_logger
        self._escalation_policies: List[EscalationPolicy] = []
        self._dedup_windows: Dict[str, float] = {}
        self._dedup_window_seconds: float = 300.0

    @property
    def tenant_id(self) -> Optional[str]:
        """Tenant ID for this alert manager."""
        return self._tenant_id

    @property
    def alert_count(self) -> int:
        """Total number of alerts created."""
        return len(self._alerts)

    @property
    def active_alert_count(self) -> int:
        """Number of non-resolved alerts."""
        return len([a for a in self._alerts if not a.resolved])

    def add_threshold(self, threshold: Threshold) -> None:
        """Add a threshold rule."""
        if self._tenant_id and threshold.tenant_id and threshold.tenant_id != self._tenant_id:
            raise ValueError(
                f"Threshold tenant {threshold.tenant_id} does not match "
                f"manager tenant {self._tenant_id}"
            )
        self._thresholds.append(threshold)

    def remove_threshold(self, metric: str) -> bool:
        """Remove all thresholds for a metric. Returns True if any removed."""
        original_len = len(self._thresholds)
        self._thresholds = [t for t in self._thresholds if t.metric != metric]
        return len(self._thresholds) < original_len

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        if self._tenant_id and channel.tenant_id and channel.tenant_id != self._tenant_id:
            raise ValueError(
                f"Channel tenant {channel.tenant_id} does not match "
                f"manager tenant {self._tenant_id}"
            )
        self._channels.append(channel)

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy."""
        if self._tenant_id and policy.tenant_id and policy.tenant_id != self._tenant_id:
            raise ValueError(
                f"Policy tenant {policy.tenant_id} does not match manager tenant {self._tenant_id}"
            )
        self._escalation_policies.append(policy)

    def remove_channel(self, name: str) -> bool:
        """Remove a notification channel by name."""
        original_len = len(self._channels)
        self._channels = [c for c in self._channels if c.name != name]
        return len(self._channels) < original_len

    def check_value(
        self,
        metric: str,
        value: float,
        timestamp: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Alert]:
        """Check a value against thresholds and create alerts."""
        if timestamp is None:
            timestamp = time.time()

        effective_tenant = tenant_id or self._tenant_id

        alerts = []
        for threshold in self._thresholds:
            if threshold.metric != metric:
                continue
            if threshold.tenant_id and effective_tenant and threshold.tenant_id != effective_tenant:
                continue

            triggered = False
            if threshold.min_value is not None and value < threshold.min_value:
                triggered = True
            if threshold.max_value is not None and value > threshold.max_value:
                triggered = True

            if triggered:
                # Deduplication check
                dedup_key = (
                    f"{effective_tenant}:{metric}:{threshold.min_value}-{threshold.max_value}"
                )
                now = time.time()
                if dedup_key in self._dedup_windows:
                    if now - self._dedup_windows[dedup_key] < self._dedup_window_seconds:
                        logger.debug("Deduplicated alert for %s", dedup_key)
                        continue

                self._alert_counter += 1
                violated = (
                    threshold.min_value
                    if threshold.min_value is not None
                    else threshold.max_value or 0.0
                )
                alert = Alert(
                    id=f"alert-{self._alert_counter}",
                    timestamp=timestamp,
                    severity=threshold.severity,
                    metric=metric,
                    message=(
                        f"{metric} is {value:.2f} "
                        f"(threshold: {threshold.min_value}-{threshold.max_value})"
                    ),
                    value=value,
                    threshold=violated,
                    tenant_id=effective_tenant,
                )
                self._alerts.append(alert)
                alerts.append(alert)
                self._dedup_windows[dedup_key] = now

                # Audit logging
                if self._audit_logger:
                    from src.decision_support.audit import AuditEventType

                    self._audit_logger.log(
                        event_type=AuditEventType.ALERT_STATE_CHANGE,
                        action="created",
                        resource=f"alert/{alert.id}",
                        tenant_id=effective_tenant,
                        metadata={"metric": metric, "value": value, "threshold": violated},
                    )

                if self._bus is not None:
                    event = DomainEvent(
                        event_type=EventType.ALERT_TRIGGERED,
                        source="decision_support.alerts",
                        payload={
                            "alert_id": alert.id,
                            "severity": alert.severity.value,
                            "metric": alert.metric,
                            "message": alert.message,
                            "value": alert.value,
                            "threshold": alert.threshold,
                        },
                    )
                    self._bus.publish(event)

        return alerts

    def check_state(self, state: FarmState, tenant_id: Optional[str] = None) -> List[Alert]:
        """Check all metrics in a farm state against thresholds."""
        alerts = []
        alerts.extend(self.check_value("soil_moisture", state.soil_moisture, tenant_id=tenant_id))
        alerts.extend(self.check_value("temperature", state.temperature, tenant_id=tenant_id))
        alerts.extend(self.check_value("crop_height", state.crop_height, tenant_id=tenant_id))
        alerts.extend(self.check_value("nutrient_level", state.nutrient_level, tenant_id=tenant_id))
        alerts.extend(self.check_value("pest_pressure", state.pest_pressure, tenant_id=tenant_id))
        return alerts

    def publish_to(self, bus: EventBus) -> None:
        """Enable publishing of alerts to the event bus.

        After calling this, any alerts created by check_value or check_state
        are automatically published as ALERT_TRIGGERED events.

        Args:
            bus: The EventBus instance to publish alerts to.
        """
        self._bus = bus

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Alert]:
        """Get all active (non-resolved) alerts."""
        alerts = [a for a in self._alerts if not a.resolved]
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        if tenant_id is not None:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        return alerts

    def get_all_alerts(self, tenant_id: Optional[str] = None) -> List[Alert]:
        """Get all alerts including resolved ones."""
        if tenant_id is not None:
            return [a for a in self._alerts if a.tenant_id == tenant_id]
        return list(self._alerts)

    def acknowledge_alert(self, alert_id: str, actor: str = "system") -> bool:
        """Acknowledge an alert. Returns True if found."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = actor
                alert.acknowledged_at = time.time()
                if self._audit_logger:
                    from src.decision_support.audit import AuditEventType

                    self._audit_logger.log(
                        event_type=AuditEventType.ALERT_STATE_CHANGE,
                        action="acknowledged",
                        resource=f"alert/{alert_id}",
                        tenant_id=alert.tenant_id,
                        actor=actor,
                    )
                return True
        return False

    def resolve_alert(self, alert_id: str, actor: str = "system") -> bool:
        """Resolve an alert. Returns True if found."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_by = actor
                alert.resolved_at = time.time()
                if self._audit_logger:
                    from src.decision_support.audit import AuditEventType

                    self._audit_logger.log(
                        event_type=AuditEventType.ALERT_STATE_CHANGE,
                        action="resolved",
                        resource=f"alert/{alert_id}",
                        tenant_id=alert.tenant_id,
                        actor=actor,
                    )
                return True
        return False

    def clear_resolved(self) -> int:
        """Remove all resolved alerts. Returns count removed."""
        original_len = len(self._alerts)
        self._alerts = [a for a in self._alerts if not a.resolved]
        return original_len - len(self._alerts)

    def add_handler(self, handler: Callable) -> None:
        """Add a notification handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable) -> None:
        """Remove a notification handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def notify(self, alert: Alert) -> None:
        """Send notifications for an alert through all matching channels."""
        for channel in self._channels:
            if channel.tenant_id and alert.tenant_id and channel.tenant_id != alert.tenant_id:
                continue
            if self._severity_rank(alert.severity) >= self._severity_rank(channel.min_severity):
                logger.info(f"Notification via {channel.name}: {alert.message}")

        for handler in self._handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    def _severity_rank(self, severity: AlertSeverity) -> int:
        """Get numeric rank for severity comparison."""
        ranks = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
        }
        return ranks.get(severity, 0)

"""Alert management system for farm monitoring."""
from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.decision_support.recommender import FarmState

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
        }


@dataclass
class Threshold:
    """Threshold configuration for a metric."""
    metric: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    severity: AlertSeverity = AlertSeverity.WARNING


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    channel_type: str
    target: str
    min_severity: AlertSeverity = AlertSeverity.INFO


class AlertManager:
    """Manages alerts based on configurable thresholds."""

    def __init__(self):
        self._thresholds: List[Threshold] = []
        self._channels: List[NotificationChannel] = []
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        self._handlers: List[Callable] = []

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
        self._thresholds.append(threshold)

    def remove_threshold(self, metric: str) -> bool:
        """Remove all thresholds for a metric. Returns True if any removed."""
        original_len = len(self._thresholds)
        self._thresholds = [t for t in self._thresholds if t.metric != metric]
        return len(self._thresholds) < original_len

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)

    def remove_channel(self, name: str) -> bool:
        """Remove a notification channel by name."""
        original_len = len(self._channels)
        self._channels = [c for c in self._channels if c.name != name]
        return len(self._channels) < original_len

    def check_value(
        self, metric: str, value: float, timestamp: Optional[float] = None
    ) -> List[Alert]:
        """Check a value against thresholds and create alerts."""
        if timestamp is None:
            timestamp = time.time()

        alerts = []
        for threshold in self._thresholds:
            if threshold.metric != metric:
                continue

            triggered = False
            if threshold.min_value is not None and value < threshold.min_value:
                triggered = True
            if threshold.max_value is not None and value > threshold.max_value:
                triggered = True

            if triggered:
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
                )
                self._alerts.append(alert)
                alerts.append(alert)

        return alerts

    def check_state(self, state: FarmState) -> List[Alert]:
        """Check all metrics in a farm state against thresholds."""
        alerts = []
        alerts.extend(self.check_value("soil_moisture", state.soil_moisture))
        alerts.extend(self.check_value("temperature", state.temperature))
        alerts.extend(self.check_value("crop_height", state.crop_height))
        alerts.extend(self.check_value("nutrient_level", state.nutrient_level))
        alerts.extend(self.check_value("pest_pressure", state.pest_pressure))
        return alerts

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get all active (non-resolved) alerts."""
        alerts = [a for a in self._alerts if not a.resolved]
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts including resolved ones."""
        return list(self._alerts)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert. Returns True if found."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert. Returns True if found."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
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
            if self._severity_rank(alert.severity) >= self._severity_rank(
                channel.min_severity
            ):
                logger.info(
                    f"Notification via {channel.name}: {alert.message}"
                )

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

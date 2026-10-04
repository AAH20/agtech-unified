"""Test alert management system for farm monitoring."""
import asyncio
import time

import pytest

from src.decision_support.alerts import (
    AlertManager,
    AlertSeverity,
    NotificationChannel,
    Threshold,
)
from src.decision_support.recommender import FarmState


def test_alert_manager_initial_state():
    """AlertManager starts with no alerts."""
    manager = AlertManager()
    assert manager.alert_count == 0
    assert manager.active_alert_count == 0


def test_alert_manager_add_threshold():
    """Thresholds can be added and removed."""
    manager = AlertManager()
    threshold = Threshold(
        metric="soil_moisture",
        min_value=0.2,
        max_value=0.8,
        severity=AlertSeverity.WARNING,
    )
    manager.add_threshold(threshold)
    assert len(manager._thresholds) == 1

    removed = manager.remove_threshold("soil_moisture")
    assert removed is True
    assert len(manager._thresholds) == 0


def test_alert_manager_check_value_triggers():
    """Check value creates alerts when threshold is violated."""
    manager = AlertManager()
    manager.add_threshold(
        Threshold(
            metric="soil_moisture",
            min_value=0.2,
            max_value=0.8,
            severity=AlertSeverity.WARNING,
        )
    )
    alerts = manager.check_value("soil_moisture", 0.1)
    assert len(alerts) == 1
    assert alerts[0].metric == "soil_moisture"
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[0].value == 0.1


def test_alert_manager_check_value_no_trigger():
    """Check value does not create alerts when within thresholds."""
    manager = AlertManager()
    manager.add_threshold(
        Threshold(
            metric="soil_moisture",
            min_value=0.2,
            max_value=0.8,
        )
    )
    alerts = manager.check_value("soil_moisture", 0.5)
    assert len(alerts) == 0


def test_alert_manager_check_state():
    """Check state evaluates all metrics."""
    manager = AlertManager()
    manager.add_threshold(
        Threshold(metric="soil_moisture", min_value=0.3)
    )
    manager.add_threshold(
        Threshold(metric="temperature", max_value=35.0)
    )
    state = FarmState(
        soil_moisture=0.1,
        temperature=40.0,
        crop_height=0.5,
        nutrient_level=0.5,
        pest_pressure=0.1,
    )
    alerts = manager.check_state(state)
    assert len(alerts) == 2
    metrics = {a.metric for a in alerts}
    assert "soil_moisture" in metrics
    assert "temperature" in metrics


def test_alert_manager_acknowledge():
    """Alerts can be acknowledged."""
    manager = AlertManager()
    manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))
    alerts = manager.check_value("soil_moisture", 0.1)
    alert_id = alerts[0].id

    result = manager.acknowledge_alert(alert_id)
    assert result is True
    assert alerts[0].acknowledged is True


def test_alert_manager_resolve():
    """Alerts can be resolved."""
    manager = AlertManager()
    manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))
    alerts = manager.check_value("soil_moisture", 0.1)
    alert_id = alerts[0].id

    result = manager.resolve_alert(alert_id)
    assert result is True
    assert alerts[0].resolved is True
    assert manager.active_alert_count == 0


def test_alert_manager_clear_resolved():
    """Resolved alerts can be cleared."""
    manager = AlertManager()
    manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))
    alerts = manager.check_value("soil_moisture", 0.1)
    manager.resolve_alert(alerts[0].id)

    removed = manager.clear_resolved()
    assert removed == 1
    assert manager.alert_count == 0


def test_alert_manager_notification_channels():
    """Notification channels can be added and removed."""
    manager = AlertManager()
    channel = NotificationChannel(
        name="email",
        channel_type="email",
        target="admin@farm.com",
        min_severity=AlertSeverity.WARNING,
    )
    manager.add_channel(channel)
    assert len(manager._channels) == 1

    removed = manager.remove_channel("email")
    assert removed is True
    assert len(manager._channels) == 0


def test_alert_manager_notify():
    """Notify sends alerts to matching channels."""
    manager = AlertManager()
    manager.add_channel(
        NotificationChannel(
            name="sms",
            channel_type="sms",
            target="+1234567890",
            min_severity=AlertSeverity.CRITICAL,
        )
    )
    manager.add_threshold(
        Threshold(
            metric="soil_moisture",
            min_value=0.3,
            severity=AlertSeverity.CRITICAL,
        )
    )
    alerts = manager.check_value("soil_moisture", 0.1)

    # Should not raise
    asyncio.run(manager.notify(alerts[0]))


def test_alert_manager_severity_filtering():
    """Active alerts can be filtered by severity."""
    manager = AlertManager()
    manager.add_threshold(
        Threshold(
            metric="soil_moisture",
            min_value=0.3,
            severity=AlertSeverity.CRITICAL,
        )
    )
    manager.add_threshold(
        Threshold(
            metric="temperature",
            max_value=35.0,
            severity=AlertSeverity.WARNING,
        )
    )
    manager.check_value("soil_moisture", 0.1)
    manager.check_value("temperature", 40.0)

    critical = manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
    assert len(critical) == 1
    assert critical[0].metric == "soil_moisture"

    warning = manager.get_active_alerts(severity=AlertSeverity.WARNING)
    assert len(warning) == 1
    assert warning[0].metric == "temperature"


def test_alert_manager_custom_handler():
    """Custom handlers are called on notify."""
    manager = AlertManager()
    handler_calls = []

    def custom_handler(alert):
        handler_calls.append(alert.id)

    manager.add_handler(custom_handler)
    manager.add_threshold(
        Threshold(metric="soil_moisture", min_value=0.3)
    )
    alerts = manager.check_value("soil_moisture", 0.1)
    asyncio.run(manager.notify(alerts[0]))

    assert len(handler_calls) == 1
    assert handler_calls[0] == alerts[0].id

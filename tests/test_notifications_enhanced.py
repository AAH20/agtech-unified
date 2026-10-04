"""Tests for notification templates, priority routing, escalation, and quiet hours."""

import time

from src.decision_support.notifications import (
    DigestNotifier,
    NotificationChannel,
    NotificationManager,
    NotificationStatus,
    NotificationTemplate,
    PriorityRouter,
    QuietHoursScheduler,
)


class RecorderChannel(NotificationChannel):
    """Test channel that records sent messages."""

    channel_type = "recorder"

    def __init__(self, name, recipient="test"):
        super().__init__(name=name, recipient=recipient)
        self.sent = []

    def _deliver(self, subject, message, **kwargs):
        self.sent.append({"subject": subject, "message": message, **kwargs})
        return f"id-{len(self.sent)}"


class TestNotificationTemplate:
    def test_render_simple_template(self):
        """Template renders with variable substitution."""
        template = NotificationTemplate(
            name="alert",
            subject="Alert: {{metric}}",
            body="Value {{value}} exceeds threshold {{threshold}}",
        )
        result = template.render(metric="soil_moisture", value=0.1, threshold=0.3)
        assert result["subject"] == "Alert: soil_moisture"
        assert "Value 0.1 exceeds threshold 0.3" in result["body"]

    def test_render_with_default_values(self):
        """Template uses default values for missing variables."""
        template = NotificationTemplate(
            name="alert",
            subject="Alert",
            body="Value: {{value|default('N/A')}}",
        )
        result = template.render()
        assert "Value: N/A" in result["body"]

    def test_template_with_conditionals(self):
        """Template supports conditional blocks."""
        template = NotificationTemplate(
            name="alert",
            subject="Alert",
            body="{% if severity == 'critical' %}URGENT: {{message}}{% else %}{{message}}{% endif %}",
        )
        result = template.render(severity="critical", message="Test")
        assert "URGENT: Test" in result["body"]

    def test_template_localization(self):
        """Template supports multiple languages."""
        template = NotificationTemplate(
            name="alert",
            subject="Alert",
            body="Value: {{value}}",
            translations={"es": {"body": "Valor: {{value}}"}},
        )
        result = template.render(value=42, lang="es")
        assert "Valor: 42" in result["body"]


class TestPriorityRouter:
    def test_critical_routes_to_all_channels(self):
        """Critical notifications go to all channels."""
        router = PriorityRouter()
        critical = RecorderChannel("critical-chan")
        normal = RecorderChannel("normal-chan")
        router.add_channel(critical, priority="critical")
        router.add_channel(normal, priority="normal")

        router.route("Test", "Message", priority="critical")
        assert len(critical.sent) == 1
        assert len(normal.sent) == 1

    def test_normal_routes_to_normal_only(self):
        """Normal notifications only go to normal channels."""
        router = PriorityRouter()
        critical = RecorderChannel("critical-chan")
        normal = RecorderChannel("normal-chan")
        router.add_channel(critical, priority="critical")
        router.add_channel(normal, priority="normal")

        router.route("Test", "Message", priority="normal")
        assert len(critical.sent) == 0
        assert len(normal.sent) == 1

    def test_escalation_on_unacknowledged(self):
        """Unacknowledged critical alerts escalate after timeout."""
        router = PriorityRouter()
        primary = RecorderChannel("primary")
        manager = RecorderChannel("manager")
        router.add_channel(primary, priority="critical")
        router.add_channel(manager, priority="escalation")

        router.route("Test", "Message", priority="critical", escalate_after=0.1)
        assert len(primary.sent) == 1
        assert len(manager.sent) == 0

        # Wait for escalation
        time.sleep(0.15)
        router.process_escalations()
        assert len(manager.sent) == 1


class TestQuietHoursScheduler:
    def test_quiet_hours_blocks_non_critical(self):
        """Non-critical notifications are blocked during quiet hours."""
        scheduler = QuietHoursScheduler(quiet_start=22, quiet_end=6)
        # Simulate 23:00 (within quiet hours)
        assert scheduler.should_send("warning", current_hour=23) is False
        assert scheduler.should_send("info", current_hour=23) is False

    def test_critical_bypasses_quiet_hours(self):
        """Critical notifications bypass quiet hours."""
        scheduler = QuietHoursScheduler(quiet_start=22, quiet_end=6)
        assert scheduler.should_send("critical", current_hour=23) is True

    def test_normal_hours_allow_all(self):
        """All notifications allowed during normal hours."""
        scheduler = QuietHoursScheduler(quiet_start=22, quiet_end=6)
        assert scheduler.should_send("info", current_hour=10) is True
        assert scheduler.should_send("warning", current_hour=10) is True
        assert scheduler.should_send("critical", current_hour=10) is True


class TestDigestNotifier:
    def test_batches_notifications(self):
        """Digest notifier batches multiple notifications."""
        digest = DigestNotifier(interval_seconds=0.1)
        channel = RecorderChannel("digest-chan")
        digest.add_channel(channel)

        digest.add("Alert 1", "Message 1")
        digest.add("Alert 2", "Message 2")
        assert len(channel.sent) == 0  # Not yet flushed

        time.sleep(0.15)
        digest.flush()
        assert len(channel.sent) == 1
        assert "Alert 1" in channel.sent[0]["message"]
        assert "Alert 2" in channel.sent[0]["message"]

    def test_immediate_flush_on_critical(self):
        """Critical notifications are flushed immediately."""
        digest = DigestNotifier(interval_seconds=60)
        channel = RecorderChannel("digest-chan")
        digest.add_channel(channel)

        digest.add("Critical", "Urgent!", priority="critical")
        assert len(channel.sent) == 1


class TestNotificationManagerEnhanced:
    def test_send_with_template(self):
        """NotificationManager can use templates."""
        manager = NotificationManager()
        channel = RecorderChannel("test")
        manager.add_channel(channel)

        template = NotificationTemplate(
            name="test",
            subject="Alert: {{metric}}",
            body="Value {{value}}",
        )
        manager.send_template(template, metric="soil_moisture", value=0.1)
        assert len(channel.sent) == 1
        assert channel.sent[0]["subject"] == "Alert: soil_moisture"

    def test_send_with_priority(self):
        """NotificationManager supports priority-based routing."""
        manager = NotificationManager()
        critical_chan = RecorderChannel("critical", recipient="crit@example.com")
        normal_chan = RecorderChannel("normal", recipient="norm@example.com")
        manager.add_channel(critical_chan)
        manager.add_channel(normal_chan)

        manager.send("Test", "Critical alert", priority="critical")
        # Both channels receive the notification (priority is metadata, not a filter)
        assert len(critical_chan.sent) == 1
        assert len(normal_chan.sent) == 1

    def test_delivery_status_tracking(self):
        """NotificationManager tracks delivery status."""
        manager = NotificationManager()
        channel = RecorderChannel("test")
        manager.add_channel(channel)

        results = manager.send("Test", "Message")
        assert len(results) == 1
        assert results[0].status == NotificationStatus.SENT
        assert results[0].delivered is True

    def test_failed_delivery_tracked(self):
        """Failed deliveries are tracked."""
        manager = NotificationManager()

        class FailingChannel(NotificationChannel):
            channel_type = "failing"

            def _deliver(self, subject, message, **kwargs):
                raise ConnectionError("down")

        channel = FailingChannel(name="failing", recipient="x", max_retries=1, retry_delay=0)
        manager.add_channel(channel)

        results = manager.send("Test", "Message")
        assert len(results) == 1
        assert results[0].status == NotificationStatus.FAILED
        assert results[0].delivered is False

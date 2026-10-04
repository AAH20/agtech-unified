"""Tests for notification channel implementations."""

import json

import pytest

from src.decision_support.notifications import (
    EmailNotification,
    NotificationChannel,
    NotificationManager,
    NotificationStatus,
    SMSNotification,
    WebhookNotification,
)


class FakeSMTP:
    """Minimal SMTP stand-in recording calls."""

    instances = []

    def __init__(self, host, port, timeout=0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_called = False
        self.login_called = False
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        self.starttls_called = True

    def login(self, username, password):
        self.login_called = True

    def send_message(self, msg):
        self.sent.append(msg)


@pytest.fixture(autouse=True)
def _reset_fake_smtp():
    FakeSMTP.instances = []
    yield
    FakeSMTP.instances = []


@pytest.fixture
def smtp(monkeypatch):
    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    return FakeSMTP


class TestEmailNotification:
    def test_delivers_email_over_smtp(self, smtp):
        """Email is sent via SMTP and result reports SENT."""
        channel = EmailNotification(
            name="email-ops",
            recipient="ops@farm.example",
            smtp_host="smtp.farm.example",
            smtp_port=587,
            username="agtech",
            password="secret",
        )
        result = channel.send("Alert", "Soil moisture critical")
        assert result.delivered
        assert result.status == NotificationStatus.SENT
        assert result.channel == "email"
        assert result.recipient == "ops@farm.example"
        assert result.attempts == 1
        assert result.message_id is not None

    def test_uses_tls_and_login_when_configured(self, smtp):
        """STARTTLS and login are invoked with credentials."""
        channel = EmailNotification(
            name="email-ops",
            recipient="ops@farm.example",
            smtp_host="smtp.farm.example",
            smtp_port=587,
            username="agtech",
            password="secret",
            use_tls=True,
        )
        channel.send("Alert", "message")
        server = FakeSMTP.instances[0]
        assert server.starttls_called
        assert server.login_called
        assert server.sent[0]["To"] == "ops@farm.example"
        assert server.sent[0]["Subject"] == "Alert"

    def test_plaintext_body_is_delivered(self, smtp):
        """The message body reaches the MIME message."""
        channel = EmailNotification(name="email-ops", recipient="ops@farm.example", use_tls=False)
        channel.send("Subject here", "Body text")
        server = FakeSMTP.instances[0]
        assert not server.starttls_called
        assert "Body text" in server.sent[0].get_payload()


class TestSMSNotification:
    def test_delivers_sms_via_http(self):
        """SMS is POSTed to the Twilio-compatible API."""
        calls = []

        def fake_post(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {"sid": "SM123"}

        channel = SMSNotification(
            name="sms-ops",
            recipient="+15551234567",
            account_sid="ACxx",
            from_number="+15550000000",
            http_post=fake_post,
        )
        result = channel.send("Alert", "Pest pressure high")
        assert result.delivered
        assert result.message_id == "SM123"
        url, payload, _ = calls[0]
        assert "ACxx" in url
        assert payload["To"] == "+15551234567"
        assert payload["From"] == "+15550000000"
        assert "Pest pressure high" in payload["Body"]

    def test_sms_body_includes_subject_prefix(self):
        """Subject is prefixed into the SMS body."""
        captured = {}

        def fake_post(url, payload, timeout):
            captured.update(payload)
            return {"sid": "SM1"}

        channel = SMSNotification(name="sms-ops", recipient="+15551234567", http_post=fake_post)
        channel.send("URGENT", "Irrigate now")
        assert captured["Body"].startswith("URGENT")
        assert "Irrigate now" in captured["Body"]


class TestWebhookNotification:
    def test_posts_json_payload(self):
        """Webhook POSTs a JSON payload to the recipient URL."""
        calls = []

        def fake_post(url, payload, timeout, headers):
            calls.append((url, payload, timeout, headers))

        channel = WebhookNotification(
            name="hook-ops",
            recipient="https://hooks.farm.example/alerts",
            http_post=fake_post,
        )
        result = channel.send("Alert", "Frost risk", severity="critical")
        assert result.delivered
        url, payload, _, headers = calls[0]
        assert url == "https://hooks.farm.example/alerts"
        assert headers["Content-Type"] == "application/json"
        data = json.loads(payload)
        assert data["subject"] == "Alert"
        assert data["message"] == "Frost risk"
        assert data["severity"] == "critical"
        assert data["channel"] == "webhook"

    def test_webhook_generates_message_id(self):
        """A webhook delivery returns a non-empty message id."""
        channel = WebhookNotification(
            name="hook-ops",
            recipient="https://hooks.farm.example/alerts",
            http_post=lambda url, payload, timeout, headers: None,
        )
        result = channel.send("S", "M")
        assert result.delivered
        assert result.message_id


class TestRetryBehavior:
    def test_retries_then_succeeds(self):
        """A failing transport retries and eventually succeeds."""
        attempts = []

        class FlakyChannel(NotificationChannel):
            channel_type = "flaky"

            def _deliver(self, subject, message, **kwargs):
                attempts.append(1)
                if len(attempts) < 3:
                    raise ConnectionError("transient")
                return "ok"

        channel = FlakyChannel(name="flaky", recipient="x", retry_delay=0.0)
        result = channel.send("S", "M")
        assert result.delivered
        assert result.attempts == 3

    def test_retries_exhausted_marks_failed(self):
        """All attempts failing yields FAILED with the last error."""

        class DeadChannel(NotificationChannel):
            channel_type = "dead"

            def _deliver(self, subject, message, **kwargs):
                raise ConnectionError("down")

        channel = DeadChannel(name="dead", recipient="x", max_retries=2, retry_delay=0.0)
        result = channel.send("S", "M")
        assert not result.delivered
        assert result.status == NotificationStatus.FAILED
        assert result.error is not None
        assert "down" in result.error
        assert result.attempts == 2


class TestNotificationManager:
    def test_dispatches_to_all_channels(self):
        """Manager sends to every registered channel."""
        results_a = []
        results_b = []

        class Recorder(NotificationChannel):
            def __init__(self, name, store):
                super().__init__(name=name, recipient=name)
                self._store = store

            def _deliver(self, subject, message, **kwargs):
                self._store.append(subject)
                return self.name

        manager = NotificationManager()
        manager.add_channel(Recorder("a", results_a))
        manager.add_channel(Recorder("b", results_b))
        results = manager.send("Hello", "World")
        assert len(results) == 2
        assert results_a == ["Hello"]
        assert results_b == ["Hello"]

    def test_deduplicates_identical_notifications(self):
        """Identical notifications within the dedup window are sent once."""
        sent = []

        class Recorder(NotificationChannel):
            channel_type = "rec"

            def _deliver(self, subject, message, **kwargs):
                sent.append(message)
                return "id"

        manager = NotificationManager(dedup_window=300.0)
        manager.add_channel(Recorder(name="a", recipient="a"))
        manager.send("Same", "Same body")
        manager.send("Same", "Same body")
        assert sent == ["Same body"]
        assert len(manager.history) == 1

    def test_history_and_clear(self):
        """History records deliveries and clear resets state."""

        class Recorder(NotificationChannel):
            channel_type = "rec"

            def _deliver(self, subject, message, **kwargs):
                return "id"

        manager = NotificationManager()
        manager.add_channel(Recorder(name="a", recipient="a"))
        manager.send("S", "M")
        assert len(manager.history) == 1
        manager.clear_history()
        assert manager.history == []

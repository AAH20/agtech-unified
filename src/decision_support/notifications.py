"""Notification channel implementations for farm alerting."""

from __future__ import annotations

import json
import logging
import smtplib
import time
import urllib.request
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationStatus(Enum):
    """Delivery status of a notification."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class NotificationResult:
    """Result of a notification delivery attempt."""

    status: NotificationStatus
    channel: str
    recipient: str
    message_id: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def delivered(self) -> bool:
        """Whether the notification was delivered successfully."""
        return self.status == NotificationStatus.SENT

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "channel": self.channel,
            "recipient": self.recipient,
            "message_id": self.message_id,
            "error": self.error,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


class NotificationChannel:
    """Base class for notification channels with retry support."""

    channel_type = "base"

    def __init__(
        self,
        name: str,
        recipient: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 10.0,
    ):
        self.name = name
        self.recipient = recipient
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def send(self, subject: str, message: str, **kwargs: Any) -> NotificationResult:
        """Send a notification with exponential-backoff retries."""
        last_error: Optional[str] = None
        start = time.monotonic()
        for attempt in range(1, self.max_retries + 1):
            try:
                message_id = self._deliver(subject, message, **kwargs)
                return NotificationResult(
                    status=NotificationStatus.SENT,
                    channel=self.channel_type,
                    recipient=self.recipient,
                    message_id=message_id,
                    attempts=attempt,
                    latency_ms=(time.monotonic() - start) * 1000.0,
                )
            except Exception as exc:  # noqa: BLE001 - channels raise varied errors
                last_error = str(exc)
                logger.warning(
                    "Notification attempt %d/%d failed on %s: %s",
                    attempt,
                    self.max_retries,
                    self.name,
                    last_error,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))
        return NotificationResult(
            status=NotificationStatus.FAILED,
            channel=self.channel_type,
            recipient=self.recipient,
            error=last_error,
            attempts=self.max_retries,
            latency_ms=(time.monotonic() - start) * 1000.0,
        )

    def _deliver(self, subject: str, message: str, **kwargs: Any) -> str:
        """Actually deliver the notification. Returns a message id."""
        raise NotImplementedError


class EmailNotification(NotificationChannel):
    """Email notification channel over SMTP."""

    channel_type = "email"

    def __init__(
        self,
        name: str,
        recipient: str,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        sender: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, recipient=recipient, **kwargs)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.sender = sender or (username if username else "agtech@localhost")

    def _deliver(self, subject: str, message: str, **kwargs: Any) -> str:
        """Send an email via SMTP and return a generated message id."""
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = self.recipient
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
            if self.use_tls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(msg)
        return f"email-{int(time.time() * 1000)}"


class SMSNotification(NotificationChannel):
    """SMS notification channel (Twilio-compatible HTTP API)."""

    channel_type = "sms"

    def __init__(
        self,
        name: str,
        recipient: str,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
        api_url: str = "https://api.twilio.com/2010-04-01",
        http_post=None,
        **kwargs: Any,
    ):
        super().__init__(name=name, recipient=recipient, **kwargs)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.api_url = api_url
        # Injectable transport for testing; defaults to urllib POST.
        self._http_post = http_post

    def _deliver(self, subject: str, message: str, **kwargs: Any) -> str:
        """Send an SMS via the Twilio-compatible API and return the SID."""
        body = f"{subject}: {message}" if subject else message
        url = f"{self.api_url}/Accounts/{self.account_sid}/Messages.json"
        payload = {
            "To": self.recipient,
            "From": self.from_number,
            "Body": body,
        }
        if self._http_post is not None:
            response = self._http_post(url, payload, self.timeout)
        else:
            data = "&".join(f"{k}={v}" for k, v in payload.items()).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                response = json.loads(resp.read().decode("utf-8"))
        return response.get("sid", f"sms-{int(time.time() * 1000)}")


class WebhookNotification(NotificationChannel):
    """Webhook notification channel (HTTP POST with JSON payload)."""

    channel_type = "webhook"

    def __init__(
        self,
        name: str,
        recipient: str,
        headers: Optional[Dict[str, str]] = None,
        http_post=None,
        **kwargs: Any,
    ):
        super().__init__(name=name, recipient=recipient, **kwargs)
        self.headers = headers or {"Content-Type": "application/json"}
        # Injectable transport for testing; defaults to urllib POST.
        self._http_post = http_post

    def _deliver(self, subject: str, message: str, **kwargs: Any) -> str:
        """POST a JSON payload to the webhook URL and return a message id."""
        payload = json.dumps(
            {
                "subject": subject,
                "message": message,
                "recipient": self.recipient,
                "channel": self.channel_type,
                "timestamp": time.time(),
                **kwargs,
            }
        ).encode("utf-8")
        if self._http_post is not None:
            self._http_post(self.recipient, payload, self.timeout, self.headers)
        else:
            req = urllib.request.Request(self.recipient, data=payload, method="POST")
            for key, value in self.headers.items():
                req.add_header(key, value)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        return f"webhook-{int(time.time() * 1000)}"


class NotificationManager:
    """Dispatches notifications to multiple channels with deduplication."""

    def __init__(self, dedup_window: float = 300.0):
        self._channels: List[NotificationChannel] = []
        self._history: List[NotificationResult] = []
        self._dedup_window = dedup_window
        self._recent_keys: Dict[str, float] = {}

    def add_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self._channels.append(channel)

    @property
    def channels(self) -> List[NotificationChannel]:
        """Registered channels."""
        return list(self._channels)

    def _dedup_key(self, channel: NotificationChannel, subject: str, message: str) -> str:
        return f"{channel.channel_type}:{channel.recipient}:{subject}:{message}"

    def send(
        self,
        subject: str,
        message: str,
        channels: Optional[List[NotificationChannel]] = None,
        deduplicate: bool = True,
        **kwargs: Any,
    ) -> List[NotificationResult]:
        """Send a notification to all (or selected) channels."""
        targets = channels if channels is not None else self._channels
        results: List[NotificationResult] = []
        now = time.time()
        for channel in targets:
            key = self._dedup_key(channel, subject, message)
            if deduplicate and key in self._recent_keys:
                if now - self._recent_keys[key] < self._dedup_window:
                    logger.info("Deduplicated notification on %s", channel.name)
                    continue
            result = channel.send(subject, message, **kwargs)
            self._recent_keys[key] = now
            self._history.append(result)
            results.append(result)
        return results

    @property
    def history(self) -> List[NotificationResult]:
        """All notification results sent through this manager."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear delivery history and dedup state."""
        self._history.clear()
        self._recent_keys.clear()

"""MQTT QoS 1/2 support: at-least-once and exactly-once delivery.

Provides MQTTQoSClient with full QoS 0/1/2 support:
- QoS 0: At most once (fire-and-forget)
- QoS 1: At least once (PUBLISH → PUBACK)
- QoS 2: Exactly once (PUBLISH → PUBREC → PUBREL → PUBCOMP)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Set, Union


class QoSLevel(IntEnum):
    """MQTT Quality of Service levels."""

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


@dataclass
class MQTTMessage:
    """MQTT message with QoS metadata."""

    topic: str
    payload: bytes
    qos: int = 0
    message_id: Optional[int] = None
    retain: bool = False

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("topic cannot be empty")
        if self.qos not in (0, 1, 2):
            raise ValueError("qos must be 0, 1, or 2")


@dataclass
class DeliveryState:
    """Tracks the delivery state of a message."""

    message_id: int
    topic: str
    qos: int
    state: str = "pending"  # pending, published, acknowledged, completed, failed

    def __post_init__(self) -> None:
        valid_states = {"pending", "published", "acknowledged", "completed", "failed"}
        if self.state not in valid_states:
            raise ValueError(f"Invalid delivery state: {self.state}")


class MQTTQoSClient:
    """MQTT client with QoS 0/1/2 support.

    Works in-memory; can be extended with paho-mqtt for real brokers.
    Tracks delivery state for QoS 1 and 2 messages.
    """

    def __init__(self, host: str = "localhost", port: int = 1883, client_id: str = ""):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._connected = False
        self._message_id_counter = 0
        self._delivery_states: Dict[int, DeliveryState] = {}
        self._acked_ids: Set[int] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Disconnect from MQTT broker."""
        if not self._connected:
            return False
        self._connected = False
        self._delivery_states.clear()
        return True

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(
        self,
        topic: str,
        payload: Union[str, bytes, dict],
        qos: int = 0,
        retain: bool = False,
    ) -> MQTTMessage:
        """Publish a message to a topic.

        Args:
            topic: MQTT topic string
            payload: Message payload (str, bytes, or dict)
            qos: Quality of service level (0, 1, or 2)
            retain: Whether to retain the message

        Returns:
            The published MQTTMessage
        """
        if not self._connected:
            raise RuntimeError("Client is not connected")

        if isinstance(payload, dict):
            payload = json.dumps(payload)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        message_id: Optional[int] = None
        if qos >= 1:
            with self._lock:
                self._message_id_counter += 1
                message_id = self._message_id_counter
            assert message_id is not None

        msg = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            message_id=message_id,
            retain=retain,
        )

        if qos >= 1:
            with self._lock:
                self._delivery_states[message_id] = DeliveryState(
                    message_id=message_id,
                    topic=topic,
                    qos=qos,
                    state="pending",
                )

        return msg

    # ------------------------------------------------------------------
    # QoS 1: At Least Once
    # ------------------------------------------------------------------

    def acknowledge(self, message_id: int) -> bool:
        """Acknowledge receipt of a QoS 1 message (PUBACK).

        Args:
            message_id: The message ID to acknowledge

        Returns:
            True if the message was pending and is now acknowledged
        """
        with self._lock:
            state = self._delivery_states.get(message_id)
            if state is None or state.state != "pending":
                return False
            state.state = "completed"
            self._acked_ids.add(message_id)
            return True

    # ------------------------------------------------------------------
    # QoS 2: Exactly Once
    # ------------------------------------------------------------------

    def handle_pubrec(self, message_id: int) -> bool:
        """Handle PUBREC for QoS 2 (step 2 of 4).

        Transitions message from 'pending' to 'published'.
        """
        with self._lock:
            state = self._delivery_states.get(message_id)
            if state is None or state.state != "pending":
                return False
            state.state = "published"
            return True

    def handle_pubrel(self, message_id: int) -> bool:
        """Handle PUBREL for QoS 2 (step 3 of 4).

        Transitions message from 'published' to 'acknowledged'.
        """
        with self._lock:
            state = self._delivery_states.get(message_id)
            if state is None or state.state != "published":
                return False
            state.state = "acknowledged"
            return True

    def handle_pubcomp(self, message_id: int) -> bool:
        """Handle PUBCOMP for QoS 2 (step 4 of 4).

        Transitions message from 'acknowledged' to 'completed'.
        """
        with self._lock:
            state = self._delivery_states.get(message_id)
            if state is None or state.state != "acknowledged":
                return False
            state.state = "completed"
            self._acked_ids.add(message_id)
            return True

    # ------------------------------------------------------------------
    # Delivery State Tracking
    # ------------------------------------------------------------------

    def get_delivery_state(self, message_id: int) -> Optional[DeliveryState]:
        """Get the delivery state for a message."""
        with self._lock:
            return self._delivery_states.get(message_id)

    def get_all_delivery_states(self) -> List[DeliveryState]:
        """Get all delivery states."""
        with self._lock:
            return list(self._delivery_states.values())

    def get_pending_messages(self) -> List[DeliveryState]:
        """Get all messages that are not yet completed."""
        with self._lock:
            return [s for s in self._delivery_states.values() if s.state != "completed"]

    def clear_delivery_states(self) -> None:
        """Clear all delivery states."""
        with self._lock:
            self._delivery_states.clear()

    # ------------------------------------------------------------------
    # Failure Handling
    # ------------------------------------------------------------------

    def mark_failed(self, message_id: int) -> bool:
        """Mark a message as failed."""
        with self._lock:
            state = self._delivery_states.get(message_id)
            if state is None:
                return False
            state.state = "failed"
            return True

    def redeliver(self, message_id: int) -> bool:
        """Redeliver a failed message."""
        with self._lock:
            state = self._delivery_states.get(message_id)
            if state is None or state.state != "failed":
                return False
            state.state = "pending"
            return True

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def is_duplicate(self, message_id: int) -> bool:
        """Check if a message ID has already been processed."""
        with self._lock:
            return message_id in self._acked_ids

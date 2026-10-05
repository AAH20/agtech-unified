"""Tests for MQTT QoS 1/2 support: at-least-once and exactly-once delivery.

TDD: these tests define the expected behavior for the new features.
"""

import pytest

from src.iot.mqtt_qos import (
    DeliveryState,
    MQTTMessage,
    MQTTQoSClient,
    QoSLevel,
)

# ===========================================================================
# QoSLevel
# ===========================================================================


class TestQoSLevel:
    """QoSLevel enum defines MQTT quality of service levels."""

    def test_qos_levels_exist(self):
        """QoS 0, 1, and 2 are defined."""
        assert QoSLevel.AT_MOST_ONCE == 0
        assert QoSLevel.AT_LEAST_ONCE == 1
        assert QoSLevel.EXACTLY_ONCE == 2

    def test_qos_level_values(self):
        """QoS levels have correct integer values."""
        assert QoSLevel.AT_MOST_ONCE.value == 0
        assert QoSLevel.AT_LEAST_ONCE.value == 1
        assert QoSLevel.EXACTLY_ONCE.value == 2


# ===========================================================================
# MQTTMessage
# ===========================================================================


class TestMQTTMessage:
    """MQTTMessage represents a message with QoS metadata."""

    def test_create_message(self):
        """Create an MQTT message."""
        msg = MQTTMessage(
            topic="sensors/temperature",
            payload=b"22.5",
            qos=1,
            message_id=42,
        )
        assert msg.topic == "sensors/temperature"
        assert msg.payload == b"22.5"
        assert msg.qos == 1
        assert msg.message_id == 42
        assert msg.retain is False

    def test_message_default_qos(self):
        """Default QoS is 0."""
        msg = MQTTMessage(topic="test", payload=b"data")
        assert msg.qos == 0
        assert msg.message_id is None

    def test_message_default_retain(self):
        """Default retain is False."""
        msg = MQTTMessage(topic="test", payload=b"data")
        assert msg.retain is False

    def test_message_with_retain(self):
        """Message can be marked as retained."""
        msg = MQTTMessage(topic="test", payload=b"data", retain=True)
        assert msg.retain is True

    def test_message_invalid_qos_raises(self):
        """QoS must be 0, 1, or 2."""
        with pytest.raises(ValueError, match="qos must be 0, 1, or 2"):
            MQTTMessage(topic="test", payload=b"data", qos=3)

    def test_message_negative_qos_raises(self):
        """Negative QoS raises ValueError."""
        with pytest.raises(ValueError, match="qos must be 0, 1, or 2"):
            MQTTMessage(topic="test", payload=b"data", qos=-1)

    def test_message_empty_topic_raises(self):
        """Empty topic raises ValueError."""
        with pytest.raises(ValueError, match="topic cannot be empty"):
            MQTTMessage(topic="", payload=b"data")

    def test_message_empty_payload_allowed(self):
        """Empty payload is allowed (e.g., for retained message deletion)."""
        msg = MQTTMessage(topic="test", payload=b"")
        assert msg.payload == b""


# ===========================================================================
# DeliveryState
# ===========================================================================


class TestDeliveryState:
    """DeliveryState tracks message delivery status."""

    def test_create_delivery_state(self):
        """Create a delivery state for a message."""
        state = DeliveryState(
            message_id=1,
            topic="test/topic",
            qos=1,
            state="pending",
        )
        assert state.message_id == 1
        assert state.topic == "test/topic"
        assert state.qos == 1
        assert state.state == "pending"

    def test_delivery_state_default_state(self):
        """Default state is 'pending'."""
        state = DeliveryState(message_id=1, topic="test", qos=0)
        assert state.state == "pending"

    def test_delivery_state_invalid_state_raises(self):
        """Invalid delivery state raises ValueError."""
        with pytest.raises(ValueError, match="Invalid delivery state"):
            DeliveryState(message_id=1, topic="test", qos=0, state="invalid")

    def test_delivery_state_valid_states(self):
        """All valid states are accepted."""
        for s in ["pending", "published", "acknowledged", "completed", "failed"]:
            state = DeliveryState(message_id=1, topic="test", qos=0, state=s)
            assert state.state == s


# ===========================================================================
# MQTTQoSClient — Connection
# ===========================================================================


class TestMQTTQoSClientConnection:
    """MQTTQoSClient connection management."""

    def test_create_client(self):
        """Create an MQTT QoS client."""
        client = MQTTQoSClient(host="localhost", port=1883, client_id="test-client")
        assert client.host == "localhost"
        assert client.port == 1883
        assert client.client_id == "test-client"

    def test_default_host_port(self):
        """Default host is localhost, port 1883."""
        client = MQTTQoSClient()
        assert client.host == "localhost"
        assert client.port == 1883

    def test_connect(self):
        """Connect to broker."""
        client = MQTTQoSClient()
        assert client.connect() is True
        assert client.is_connected is True

    def test_disconnect(self):
        """Disconnect from broker."""
        client = MQTTQoSClient()
        client.connect()
        assert client.disconnect() is True
        assert client.is_connected is False

    def test_connect_already_connected(self):
        """Connecting when already connected returns True."""
        client = MQTTQoSClient()
        client.connect()
        assert client.connect() is True

    def test_disconnect_not_connected(self):
        """Disconnecting when not connected returns False."""
        client = MQTTQoSClient()
        assert client.disconnect() is False


# ===========================================================================
# MQTTQoSClient — QoS 0 (At Most Once)
# ===========================================================================


class TestMQTTQoSClientQoS0:
    """QoS 0: fire-and-forget delivery."""

    def test_publish_qos0(self):
        """Publish a QoS 0 message."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=0)
        assert msg.qos == 0
        assert msg.topic == "test/topic"
        assert msg.payload == b"hello"

    def test_publish_qos0_no_message_id(self):
        """QoS 0 messages don't get a message ID."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=0)
        assert msg.message_id is None

    def test_publish_qos0_not_connected(self):
        """Publishing when not connected raises RuntimeError."""
        client = MQTTQoSClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client.publish("test/topic", b"hello", qos=0)

    def test_publish_qos0_no_delivery_tracking(self):
        """QoS 0 messages are not tracked for delivery."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=0)
        # QoS 0 messages should not appear in delivery tracking
        assert client.get_delivery_state(msg.message_id) is None


# ===========================================================================
# MQTTQoSClient — QoS 1 (At Least Once)
# ===========================================================================


class TestMQTTQoSClientQoS1:
    """QoS 1: at-least-once delivery with PUBACK."""

    def test_publish_qos1(self):
        """Publish a QoS 1 message."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        assert msg.qos == 1
        assert msg.message_id is not None

    def test_publish_qos1_assigns_message_id(self):
        """QoS 1 messages get a unique message ID."""
        client = MQTTQoSClient()
        client.connect()
        msg1 = client.publish("test/topic", b"hello1", qos=1)
        msg2 = client.publish("test/topic", b"hello2", qos=1)
        assert msg1.message_id is not None
        assert msg2.message_id is not None
        assert msg1.message_id != msg2.message_id

    def test_publish_qos1_not_connected(self):
        """Publishing QoS 1 when not connected raises RuntimeError."""
        client = MQTTQoSClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client.publish("test/topic", b"hello", qos=1)

    def test_qos1_message_tracked_as_pending(self):
        """QoS 1 messages are tracked as pending."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        state = client.get_delivery_state(msg.message_id)
        assert state is not None
        assert state.state == "pending"

    def test_qos1_ack_completes_delivery(self):
        """Acknowledging a QoS 1 message marks it completed."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        assert client.acknowledge(msg.message_id) is True
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "completed"

    def test_qos1_ack_nonexistent_message(self):
        """Acknowledging a nonexistent message returns False."""
        client = MQTTQoSClient()
        client.connect()
        assert client.acknowledge(99999) is False

    def test_qos1_redelivery_on_failure(self):
        """Failed QoS 1 messages can be redelivered."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        # Simulate delivery failure
        client.mark_failed(msg.message_id)
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "failed"
        # Redeliver
        assert client.redeliver(msg.message_id) is True
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "pending"

    def test_qos1_redeliver_nonexistent(self):
        """Redelivering a nonexistent message returns False."""
        client = MQTTQoSClient()
        client.connect()
        assert client.redeliver(99999) is False

    def test_qos1_get_pending_messages(self):
        """Get all pending QoS 1 messages."""
        client = MQTTQoSClient()
        client.connect()
        msg1 = client.publish("test/topic", b"hello1", qos=1)
        msg2 = client.publish("test/topic", b"hello2", qos=1)
        client.acknowledge(msg1.message_id)
        pending = client.get_pending_messages()
        assert len(pending) == 1
        assert pending[0].message_id == msg2.message_id

    def test_qos1_duplicate_detection(self):
        """Duplicate message IDs are detected."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        # Simulate duplicate delivery
        assert client.is_duplicate(msg.message_id) is False
        client.acknowledge(msg.message_id)
        # After ack, the message ID is in the dedup set
        assert client.is_duplicate(msg.message_id) is True


# ===========================================================================
# MQTTQoSClient — QoS 2 (Exactly Once)
# ===========================================================================


class TestMQTTQoSClientQoS2:
    """QoS 2: exactly-once delivery with 4-step handshake."""

    def test_publish_qos2(self):
        """Publish a QoS 2 message."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        assert msg.qos == 2
        assert msg.message_id is not None

    def test_publish_qos2_not_connected(self):
        """Publishing QoS 2 when not connected raises RuntimeError."""
        client = MQTTQoSClient()
        with pytest.raises(RuntimeError, match="not connected"):
            client.publish("test/topic", b"hello", qos=2)

    def test_qos2_message_tracked_as_pending(self):
        """QoS 2 messages start as pending."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        state = client.get_delivery_state(msg.message_id)
        assert state is not None
        assert state.state == "pending"

    def test_qos2_pubrec_transitions_to_published(self):
        """PUBREC transitions QoS 2 message to 'published' state."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        assert client.handle_pubrec(msg.message_id) is True
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "published"

    def test_qos2_pubrel_transitions_to_acknowledged(self):
        """PUBREL transitions QoS 2 message to 'acknowledged' state."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        client.handle_pubrec(msg.message_id)
        assert client.handle_pubrel(msg.message_id) is True
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "acknowledged"

    def test_qos2_pubcomp_completes_delivery(self):
        """PUBCOMP completes QoS 2 delivery."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        client.handle_pubrec(msg.message_id)
        client.handle_pubrel(msg.message_id)
        assert client.handle_pubcomp(msg.message_id) is True
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "completed"

    def test_qos2_full_handshake(self):
        """Full QoS 2 handshake: PUBLISH → PUBREC → PUBREL → PUBCOMP."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        assert client.get_delivery_state(msg.message_id).state == "pending"
        client.handle_pubrec(msg.message_id)
        assert client.get_delivery_state(msg.message_id).state == "published"
        client.handle_pubrel(msg.message_id)
        assert client.get_delivery_state(msg.message_id).state == "acknowledged"
        client.handle_pubcomp(msg.message_id)
        assert client.get_delivery_state(msg.message_id).state == "completed"

    def test_qos2_pubrec_nonexistent(self):
        """PUBREC for nonexistent message returns False."""
        client = MQTTQoSClient()
        client.connect()
        assert client.handle_pubrec(99999) is False

    def test_qos2_pubrel_nonexistent(self):
        """PUBREL for nonexistent message returns False."""
        client = MQTTQoSClient()
        client.connect()
        assert client.handle_pubrel(99999) is False

    def test_qos2_pubcomp_nonexistent(self):
        """PUBCOMP for nonexistent message returns False."""
        client = MQTTQoSClient()
        client.connect()
        assert client.handle_pubcomp(99999) is False

    def test_qos2_duplicate_detection(self):
        """QoS 2 messages are tracked for deduplication."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        assert client.is_duplicate(msg.message_id) is False
        # Complete the handshake
        client.handle_pubrec(msg.message_id)
        client.handle_pubrel(msg.message_id)
        client.handle_pubcomp(msg.message_id)
        # After completion, the message ID is in the dedup set
        assert client.is_duplicate(msg.message_id) is True

    def test_qos2_exactly_once_no_duplicate_processing(self):
        """QoS 2 ensures exactly-once processing via dedup."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        # Complete handshake
        client.handle_pubrec(msg.message_id)
        client.handle_pubrel(msg.message_id)
        client.handle_pubcomp(msg.message_id)
        # Duplicate delivery of same message ID should be detected
        assert client.is_duplicate(msg.message_id) is True


# ===========================================================================
# MQTTQoSClient — Delivery State Tracking
# ===========================================================================


class TestMQTTQoSClientDeliveryTracking:
    """Delivery state tracking across QoS levels."""

    def test_get_delivery_state_nonexistent(self):
        """Getting state for nonexistent message returns None."""
        client = MQTTQoSClient()
        client.connect()
        assert client.get_delivery_state(99999) is None

    def test_get_all_delivery_states(self):
        """Get all delivery states."""
        client = MQTTQoSClient()
        client.connect()
        client.publish("test/topic", b"hello1", qos=1)
        client.publish("test/topic", b"hello2", qos=2)
        states = client.get_all_delivery_states()
        assert len(states) == 2

    def test_clear_delivery_states(self):
        """Clear all delivery states."""
        client = MQTTQoSClient()
        client.connect()
        client.publish("test/topic", b"hello", qos=1)
        client.clear_delivery_states()
        states = client.get_all_delivery_states()
        assert len(states) == 0

    def test_mark_failed(self):
        """Mark a message as failed."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        assert client.mark_failed(msg.message_id) is True
        state = client.get_delivery_state(msg.message_id)
        assert state.state == "failed"

    def test_mark_failed_nonexistent(self):
        """Marking nonexistent message as failed returns False."""
        client = MQTTQoSClient()
        client.connect()
        assert client.mark_failed(99999) is False

    def test_get_pending_messages_empty(self):
        """No pending messages when all are completed."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        client.acknowledge(msg.message_id)
        pending = client.get_pending_messages()
        assert len(pending) == 0

    def test_get_pending_messages_mixed_qos(self):
        """Pending messages include both QoS 1 and QoS 2."""
        client = MQTTQoSClient()
        client.connect()
        msg1 = client.publish("test/topic", b"hello1", qos=1)
        msg2 = client.publish("test/topic", b"hello2", qos=2)
        pending = client.get_pending_messages()
        assert len(pending) == 2
        ids = {p.message_id for p in pending}
        assert ids == {msg1.message_id, msg2.message_id}


# ===========================================================================
# MQTTQoSClient — Edge Cases
# ===========================================================================


class TestMQTTQoSClientEdgeCases:
    """Edge cases and boundary conditions."""

    def test_publish_dict_payload(self):
        """Publishing a dict payload encodes to JSON bytes."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", {"temp": 22.5}, qos=0)
        assert msg.payload == b'{"temp": 22.5}'

    def test_publish_str_payload(self):
        """Publishing a string payload encodes to bytes."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", "hello", qos=0)
        assert msg.payload == b"hello"

    def test_publish_bytes_payload(self):
        """Publishing bytes payload keeps them as-is."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"raw-bytes", qos=0)
        assert msg.payload == b"raw-bytes"

    def test_message_id_increments(self):
        """Message IDs increment monotonically."""
        client = MQTTQoSClient()
        client.connect()
        msg1 = client.publish("test/topic", b"hello1", qos=1)
        msg2 = client.publish("test/topic", b"hello2", qos=1)
        msg3 = client.publish("test/topic", b"hello3", qos=2)
        assert msg2.message_id > msg1.message_id
        assert msg3.message_id > msg2.message_id

    def test_disconnect_clears_pending(self):
        """Disconnecting clears pending delivery states."""
        client = MQTTQoSClient()
        client.connect()
        client.publish("test/topic", b"hello", qos=1)
        client.disconnect()
        states = client.get_all_delivery_states()
        assert len(states) == 0

    def test_reconnect_resumes_message_ids(self):
        """Reconnecting continues message ID sequence."""
        client = MQTTQoSClient()
        client.connect()
        msg1 = client.publish("test/topic", b"hello1", qos=1)
        client.disconnect()
        client.connect()
        msg2 = client.publish("test/topic", b"hello2", qos=1)
        assert msg2.message_id > msg1.message_id

    def test_qos1_ack_already_acked(self):
        """Acknowledging an already-acked message returns False."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=1)
        assert client.acknowledge(msg.message_id) is True
        assert client.acknowledge(msg.message_id) is False

    def test_qos2_pubrec_already_received(self):
        """PUBREC for already-received message returns False."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        assert client.handle_pubrec(msg.message_id) is True
        assert client.handle_pubrec(msg.message_id) is False

    def test_qos2_pubrel_before_pubrec(self):
        """PUBREL before PUBREC returns False."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        assert client.handle_pubrel(msg.message_id) is False

    def test_qos2_pubcomp_before_pubrel(self):
        """PUBCOMP before PUBREL returns False."""
        client = MQTTQoSClient()
        client.connect()
        msg = client.publish("test/topic", b"hello", qos=2)
        client.handle_pubrec(msg.message_id)
        assert client.handle_pubcomp(msg.message_id) is False

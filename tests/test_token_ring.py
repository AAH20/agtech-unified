"""Test multi-agent token-ring coordination for narrow passage serialization."""

import pytest

from src.multi_agent.token_ring import (
    TokenRing,
    TokenRingError,
    TokenRingState,
)

# ── MA-TR-001: Token ring initialization ──────────────────────────────


def test_token_ring_init():
    """TokenRing initializes with agents in ring order."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    assert ring.agent_count == 3
    assert ring.ring_order == ["A1", "A2", "A3"]


def test_token_ring_empty_raises():
    """TokenRing with no agents raises ValueError."""
    with pytest.raises(ValueError, match="at least one agent"):
        TokenRing(agents=[])


def test_token_ring_duplicate_agent_raises():
    """TokenRing with duplicate agent IDs raises ValueError."""
    with pytest.raises(ValueError, match="Duplicate agent"):
        TokenRing(agents=["A1", "A1"])


# ── MA-TR-002: Token request and grant ────────────────────────────────


def test_request_token_grants_to_first_requester():
    """First agent to request token becomes the token holder."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    assert ring.token_holder == "A1"
    assert ring.state == TokenRingState.HELD


def test_request_token_queues_second_requester():
    """Second agent to request token is queued."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    assert ring.token_holder == "A1"
    assert "A2" in ring.wait_queue


def test_request_token_unknown_agent_raises():
    """Requesting token for unknown agent raises ValueError."""
    ring = TokenRing(agents=["A1", "A2"])
    with pytest.raises(ValueError, match="Unknown agent"):
        ring.request_token("X9")


def test_request_token_already_holder_raises():
    """Requesting token when already holding raises TokenRingError."""
    ring = TokenRing(agents=["A1", "A2"])
    ring.request_token("A1")
    with pytest.raises(TokenRingError, match="already holds"):
        ring.request_token("A1")


def test_request_token_already_queued_raises():
    """Requesting token when already in queue raises TokenRingError."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    with pytest.raises(TokenRingError, match="already queued"):
        ring.request_token("A2")


# ── MA-TR-003: Token release and pass ─────────────────────────────────


def test_release_token_passes_to_next_in_ring():
    """Releasing token passes it to the next agent in ring order."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.release_token("A1")
    # Token should pass to A2 (next in ring after A1)
    assert ring.token_holder == "A2"


def test_release_token_no_next_holder():
    """Releasing token with no other agents leaves token unheld."""
    ring = TokenRing(agents=["A1"])
    ring.request_token("A1")
    ring.release_token("A1")
    assert ring.token_holder is None
    assert ring.state == TokenRingState.FREE


def test_release_token_not_holder_raises():
    """Releasing token when not holding raises TokenRingError."""
    ring = TokenRing(agents=["A1", "A2"])
    ring.request_token("A1")
    with pytest.raises(TokenRingError, match="does not hold"):
        ring.release_token("A2")


def test_release_token_unknown_agent_raises():
    """Releasing token for unknown agent raises ValueError."""
    ring = TokenRing(agents=["A1"])
    with pytest.raises(ValueError, match="Unknown agent"):
        ring.release_token("X9")


# ── MA-TR-004: Token pass with queued agents ──────────────────────────


def test_release_token_passes_to_queued_agent():
    """Relealing token passes to queued agent if any."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A3")
    ring.release_token("A1")
    # A3 requested before A2 in ring order, but A3 is queued
    # Token should go to the queued agent
    assert ring.token_holder == "A3"


def test_release_token_queue_order_preserved():
    """Multiple queued agents are served in request order."""
    ring = TokenRing(agents=["A1", "A2", "A3", "A4"])
    ring.request_token("A1")
    ring.request_token("A3")
    ring.request_token("A2")
    ring.release_token("A1")
    assert ring.token_holder == "A3"
    ring.release_token("A3")
    assert ring.token_holder == "A2"


# ── MA-TR-005: Narrow passage serialization ───────────────────────────


def test_narrow_passage_only_one_agent():
    """Only one agent can hold the token (enter narrow passage) at a time."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    ring.request_token("A3")
    # Only A1 holds the token
    assert ring.token_holder == "A1"
    # A2 and A3 are waiting
    assert "A2" in ring.wait_queue
    assert "A3" in ring.wait_queue


def test_narrow_passage_serialization_order():
    """Agents pass through narrow passage in ring order."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    order = []
    # A1 requests and holds token
    ring.request_token("A1")
    order.append(ring.token_holder)
    # A1 releases — token passes to A2 (next in ring)
    ring.release_token("A1")
    order.append(ring.token_holder)
    # A2 releases — token passes to A3 (next in ring)
    ring.release_token("A2")
    order.append(ring.token_holder)
    assert order == ["A1", "A2", "A3"]


def test_narrow_passage_mutual_exclusion():
    """Token holder is unique — mutual exclusion is guaranteed."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    # A1 holds, A2 waits — A2 cannot enter
    assert ring.token_holder == "A1"
    assert ring.token_holder != "A2"


# ── MA-TR-006: Token ring state queries ───────────────────────────────


def test_is_token_holder():
    """is_token_holder returns True only for the current holder."""
    ring = TokenRing(agents=["A1", "A2"])
    ring.request_token("A1")
    assert ring.is_token_holder("A1")
    assert not ring.is_token_holder("A2")


def test_get_wait_queue():
    """get_wait_queue returns agents waiting for the token."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    ring.request_token("A3")
    queue = ring.get_wait_queue()
    assert "A2" in queue
    assert "A3" in queue
    assert "A1" not in queue


def test_state_free_when_no_holder():
    """State is FREE when no agent holds the token."""
    ring = TokenRing(agents=["A1", "A2"])
    assert ring.state == TokenRingState.FREE


def test_state_held_when_holder_exists():
    """State is HELD when an agent holds the token."""
    ring = TokenRing(agents=["A1", "A2"])
    ring.request_token("A1")
    assert ring.state == TokenRingState.HELD


# ── MA-TR-007: Token ring with agent failure ──────────────────────────


def test_handle_failed_holder_releases_token():
    """When token holder fails, token passes to next agent."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    ring.handle_failed_agent("A1")
    assert ring.token_holder == "A2"


def test_handle_failed_queued_agent_removed():
    """Failed queued agent is removed from wait queue."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    ring.request_token("A3")
    ring.handle_failed_agent("A2")
    assert "A2" not in ring.wait_queue
    assert "A3" in ring.wait_queue


def test_handle_failed_agent_not_in_ring_raises():
    """Handling failure of unknown agent raises ValueError."""
    ring = TokenRing(agents=["A1"])
    with pytest.raises(ValueError, match="Unknown agent"):
        ring.handle_failed_agent("X9")


def test_handle_failed_agent_no_holder():
    """Handling failure when no token holder is a no-op."""
    ring = TokenRing(agents=["A1", "A2"])
    ring.handle_failed_agent("A1")
    assert ring.token_holder is None


# ── MA-TR-008: Token ring integration with positions ──────────────────


def test_agent_in_narrow_passage():
    """Agent holding token is considered in the narrow passage."""
    ring = TokenRing(agents=["A1", "A2"])
    ring.request_token("A1")
    assert ring.is_in_narrow_passage("A1")
    assert not ring.is_in_narrow_passage("A2")


def test_agent_not_in_narrow_passage_when_free():
    """No agent is in narrow passage when token is free."""
    ring = TokenRing(agents=["A1", "A2"])
    assert not ring.is_in_narrow_passage("A1")
    assert not ring.is_in_narrow_passage("A2")


# ── MA-TR-009: Token ring reset ───────────────────────────────────────


def test_reset_releases_all():
    """Reset releases token and clears wait queue."""
    ring = TokenRing(agents=["A1", "A2", "A3"])
    ring.request_token("A1")
    ring.request_token("A2")
    ring.reset()
    assert ring.token_holder is None
    assert ring.state == TokenRingState.FREE
    assert len(ring.wait_queue) == 0


# ── MA-TR-010: Token ring with custom ring order ─────────────────────


def test_custom_ring_order():
    """TokenRing respects custom ring order."""
    ring = TokenRing(agents=["A1", "A2", "A3"], ring_order=["A3", "A1", "A2"])
    ring.request_token("A3")
    ring.release_token("A3")
    # Next in custom order after A3 is A1
    assert ring.token_holder == "A1"


def test_custom_ring_order_invalid_raises():
    """Custom ring order with unknown agent raises ValueError."""
    with pytest.raises(ValueError, match="Unknown agent"):
        TokenRing(agents=["A1", "A2"], ring_order=["A1", "X9"])


def test_custom_ring_order_missing_agent_raises():
    """Custom ring order missing an agent raises ValueError."""
    with pytest.raises(ValueError, match="missing"):
        TokenRing(agents=["A1", "A2", "A3"], ring_order=["A1", "A2"])

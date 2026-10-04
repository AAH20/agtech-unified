"""Tests for Contract Net Protocol (CNP) task allocation."""

from src.multi_agent.contract_net import (
    CNPAgent,
    CNPManager,
    CNPState,
)

# ── MA-CNP-001: Basic CNP flow ────────────────────────────────────────


def test_cnp_full_round_task_announcement_to_award():
    """Full CNP round: announce → bid → award → complete."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])
    manager.register_agent("A2", capabilities=["scan"])

    msg = manager.announce_task("T1", requirements=["spray"])
    assert msg.task_id == "T1"
    assert msg.requirements == ["spray"]


def test_cnp_agent_submits_bid():
    """Agent submits a bid for a task."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])
    manager.announce_task("T1", requirements=["spray"])

    bid = manager.submit_bid("A1", "T1", bid_amount=5.0)
    assert bid is not None
    assert bid.agent_id == "A1"
    assert bid.task_id == "T1"
    assert bid.bid_amount == 5.0


def test_cnp_manager_awards_to_lowest_bidder():
    """Manager awards task to agent with lowest bid."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])
    manager.register_agent("A2", capabilities=["spray"])
    manager.announce_task("T1", requirements=["spray"])

    manager.submit_bid("A1", "T1", bid_amount=10.0)
    manager.submit_bid("A2", "T1", bid_amount=5.0)

    award = manager.award_task("T1")
    assert award is not None
    assert award.winner_id == "A2"
    assert award.winning_price == 5.0


def test_cnp_bid_rejected_when_no_capability():
    """Bid from agent without required capability is rejected."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["scan"])

    bid = manager.submit_bid("A1", "T1", bid_amount=5.0)
    assert bid is None


def test_cnp_award_no_bids():
    """Award with no bids returns None."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])

    award = manager.award_task("T1")
    assert award is None


# ── MA-CNP-002: CNP state machine ─────────────────────────────────────


def test_cnp_state_transitions():
    """CNP state follows correct lifecycle."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])

    manager.announce_task("T1", requirements=["spray"])
    state = manager.get_state("T1")
    assert state == CNPState.ANNOUNCED

    manager.submit_bid("A1", "T1", bid_amount=5.0)
    state = manager.get_state("T1")
    assert state == CNPState.BIDDING

    award = manager.award_task("T1")
    state = manager.get_state("T1")
    assert state == CNPState.AWARDED
    assert award.winner_id == "A1"


def test_cnp_state_completed():
    """CNP state transitions to COMPLETED after task completion."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])

    manager.announce_task("T1", requirements=["spray"])
    manager.submit_bid("A1", "T1", bid_amount=5.0)
    manager.award_task("T1")
    manager.complete_task("T1")
    assert manager.get_state("T1") == CNPState.COMPLETED


def test_cnp_state_failed():
    """CNP state transitions to FAILED when winner fails."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])

    manager.announce_task("T1", requirements=["spray"])
    manager.submit_bid("A1", "T1", bid_amount=5.0)
    manager.award_task("T1")
    manager.fail_task("T1")
    assert manager.get_state("T1") == CNPState.FAILED


# ── MA-CNP-003: Multi-agent CNP ───────────────────────────────────────


def test_cnp_multiple_tasks():
    """Multiple tasks can be auctioned independently."""
    manager = CNPManager()
    manager.register_agent("A1", capabilities=["spray"])
    manager.register_agent("A2", capabilities=["scan"])

    manager.announce_task("T1", requirements=["spray"])
    manager.announce_task("T2", requirements=["scan"])

    manager.submit_bid("A1", "T1", bid_amount=5.0)
    manager.submit_bid("A2", "T2", bid_amount=3.0)

    award1 = manager.award_task("T1")
    award2 = manager.award_task("T2")

    assert award1.winner_id == "A1"
    assert award2.winner_id == "A2"


# ── MA-CNP-004: CNPAgent bid evaluation ───────────────────────────────


def test_cnp_agent_evaluates_bid():
    """CNPAgent can evaluate whether to bid on a task."""
    agent = CNPAgent("A1", capabilities=["spray"], cost_estimator=lambda task_id: 5.0)
    should_bid = agent.evaluate_bid("T1", requirements=["spray"])
    assert should_bid is True


def test_cnp_agent_no_bid_without_capability():
    """CNPAgent doesn't bid when lacking capability."""
    agent = CNPAgent("A1", capabilities=["scan"], cost_estimator=lambda task_id: 5.0)
    should_bid = agent.evaluate_bid("T1", requirements=["spray"])
    assert should_bid is False


def test_cnp_agent_cost_estimation():
    """CNPAgent uses cost estimator to determine bid amount."""
    agent = CNPAgent("A1", capabilities=["spray"], cost_estimator=lambda task_id: 7.0)
    bid_amount = agent.estimate_cost("T1")
    assert bid_amount == 7.0

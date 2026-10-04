"""Tests for auction-based task allocation (Vickrey auction)."""

from src.multi_agent.auction import Auctioneer, Bid

# ── MA-AUC-001: Basic Vickrey auction ─────────────────────────────────


def test_vickrey_auction_highest_bidder_wins():
    """Highest bidder wins the auction."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=10.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=20.0),
        Bid(agent_id="A3", task_id="T1", bid_amount=15.0),
    ]
    result = auctioneer.run_auction("T1", bids)
    assert result.winner_id == "A2"
    assert result.winning_price == 15.0  # Second-highest bid (Vickrey)


def test_vickrey_auction_single_bidder_pays_zero():
    """Single bidder wins and pays zero (no competition)."""
    auctioneer = Auctioneer()
    bids = [Bid(agent_id="A1", task_id="T1", bid_amount=10.0)]
    result = auctioneer.run_auction("T1", bids)
    assert result.winner_id == "A1"
    assert result.winning_price == 0.0


def test_vickrey_auction_no_bids():
    """No bids results in no winner."""
    auctioneer = Auctioneer()
    result = auctioneer.run_auction("T1", [])
    assert result.winner_id is None
    assert result.winning_price == 0.0


def test_vickrey_auction_tie_breaking():
    """Tie between equal bids: first bidder wins (deterministic)."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=10.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=10.0),
    ]
    result = auctioneer.run_auction("T1", bids)
    assert result.winner_id == "A1"
    assert result.winning_price == 10.0


# ── MA-AUC-002: Capability constraints ────────────────────────────────


def test_auction_respects_capability_constraints():
    """Agents can only bid on tasks they have capabilities for."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=5.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=20.0),
    ]
    # A2 doesn't have the required capability
    capabilities = {"A1": ["spray"], "A2": ["scan"]}
    requirements = {"T1": ["spray"]}
    result = auctioneer.run_auction(
        "T1", bids, capabilities=capabilities, requirements=requirements
    )
    assert result.winner_id == "A1"
    assert result.winning_price == 0.0  # Only one valid bid


def test_auction_all_bids_filtered_by_capability():
    """When no agent has required capability, no winner."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=10.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=20.0),
    ]
    capabilities = {"A1": ["scan"], "A2": ["scan"]}
    requirements = {"T1": ["harvest"]}
    result = auctioneer.run_auction(
        "T1", bids, capabilities=capabilities, requirements=requirements
    )
    assert result.winner_id is None


# ── MA-AUC-003: Multi-task auction ────────────────────────────────────


def test_auction_multiple_tasks():
    """Run auctions for multiple tasks."""
    auctioneer = Auctioneer()
    all_bids = {
        "T1": [
            Bid(agent_id="A1", task_id="T1", bid_amount=10.0),
            Bid(agent_id="A2", task_id="T1", bid_amount=20.0),
        ],
        "T2": [
            Bid(agent_id="A1", task_id="T2", bid_amount=30.0),
            Bid(agent_id="A2", task_id="T2", bid_amount=15.0),
        ],
    }
    results = auctioneer.run_multi_task_auction(all_bids)
    assert results["T1"].winner_id == "A2"
    assert results["T1"].winning_price == 10.0
    assert results["T2"].winner_id == "A1"
    assert results["T2"].winning_price == 15.0


# ── MA-AUC-004: Bid validation ────────────────────────────────────────


def test_auction_negative_bid_rejected():
    """Negative bids are rejected."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=-5.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=10.0),
    ]
    result = auctioneer.run_auction("T1", bids)
    assert result.winner_id == "A2"
    assert result.winning_price == 0.0


def test_auction_zero_bid_valid():
    """Zero bid is valid (agent willing to do task for free)."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=0.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=10.0),
    ]
    result = auctioneer.run_auction("T1", bids)
    assert result.winner_id == "A2"
    assert result.winning_price == 0.0


# ── MA-AUC-005: Auction result properties ─────────────────────────────


def test_auction_result_contains_all_bidders():
    """AuctionResult tracks all bidders."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=10.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=20.0),
    ]
    result = auctioneer.run_auction("T1", bids)
    assert set(result.all_bidders) == {"A1", "A2"}


def test_auction_result_losing_bids():
    """AuctionResult tracks losing bids."""
    auctioneer = Auctioneer()
    bids = [
        Bid(agent_id="A1", task_id="T1", bid_amount=10.0),
        Bid(agent_id="A2", task_id="T1", bid_amount=20.0),
    ]
    result = auctioneer.run_auction("T1", bids)
    assert result.losing_bids["A1"] == 10.0

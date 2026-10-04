"""Test Byzantine fault tolerance consensus for swarm decisions."""

import pytest

from src.multi_agent.consensus import (
    ByzantineConsensus,
    ConsensusStatus,
)


def test_consensus_single_node():
    """Single node reaches consensus trivially."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=1, max_faulty=0)
    result = consensus.propose("decision_A")
    assert result.status == ConsensusStatus.COMMITTED
    assert result.value == "decision_A"


def test_consensus_quorum_calculation():
    """Quorum is 2f+1 for f faulty nodes."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=4, max_faulty=1)
    assert consensus.quorum == 3


def test_consensus_too_many_faulty():
    """System with n < 3f+1 cannot tolerate f faults."""
    with pytest.raises(ValueError, match="Cannot tolerate"):
        ByzantineConsensus(node_id="node_1", total_nodes=3, max_faulty=1)


def test_consensus_commit_with_2f_plus_1_votes():
    """2f+1 matching votes commit the value."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=4, max_faulty=1)
    votes = ["A", "A", "A"]
    result = consensus.tally_votes(votes)
    assert result.status == ConsensusStatus.COMMITTED
    assert result.value == "A"


def test_consensus_no_commit_with_insufficient_votes():
    """Fewer than 2f+1 votes cannot commit."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=4, max_faulty=1)
    votes = ["A", "A"]
    result = consensus.tally_votes(votes)
    assert result.status == ConsensusStatus.PENDING


def test_consensus_different_values_no_commit():
    """Conflicting votes prevent commit."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=4, max_faulty=1)
    votes = ["A", "B", "C"]
    result = consensus.tally_votes(votes)
    assert result.status == ConsensusStatus.PENDING


def test_consensus_majority_value_wins():
    """When multiple values have votes, the one reaching quorum wins."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=7, max_faulty=2)
    votes = ["A", "A", "A", "A", "A", "B", "C"]
    result = consensus.tally_votes(votes)
    assert result.status == ConsensusStatus.COMMITTED
    assert result.value == "A"


def test_consensus_empty_votes():
    """Empty vote list returns PENDING."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=4, max_faulty=1)
    result = consensus.tally_votes([])
    assert result.status == ConsensusStatus.PENDING


def test_consensus_max_faulty_zero():
    """Zero faulty nodes tolerated in 1-node system."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=1, max_faulty=0)
    assert consensus.max_faulty == 0
    result = consensus.propose("X")
    assert result.status == ConsensusStatus.COMMITTED


def test_consensus_result_contains_node_count():
    """ConsensusResult includes total node count."""
    consensus = ByzantineConsensus(node_id="node_1", total_nodes=4, max_faulty=1)
    result = consensus.propose("test")
    assert result.total_nodes == 4
    assert result.quorum == 3

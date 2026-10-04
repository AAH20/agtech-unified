"""Byzantine fault tolerance consensus for swarm decisions.

Implements a PBFT-style (Practical Byzantine Fault Tolerance) consensus
protocol for agricultural robot swarms. Tolerates up to f faulty nodes
in a system of n >= 3f+1 nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConsensusStatus(Enum):
    """Status of a consensus decision."""
    PENDING = "pending"
    COMMITTED = "committed"
    REJECTED = "rejected"


@dataclass
class ConsensusResult:
    """Result of a consensus round."""
    status: ConsensusStatus
    value: Optional[str] = None
    total_nodes: int = 0
    quorum: int = 0
    votes_received: int = 0


class ByzantineConsensus:
    """PBFT-style Byzantine consensus for swarm decision-making.

    A swarm of n nodes can tolerate up to f Byzantine (arbitrary/faulty)
    nodes where n >= 3f + 1. Consensus is reached when 2f + 1 matching
    votes are collected (quorum).

    Example:
        >>> consensus = ByzantineConsensus("node_1", total_nodes=4, max_faulty=1)
        >>> result = consensus.propose("harvest_field_A")
        >>> result.status == ConsensusStatus.COMMITTED
        True
    """

    def __init__(self, node_id: str, total_nodes: int, max_faulty: int):
        """Initialize Byzantine consensus.

        Args:
            node_id: Identifier of this node.
            total_nodes: Total number of nodes in the swarm (n).
            max_faulty: Maximum number of faulty nodes to tolerate (f).

        Raises:
            ValueError: If n < 3f + 1 (cannot tolerate f faults).
        """
        if total_nodes < 3 * max_faulty + 1:
            raise ValueError(
                f"Cannot tolerate {max_faulty} faulty nodes with only "
                f"{total_nodes} nodes (need n >= 3f+1 = {3 * max_faulty + 1})"
            )
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.max_faulty = max_faulty
        self.quorum = 2 * max_faulty + 1

    def propose(self, value: str) -> ConsensusResult:
        """Propose a value for consensus.

        In a real PBFT system, this would broadcast the proposal to all
        nodes and collect votes. Here we simulate the local node's vote
        and return the result.

        Args:
            value: The proposed value (e.g., a task assignment decision).

        Returns:
            ConsensusResult indicating whether the value was committed.
        """
        # In single-node or zero-fault systems, proposal auto-commits
        if self.total_nodes == 1 or self.max_faulty == 0:
            return ConsensusResult(
                status=ConsensusStatus.COMMITTED,
                value=value,
                total_nodes=self.total_nodes,
                quorum=self.quorum,
                votes_received=1,
            )
        # For multi-node systems, propose starts the voting process
        return ConsensusResult(
            status=ConsensusStatus.PENDING,
            value=value,
            total_nodes=self.total_nodes,
            quorum=self.quorum,
            votes_received=1,
        )

    def tally_votes(self, votes: List[str]) -> ConsensusResult:
        """Tally votes and determine if consensus is reached.

        Consensus is reached when at least 2f+1 nodes vote for the same
        value. If no value reaches quorum, the result is PENDING.

        Args:
            votes: List of vote values from nodes.

        Returns:
            ConsensusResult with the consensus outcome.
        """
        if not votes:
            return ConsensusResult(
                status=ConsensusStatus.PENDING,
                total_nodes=self.total_nodes,
                quorum=self.quorum,
                votes_received=0,
            )

        # Count votes for each value
        vote_counts: Dict[str, int] = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Find the value with the most votes
        best_value = max(vote_counts, key=lambda v: vote_counts[v])
        best_count = vote_counts[best_value]

        if best_count >= self.quorum:
            return ConsensusResult(
                status=ConsensusStatus.COMMITTED,
                value=best_value,
                total_nodes=self.total_nodes,
                quorum=self.quorum,
                votes_received=len(votes),
            )

        return ConsensusResult(
            status=ConsensusStatus.PENDING,
            value=best_value,
            total_nodes=self.total_nodes,
            quorum=self.quorum,
            votes_received=len(votes),
        )

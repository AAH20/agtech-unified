"""Contract Net Protocol (CNP) for multi-agent task allocation.

Implements the standard CNP negotiation loop:
1. Manager announces a task
2. Agents bid on the task
3. Manager awards the task to the best bidder
4. Winner executes and reports completion/failure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from src.multi_agent.auction import AuctionResult, Bid

logger = logging.getLogger(__name__)


class CNPState(Enum):
    """State of a CNP task negotiation."""

    ANNOUNCED = "announced"
    BIDDING = "bidding"
    AWARDED = "awarded"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CNPMessage:
    """A message in the CNP protocol."""

    task_id: str
    message_type: str
    sender: str
    requirements: List[str] = field(default_factory=list)
    bid_amount: Optional[float] = None


@dataclass
class CNPAgent:
    """An agent participating in the Contract Net Protocol."""

    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    cost_estimator: Callable[[str], float] = field(default=lambda task_id: 0.0)

    def evaluate_bid(self, task_id: str, requirements: List[str]) -> bool:
        """Decide whether to bid on a task.

        Args:
            task_id: The task to evaluate.
            requirements: Required capabilities for the task.

        Returns:
            True if the agent has all required capabilities.
        """
        return all(req in self.capabilities for req in requirements)

    def estimate_cost(self, task_id: str) -> float:
        """Estimate the cost of performing a task.

        Args:
            task_id: The task to estimate cost for.

        Returns:
            Estimated cost.
        """
        return self.cost_estimator(task_id)


class CNPManager:
    """Manager for Contract Net Protocol negotiations.

    Handles task announcement, bid collection, and task awarding.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, CNPAgent] = {}
        self._tasks: Dict[str, CNPMessage] = {}
        self._bids: Dict[str, List[Bid]] = {}
        self._states: Dict[str, CNPState] = {}
        self._awards: Dict[str, AuctionResult] = {}

    def register_agent(self, agent_id: str, capabilities: Optional[List[str]] = None) -> None:
        """Register an agent for CNP participation.

        Args:
            agent_id: Unique identifier for the agent.
            capabilities: List of capabilities the agent provides.
        """
        self._agents[agent_id] = CNPAgent(
            agent_id=agent_id,
            capabilities=capabilities or [],
        )

    def announce_task(self, task_id: str, requirements: Optional[List[str]] = None) -> CNPMessage:
        """Announce a task for bidding.

        Args:
            task_id: The task to announce.
            requirements: Required capabilities for the task.

        Returns:
            The announcement message.
        """
        msg = CNPMessage(
            task_id=task_id,
            message_type="ANNOUNCE",
            sender="manager",
            requirements=requirements or [],
        )
        self._tasks[task_id] = msg
        self._bids[task_id] = []
        self._states[task_id] = CNPState.ANNOUNCED
        logger.info("Announced task %s with requirements %s", task_id, requirements)
        return msg

    def submit_bid(self, agent_id: str, task_id: str, bid_amount: float) -> Optional[Bid]:
        """Submit a bid for a task.

        Args:
            agent_id: The bidding agent.
            task_id: The task being bid on.
            bid_amount: The bid amount.

        Returns:
            The Bid if accepted, None if rejected.
        """
        if agent_id not in self._agents:
            logger.warning("Bid from unknown agent %s", agent_id)
            return None

        if task_id not in self._tasks:
            logger.warning("Bid for unknown task %s", task_id)
            return None

        task = self._tasks[task_id]
        agent = self._agents[agent_id]

        # Check capability constraints
        if not agent.evaluate_bid(task_id, task.requirements):
            logger.debug("Bid from %s rejected: lacks capabilities", agent_id)
            return None

        bid = Bid(agent_id=agent_id, task_id=task_id, bid_amount=bid_amount)
        self._bids[task_id].append(bid)
        self._states[task_id] = CNPState.BIDDING
        logger.info("Agent %s bid %.2f on task %s", agent_id, bid_amount, task_id)
        return bid

    def award_task(self, task_id: str) -> Optional[AuctionResult]:
        """Award a task to the lowest bidder.

        Args:
            task_id: The task to award.

        Returns:
            AuctionResult with winner info, or None if no valid bids.
        """
        if task_id not in self._tasks:
            logger.warning("Cannot award unknown task %s", task_id)
            return None

        bids = self._bids.get(task_id, [])
        if not bids:
            logger.warning("No bids for task %s", task_id)
            return None

        # Award to lowest bidder (first-price: winner pays their own bid)
        sorted_bids = sorted(bids, key=lambda b: b.bid_amount)
        winner = sorted_bids[0]
        winning_price = winner.bid_amount

        result = AuctionResult(
            task_id=task_id,
            winner_id=winner.agent_id,
            winning_price=winning_price,
            all_bidders=[b.agent_id for b in bids],
            losing_bids={b.agent_id: b.bid_amount for b in sorted_bids[1:]},
        )

        self._awards[task_id] = result
        self._states[task_id] = CNPState.AWARDED
        logger.info("Awarded task %s to %s at price %.2f", task_id, winner.agent_id, winning_price)
        return result

    def get_state(self, task_id: str) -> CNPState:
        """Get the current CNP state of a task.

        Args:
            task_id: The task to check.

        Returns:
            Current CNPState.
        """
        return self._states.get(task_id, CNPState.ANNOUNCED)

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed.

        Args:
            task_id: The completed task.
        """
        if task_id in self._states:
            self._states[task_id] = CNPState.COMPLETED
            logger.info("Task %s completed", task_id)

    def fail_task(self, task_id: str) -> None:
        """Mark a task as failed.

        Args:
            task_id: The failed task.
        """
        if task_id in self._states:
            self._states[task_id] = CNPState.FAILED
            logger.warning("Task %s failed", task_id)

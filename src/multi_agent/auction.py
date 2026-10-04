"""Auction-based task allocation using Vickrey (second-price) auction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Bid:
    """A bid from an agent for a task."""

    agent_id: str
    task_id: str
    bid_amount: float


@dataclass
class AuctionResult:
    """Result of an auction."""

    task_id: str
    winner_id: Optional[str]
    winning_price: float
    all_bidders: List[str] = field(default_factory=list)
    losing_bids: Dict[str, float] = field(default_factory=dict)


class Auctioneer:
    """Vickrey auctioneer for task allocation.

    In a Vickrey (second-price) auction, the highest bidder wins but
    pays the second-highest bid price. This incentivizes agents to
    bid their true cost.
    """

    def run_auction(
        self,
        task_id: str,
        bids: List[Bid],
        capabilities: Optional[Dict[str, List[str]]] = None,
        requirements: Optional[Dict[str, List[str]]] = None,
    ) -> AuctionResult:
        """Run a Vickrey auction for a single task.

        Args:
            task_id: The task being auctioned.
            bids: List of bids from agents.
            capabilities: Optional dict mapping agent_id to capabilities.
            requirements: Optional dict mapping task_id to required capabilities.

        Returns:
            AuctionResult with winner and price.
        """
        # Filter bids by capability constraints
        valid_bids = self._filter_bids(bids, capabilities, requirements)

        if not valid_bids:
            return AuctionResult(
                task_id=task_id,
                winner_id=None,
                winning_price=0.0,
                all_bidders=[b.agent_id for b in bids],
            )

        # Sort by bid amount descending
        sorted_bids = sorted(valid_bids, key=lambda b: b.bid_amount, reverse=True)

        winner = sorted_bids[0]
        # Vickrey price is the second-highest bid, or 0 if only one bid
        winning_price = sorted_bids[1].bid_amount if len(sorted_bids) > 1 else 0.0

        losing_bids = {b.agent_id: b.bid_amount for b in sorted_bids[1:]}

        return AuctionResult(
            task_id=task_id,
            winner_id=winner.agent_id,
            winning_price=winning_price,
            all_bidders=[b.agent_id for b in valid_bids],
            losing_bids=losing_bids,
        )

    def run_multi_task_auction(
        self,
        all_bids: Dict[str, List[Bid]],
        capabilities: Optional[Dict[str, List[str]]] = None,
        requirements: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, AuctionResult]:
        """Run auctions for multiple tasks.

        Args:
            all_bids: Dict mapping task_id to list of bids.
            capabilities: Optional dict mapping agent_id to capabilities.
            requirements: Optional dict mapping task_id to required capabilities.

        Returns:
            Dict mapping task_id to AuctionResult.
        """
        results: Dict[str, AuctionResult] = {}
        for task_id, bids in all_bids.items():
            results[task_id] = self.run_auction(
                task_id, bids, capabilities=capabilities, requirements=requirements
            )
        return results

    def _filter_bids(
        self,
        bids: List[Bid],
        capabilities: Optional[Dict[str, List[str]]],
        requirements: Optional[Dict[str, List[str]]],
    ) -> List[Bid]:
        """Filter bids to only valid ones (non-negative, capable)."""
        valid: List[Bid] = []
        for bid in bids:
            # Reject negative bids
            if bid.bid_amount < 0:
                logger.warning("Rejected negative bid from %s: %f", bid.agent_id, bid.bid_amount)
                continue

            # Check capability constraints
            if capabilities is not None and requirements is not None:
                reqs = requirements.get(bid.task_id, [])
                if reqs:
                    agent_caps = capabilities.get(bid.agent_id, [])
                    if not all(req in agent_caps for req in reqs):
                        logger.debug(
                            "Bid from %s rejected: lacks capabilities for %s",
                            bid.agent_id,
                            bid.task_id,
                        )
                        continue

            valid.append(bid)

        return valid

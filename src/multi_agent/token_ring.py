"""Token-ring coordination for multi-agent narrow passage serialization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class TokenRingState(Enum):
    """State of the token ring."""

    FREE = "free"
    HELD = "held"


class TokenRingError(Exception):
    """Error raised for invalid token ring operations."""


@dataclass
class TokenRing:
    """Token-ring coordination for serializing agents through narrow passages.

    Agents are arranged in a logical ring. Only the token holder may enter
    the narrow passage. When the holder releases the token, it passes to
    the next agent in ring order (or to the next queued requester).
    """

    agents: List[str]
    ring_order: Optional[List[str]] = None
    token_holder: Optional[str] = None
    wait_queue: List[str] = field(default_factory=list)
    state: TokenRingState = TokenRingState.FREE

    def __post_init__(self):
        """Validate and initialize the token ring."""
        if not self.agents:
            raise ValueError("TokenRing requires at least one agent")
        if len(self.agents) != len(set(self.agents)):
            raise ValueError(f"Duplicate agent in ring: {self.agents}")

        if self.ring_order is None:
            self.ring_order = list(self.agents)
        else:
            if set(self.ring_order) != set(self.agents):
                missing = set(self.agents) - set(self.ring_order)
                unknown = set(self.ring_order) - set(self.agents)
                if unknown:
                    raise ValueError(f"Unknown agent in ring_order: {unknown}")
                if missing:
                    raise ValueError(f"Agents missing from ring_order: {missing}")

    @property
    def agent_count(self) -> int:
        """Number of agents in the ring."""
        return len(self.agents)

    def _validate_agent(self, agent_id: str) -> None:
        """Validate that an agent is in the ring."""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")

    def request_token(self, agent_id: str) -> None:
        """Request the token for an agent.

        Args:
            agent_id: The agent requesting the token.

        Raises:
            ValueError: If the agent is not in the ring.
            TokenRingError: If the agent already holds the token or is queued.
        """
        self._validate_agent(agent_id)
        if self.token_holder == agent_id:
            raise TokenRingError(f"Agent '{agent_id}' already holds the token")
        if agent_id in self.wait_queue:
            raise TokenRingError(f"Agent '{agent_id}' is already queued")

        if self.token_holder is None:
            self.token_holder = agent_id
            self.state = TokenRingState.HELD
            logger.debug("Token granted to %s", agent_id)
        else:
            self.wait_queue.append(agent_id)
            logger.debug("Agent %s queued for token", agent_id)

    def release_token(self, agent_id: str) -> None:
        """Release the token from an agent.

        The token passes to the next queued requester if any, otherwise
        to the next agent in ring order.

        Args:
            agent_id: The agent releasing the token.

        Raises:
            ValueError: If the agent is not in the ring.
            TokenRingError: If the agent does not hold the token.
        """
        self._validate_agent(agent_id)
        if self.token_holder != agent_id:
            raise TokenRingError(f"Agent '{agent_id}' does not hold the token")

        self.token_holder = None
        self.state = TokenRingState.FREE
        logger.debug("Token released by %s", agent_id)

        # Pass to queued agent if any
        if self.wait_queue:
            next_agent = self.wait_queue.pop(0)
            self.token_holder = next_agent
            self.state = TokenRingState.HELD
            logger.debug("Token passed to queued agent %s", next_agent)
        else:
            # Pass to next agent in ring order
            self._pass_to_next_in_ring(agent_id)

    def _pass_to_next_in_ring(self, current_holder: str) -> None:
        """Pass token to the next agent in ring order.

        Args:
            current_holder: The agent that just released the token.
        """
        if not self.ring_order or len(self.ring_order) <= 1:
            return
        try:
            idx = self.ring_order.index(current_holder)
        except ValueError:
            return
        next_idx = (idx + 1) % len(self.ring_order)
        next_agent = self.ring_order[next_idx]
        if next_agent != current_holder:
            self.token_holder = next_agent
            self.state = TokenRingState.HELD
            logger.debug("Token passed to next in ring: %s", next_agent)

    def handle_failed_agent(self, agent_id: str) -> None:
        """Handle an agent failure by removing it from the ring.

        If the failed agent holds the token, the token is passed on.
        If the failed agent is in the wait queue, it is removed.

        Args:
            agent_id: The failed agent.

        Raises:
            ValueError: If the agent is not in the ring.
        """
        self._validate_agent(agent_id)

        if self.token_holder == agent_id:
            self.token_holder = None
            self.state = TokenRingState.FREE
            if self.wait_queue:
                next_agent = self.wait_queue.pop(0)
                self.token_holder = next_agent
                self.state = TokenRingState.HELD
                logger.debug("Token passed to queued agent %s after failure", next_agent)
            else:
                self._pass_to_next_in_ring(agent_id)

        if agent_id in self.wait_queue:
            self.wait_queue.remove(agent_id)
            logger.debug("Failed agent %s removed from wait queue", agent_id)

        logger.info("Handled failure of agent %s", agent_id)

    def is_token_holder(self, agent_id: str) -> bool:
        """Check if an agent currently holds the token.

        Args:
            agent_id: The agent to check.

        Returns:
            True if the agent holds the token.
        """
        return self.token_holder == agent_id

    def is_in_narrow_passage(self, agent_id: str) -> bool:
        """Check if an agent is in the narrow passage (holds the token).

        Args:
            agent_id: The agent to check.

        Returns:
            True if the agent is in the narrow passage.
        """
        return self.token_holder == agent_id

    def get_wait_queue(self) -> List[str]:
        """Get the current wait queue.

        Returns:
            List of agent IDs waiting for the token.
        """
        return list(self.wait_queue)

    def reset(self) -> None:
        """Reset the token ring, releasing the token and clearing the queue."""
        self.token_holder = None
        self.wait_queue.clear()
        self.state = TokenRingState.FREE
        logger.debug("Token ring reset")

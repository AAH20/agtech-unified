"""Multi-agent coalition formation for multi-capability tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class CoalitionStatus(Enum):
    """Status of a coalition."""

    ACTIVE = "active"
    DISSOLVED = "dissolved"


@dataclass
class AgentProfile:
    """Profile of an agent with capabilities and optional cost."""

    agent_id: str
    capabilities: List[str]
    cost: float = 1.0


@dataclass
class TaskRequirement:
    """Task requirement with required capabilities and optional priority."""

    task_id: str
    required_capabilities: List[str]
    priority: int = 0


@dataclass
class Coalition:
    """A coalition of agents formed to accomplish a task."""

    coalition_id: str
    members: List[str] = field(default_factory=list)
    status: CoalitionStatus = CoalitionStatus.ACTIVE
    _agent_costs: Dict[str, float] = field(default_factory=dict)
    _capabilities: Set[str] = field(default_factory=set)

    def get_covered_capabilities(self) -> Set[str]:
        """Get the set of capabilities covered by this coalition.

        Returns:
            Set of capability strings.
        """
        return set(self._capabilities)

    def get_utility(self, required_capabilities: List[str]) -> float:
        """Compute utility of this coalition for given required capabilities.

        Args:
            required_capabilities: The capabilities required by the task.

        Returns:
            Utility score (higher is better).
        """
        if self.status == CoalitionStatus.DISSOLVED:
            return 0.0
        # Utility is based on the fraction of required capabilities covered
        # Actual coverage is determined by CoalitionFormation
        covered = self.get_covered_capabilities()
        if not required_capabilities:
            return 0.0
        covered_required = sum(1 for cap in required_capabilities if cap in covered)
        return covered_required / len(required_capabilities)

    def get_cost(self) -> float:
        """Compute total cost of this coalition.

        Returns:
            Sum of member costs.
        """
        return sum(self._agent_costs.get(m, 1.0) for m in self.members)

    def dissolve(self) -> None:
        """Dissolve this coalition."""
        self.status = CoalitionStatus.DISSOLVED
        logger.debug("Coalition %s dissolved", self.coalition_id)

    def merge(self, other: "Coalition") -> "Coalition":
        """Merge another coalition into this one.

        Args:
            other: The coalition to merge.

        Returns:
            A new merged coalition.
        """
        merged_members = list(set(self.members + other.members))
        merged = Coalition(
            coalition_id=f"{self.coalition_id}+{other.coalition_id}",
            members=merged_members,
        )
        merged._agent_costs = {**self._agent_costs, **other._agent_costs}
        logger.debug("Merged coalition %s with %s", self.coalition_id, other.coalition_id)
        return merged

    def split(self, agent_id: str) -> None:
        """Remove an agent from this coalition.

        Args:
            agent_id: The agent to remove.
        """
        if agent_id in self.members:
            self.members.remove(agent_id)
            logger.debug("Agent %s removed from coalition %s", agent_id, self.coalition_id)

    def handle_agent_failure(self, agent_id: str) -> None:
        """Handle an agent failure by removing it from the coalition.

        Args:
            agent_id: The failed agent.
        """
        if agent_id in self.members:
            self.members.remove(agent_id)
            logger.debug("Failed agent %s removed from coalition %s", agent_id, self.coalition_id)
        if not self.members:
            self.dissolve()
            logger.debug("Coalition %s dissolved (no members)", self.coalition_id)


class CoalitionFormation:
    """Forms coalitions of agents for multi-capability tasks.

    Uses a greedy set-cover approach to find minimal coalitions that
    cover all required capabilities for each task. Each agent can
    only be in one coalition at a time.
    """

    def __init__(
        self,
        agents: List[AgentProfile],
        tasks: List[TaskRequirement],
    ):
        """Initialize coalition formation.

        Args:
            agents: Available agents with their capabilities.
            tasks: Tasks with required capabilities.
        """
        self.agents = {a.agent_id: a for a in agents}
        self.tasks = tasks
        self._coalitions: List[Coalition] = []

    def form_coalitions(self) -> List[Coalition]:
        """Form coalitions for all tasks.

        Tasks are processed in priority order (highest first). For each task,
        a minimal coalition is formed using a greedy set-cover approach.
        Each agent can only be in one coalition.

        Returns:
            List of formed coalitions.
        """
        self._coalitions = []
        available_agents = set(self.agents.keys())
        agent_capabilities: Dict[str, Set[str]] = {
            aid: set(a.capabilities) for aid, a in self.agents.items()
        }

        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)

        for task in sorted_tasks:
            required = set(task.required_capabilities)
            if not required:
                continue

            coalition_members = self._greedy_set_cover(
                required, available_agents, agent_capabilities
            )

            if coalition_members:
                coalition_id = f"C{len(self._coalitions) + 1}"
                coalition = Coalition(
                    coalition_id=coalition_id,
                    members=list(coalition_members),
                )
                coalition._agent_costs = {m: self.agents[m].cost for m in coalition_members}
                coalition._capabilities = set().union(
                    *(agent_capabilities.get(m, set()) for m in coalition_members)
                )
                self._coalitions.append(coalition)
                available_agents -= coalition_members
                logger.debug(
                    "Formed coalition %s for task %s with members %s",
                    coalition_id,
                    task.task_id,
                    coalition_members,
                )

        return self._coalitions

    def _greedy_set_cover(
        self,
        required: Set[str],
        available_agents: Set[str],
        agent_capabilities: Dict[str, Set[str]],
    ) -> Set[str]:
        """Find a minimal set of agents that covers all required capabilities.

        Uses a greedy set-cover algorithm: iteratively pick the agent that
        covers the most uncovered required capabilities.

        Args:
            required: Set of required capabilities.
            available_agents: Set of available agent IDs.
            agent_capabilities: Mapping from agent ID to capabilities.

        Returns:
            Set of agent IDs forming the coalition.
        """
        uncovered = set(required)
        selected: Set[str] = set()

        while uncovered:
            best_agent = None
            best_coverage = 0

            for agent_id in available_agents - selected:
                caps = agent_capabilities.get(agent_id, set())
                coverage = len(uncovered & caps)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_agent = agent_id

            if best_agent is None or best_coverage == 0:
                break

            selected.add(best_agent)
            uncovered -= agent_capabilities.get(best_agent, set())

        return selected

    def get_coalitions(self) -> List[Coalition]:
        """Get all formed coalitions.

        Returns:
            List of coalitions.
        """
        return list(self._coalitions)

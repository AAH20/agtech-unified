"""Multi-agent fault tolerance: task reassignment and leader election."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about an agent in the fault-tolerance system."""

    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class TaskInfo:
    """Information about a task assigned to an agent."""

    task_id: str
    assigned_agent: Optional[str] = None
    requirements: List[str] = field(default_factory=list)


class TaskReassignment:
    """Handles task reassignment when an agent fails.

    When an agent becomes unavailable, its tasks are reassigned to
    other active agents that have the required capabilities.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, AgentInfo] = {}
        self._tasks: Dict[str, TaskInfo] = {}

    def register_agent(self, agent_id: str, capabilities: Optional[List[str]] = None) -> None:
        """Register an agent with the reassignment system.

        Args:
            agent_id: Unique identifier for the agent.
            capabilities: List of capabilities the agent provides.
        """
        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            capabilities=capabilities or [],
        )

    def assign_task(
        self, task_id: str, agent_id: str, requirements: Optional[List[str]] = None
    ) -> None:
        """Assign a task to an agent.

        Args:
            task_id: Unique task identifier.
            agent_id: The agent to assign the task to.
            requirements: Required capabilities for the task.
        """
        self._tasks[task_id] = TaskInfo(
            task_id=task_id,
            assigned_agent=agent_id,
            requirements=requirements or [],
        )

    def reassign_failed_agent(self, failed_agent_id: str) -> Dict[str, Optional[str]]:
        """Reassign all tasks from a failed agent to other capable agents.

        Args:
            failed_agent_id: The agent that has failed.

        Returns:
            Dictionary mapping task_id to the new agent_id (or None if
            no suitable agent was found).

        Raises:
            ValueError: If the failed_agent_id is not registered.
        """
        if failed_agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {failed_agent_id}")

        self._agents[failed_agent_id].is_active = False

        reassignments: Dict[str, Optional[str]] = {}
        for task_id, task in self._tasks.items():
            if task.assigned_agent == failed_agent_id:
                new_agent = self._find_replacement(task, exclude={failed_agent_id})
                task.assigned_agent = new_agent
                reassignments[task_id] = new_agent

        return reassignments

    def _find_replacement(self, task: TaskInfo, exclude: set) -> Optional[str]:
        """Find a replacement agent for a task, excluding given agents.

        Args:
            task: The task needing reassignment.
            exclude: Agent IDs to exclude from consideration.

        Returns:
            Best replacement agent ID, or None if none found.
        """
        best_agent: Optional[str] = None
        best_load = float("inf")

        for agent_id, agent in self._agents.items():
            if agent_id in exclude or not agent.is_active:
                continue
            if all(req in agent.capabilities for req in task.requirements):
                load = sum(1 for t in self._tasks.values() if t.assigned_agent == agent_id)
                if load < best_load:
                    best_load = load
                    best_agent = agent_id

        return best_agent


class LeaderElection:
    """Simple leader election for multi-agent coordination.

    Elects a leader from registered candidates. When the leader fails,
    a new leader is elected from the remaining candidates.
    """

    def __init__(self) -> None:
        self._candidates: List[str] = []
        self._leader: Optional[str] = None

    def register_candidate(self, agent_id: str) -> None:
        """Register a candidate for leader election.

        Args:
            agent_id: The agent to add as a candidate.
        """
        if agent_id not in self._candidates:
            self._candidates.append(agent_id)

    def remove_candidate(self, agent_id: str) -> None:
        """Remove a candidate (e.g., when the leader fails).

        Args:
            agent_id: The agent to remove.
        """
        if agent_id in self._candidates:
            self._candidates.remove(agent_id)
        if self._leader == agent_id:
            self._leader = None

    def elect_leader(self) -> str:
        """Elect a leader from the registered candidates.

        Returns:
            The elected leader's agent ID.

        Raises:
            ValueError: If there are no candidates.
        """
        if not self._candidates:
            raise ValueError("No candidates for leader election")

        # Simple election: pick the first candidate (deterministic)
        self._leader = self._candidates[0]
        return self._leader

    @property
    def leader(self) -> Optional[str]:
        """Get the current leader, or None if no leader elected."""
        return self._leader

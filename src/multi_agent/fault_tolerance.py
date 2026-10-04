"""Multi-agent fault tolerance: task reassignment and leader election."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about an agent in the fault-tolerance system."""

    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    is_active: bool = True
    health_score: float = 1.0
    error_count: int = 0
    success_count: int = 0
    last_heartbeat: float = field(default_factory=time.time)

    def record_error(self) -> None:
        """Record an error and degrade health score."""
        self.error_count += 1
        self.health_score = max(0.0, self.health_score - 0.1)

    def record_success(self) -> None:
        """Record a success and improve health score."""
        self.success_count += 1
        self.health_score = min(1.0, self.health_score + 0.05)

    def is_degraded(self) -> bool:
        """Check if agent health is degraded (below 0.5)."""
        return self.health_score < 0.5


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
    """Leader election for multi-agent coordination.

    Supports multiple election algorithms:
    - "simple": First candidate (deterministic, default)
    - "bully": Highest priority candidate wins

    When the leader fails, a new leader is elected from the remaining candidates.
    Heartbeat monitoring detects leader failure and triggers re-election.
    """

    def __init__(
        self,
        algorithm: str = "simple",
        heartbeat_timeout: float = 30.0,
        auto_reelect: bool = False,
    ) -> None:
        """Initialize leader election.

        Args:
            algorithm: Election algorithm ("simple" or "bully").
            heartbeat_timeout: Seconds without heartbeat before leader is
                considered failed.
            auto_reelect: If True, automatically re-elect when leader fails.
        """
        self._candidates: List[str] = []
        self._priorities: Dict[str, int] = {}
        self._leader: Optional[str] = None
        self._algorithm = algorithm
        self._heartbeat_timeout = heartbeat_timeout
        self._auto_reelect = auto_reelect
        self._last_leader_heartbeat: float = time.time()

    def register_candidate(self, agent_id: str, priority: int = 0) -> None:
        """Register a candidate for leader election.

        Args:
            agent_id: The agent to add as a candidate.
            priority: Priority for bully algorithm (higher = more likely to win).
        """
        if agent_id not in self._candidates:
            self._candidates.append(agent_id)
            self._priorities[agent_id] = priority

    def remove_candidate(self, agent_id: str) -> None:
        """Remove a candidate (e.g., when the leader fails).

        Args:
            agent_id: The agent to remove.
        """
        if agent_id in self._candidates:
            self._candidates.remove(agent_id)
            self._priorities.pop(agent_id, None)
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

        if self._algorithm == "bully":
            # Bully: highest priority wins
            self._leader = max(
                self._candidates,
                key=lambda aid: self._priorities.get(aid, 0),
            )
        else:
            # Simple: first candidate (deterministic)
            self._leader = self._candidates[0]

        self._last_leader_heartbeat = time.time()
        return self._leader

    def heartbeat(self, agent_id: str) -> None:
        """Record a heartbeat from the leader.

        Args:
            agent_id: The agent sending the heartbeat.

        Raises:
            ValueError: If the agent is not the current leader.
        """
        if agent_id != self._leader:
            raise ValueError(f"Agent '{agent_id}' is not the leader")
        self._last_leader_heartbeat = time.time()

    def is_leader_alive(self) -> bool:
        """Check if the leader is still alive based on heartbeat.

        Returns:
            True if the leader has sent a heartbeat within the timeout.
        """
        if self._leader is None:
            return False
        return (time.time() - self._last_leader_heartbeat) < self._heartbeat_timeout

    def check_leader_health(self) -> Optional[str]:
        """Check leader health and re-elect if necessary.

        Returns:
            The new leader ID if re-election occurred, None otherwise.
        """
        if self._leader is not None and not self.is_leader_alive():
            logger.warning("Leader %s failed (heartbeat timeout)", self._leader)
            old_leader = self._leader
            self._leader = None
            self.remove_candidate(old_leader)
            if self._auto_reelect and self._candidates:
                new_leader = self.elect_leader()
                logger.info("Re-elected leader: %s", new_leader)
                return new_leader
        return None

    @property
    def leader(self) -> Optional[str]:
        """Get the current leader, or None if no leader elected."""
        return self._leader

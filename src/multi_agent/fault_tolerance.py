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


class DeadlockDetector:
    """Detects deadlocks in multi-agent wait-for graphs.

    A deadlock occurs when there is a cycle in the wait-for graph,
    meaning agents are circularly waiting for each other.
    """

    def __init__(self) -> None:
        self._wait_for: Dict[str, List[str]] = {}

    def add_wait_edge(self, waiter: str, waited_for: str) -> None:
        """Add a wait-for edge: waiter is waiting for waited_for.

        Args:
            waiter: The agent that is waiting.
            waited_for: The agent being waited for.
        """
        if waiter not in self._wait_for:
            self._wait_for[waiter] = []
        if waited_for not in self._wait_for[waiter]:
            self._wait_for[waiter].append(waited_for)

    def remove_wait_edge(self, waiter: str, waited_for: str) -> None:
        """Remove a wait-for edge.

        Args:
            waiter: The agent that was waiting.
            waited_for: The agent that was being waited for.
        """
        if waiter in self._wait_for and waited_for in self._wait_for[waiter]:
            self._wait_for[waiter].remove(waited_for)

    def clear(self) -> None:
        """Clear all wait-for edges."""
        self._wait_for.clear()

    def detect_deadlock(self) -> bool:
        """Detect if there is a cycle in the wait-for graph.

        Uses DFS to detect cycles.

        Returns:
            True if a deadlock (cycle) is detected.
        """
        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

        def has_cycle(node: str) -> bool:
            """DFS to detect cycle from node."""
            visited[node] = 1  # visiting
            for neighbor in self._wait_for.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    return True  # Back edge = cycle
                if visited.get(neighbor, 0) == 0:
                    if has_cycle(neighbor):
                        return True
            visited[node] = 2  # visited
            return False

        for node in self._wait_for:
            if visited.get(node, 0) == 0:
                if has_cycle(node):
                    return True
        return False


class DeadlockResolver:
    """Resolves deadlocks by selecting a victim agent to break the cycle.

    Uses priority-based victim selection: the agent with the lowest
    priority is chosen as the victim to minimize impact.
    """

    def __init__(self) -> None:
        self._wait_for: Dict[str, List[str]] = {}
        self._priorities: Dict[str, int] = {}
        self._reassignment_callback = None

    def add_wait_edge(self, waiter: str, waited_for: str) -> None:
        """Add a wait-for edge.

        Args:
            waiter: The agent that is waiting.
            waited_for: The agent being waited for.
        """
        if waiter not in self._wait_for:
            self._wait_for[waiter] = []
        if waited_for not in self._wait_for[waiter]:
            self._wait_for[waiter].append(waited_for)

    def remove_wait_edge(self, waiter: str, waited_for: str) -> None:
        """Remove a wait-for edge.

        Args:
            waiter: The agent that was waiting.
            waited_for: The agent that was being waited for.
        """
        if waiter in self._wait_for and waited_for in self._wait_for[waiter]:
            self._wait_for[waiter].remove(waited_for)

    def set_priority(self, agent_id: str, priority: int) -> None:
        """Set priority for an agent (higher = more important).

        Args:
            agent_id: The agent to set priority for.
            priority: Priority value.
        """
        self._priorities[agent_id] = priority

    def set_reassignment_callback(self, callback) -> None:
        """Set callback for task reassignment when victim is selected.

        Args:
            callback: Function to call with victim agent_id.
        """
        self._reassignment_callback = callback

    def detect_deadlock(self) -> bool:
        """Detect if there is a cycle in the wait-for graph.

        Returns:
            True if a deadlock is detected.
        """
        visited: Dict[str, int] = {}

        def has_cycle(node: str) -> bool:
            visited[node] = 1
            for neighbor in self._wait_for.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    return True
                if visited.get(neighbor, 0) == 0:
                    if has_cycle(neighbor):
                        return True
            visited[node] = 2
            return False

        for node in self._wait_for:
            if visited.get(node, 0) == 0:
                if has_cycle(node):
                    return True
        return False

    def resolve_deadlock(self) -> Optional[str]:
        """Resolve deadlock by selecting a victim agent.

        The victim is the agent with the lowest priority in the cycle.
        If priorities are equal, selects deterministically (first found).

        Returns:
            The victim agent ID, or None if no deadlock.
        """
        if not self.detect_deadlock():
            return None

        # Find all agents in cycles
        cycle_agents = self._find_cycle_agents()
        if not cycle_agents:
            return None

        # Select victim: lowest priority (or first if tied)
        victim = min(cycle_agents, key=lambda a: self._priorities.get(a, 0))

        # Trigger reassignment callback if set
        if self._reassignment_callback is not None:
            self._reassignment_callback(victim)

        logger.info("Deadlock resolved by selecting victim: %s", victim)
        return victim

    def resolve_and_recover(self) -> Optional[str]:
        """Detect deadlock, select victim, and break the cycle.

        This is the full recovery action: it removes the victim's outgoing
        wait-for edges to break the cycle, then triggers the reassignment
        callback so the victim's tasks are reassigned to other agents.

        Returns:
            The victim agent ID, or None if no deadlock was found.
        """
        if not self.detect_deadlock():
            return None

        cycle_agents = self._find_cycle_agents()
        if not cycle_agents:
            return None

        victim = min(cycle_agents, key=lambda a: self._priorities.get(a, 0))

        # Break the cycle: remove all outgoing edges from the victim
        edges_to_remove = list(self._wait_for.get(victim, []))
        for waited_for in edges_to_remove:
            self.remove_wait_edge(victim, waited_for)

        # Trigger reassignment callback
        if self._reassignment_callback is not None:
            self._reassignment_callback(victim)

        logger.info(
            "Deadlock resolved: victim=%s, removed %d edge(s)",
            victim,
            len(edges_to_remove),
        )
        return victim

    def get_cycle_path(self) -> Optional[List[str]]:
        """Return the actual cycle path if a deadlock exists.

        Returns:
            List of agent IDs forming the cycle (e.g., ["A", "B", "C"]),
            or None if no deadlock.
        """
        visited: Dict[str, int] = {}
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited[node] = 1
            path.append(node)
            for neighbor in self._wait_for.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
                if visited.get(neighbor, 0) == 0:
                    result = dfs(neighbor)
                    if result is not None:
                        return result
            path.pop()
            visited[node] = 2
            return None

        for node in self._wait_for:
            if visited.get(node, 0) == 0:
                result = dfs(node)
                if result is not None:
                    return result
        return None

    def _find_cycle_agents(self) -> List[str]:
        """Find all agents that are part of a cycle.

        Returns:
            List of agent IDs in cycles.
        """
        visited: Dict[str, int] = {}
        in_cycle: Dict[str, bool] = {}

        def dfs(node: str, path: List[str]) -> None:
            visited[node] = 1
            path.append(node)
            for neighbor in self._wait_for.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    # Found cycle - mark all agents in the cycle
                    cycle_start = path.index(neighbor)
                    for agent in path[cycle_start:]:
                        in_cycle[agent] = True
                elif visited.get(neighbor, 0) == 0:
                    dfs(neighbor, path)
            path.pop()
            visited[node] = 2

        for node in self._wait_for:
            if visited.get(node, 0) == 0:
                dfs(node, [])

        return [agent for agent, is_cycle in in_cycle.items() if is_cycle]

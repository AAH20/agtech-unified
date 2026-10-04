"""Multi-agent swarm coordination for agricultural robots."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.multi_agent.collision_avoidance import CollisionAvoidance, Position
from src.multi_agent.fault_tolerance import LeaderElection, TaskReassignment

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of a swarm agent."""

    ACTIVE = "active"
    FAILED = "failed"
    BUSY = "busy"


class TaskStatus(Enum):
    """Status of a swarm task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Agent:
    """A swarm agent (e.g., agricultural robot)."""

    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.ACTIVE
    last_heartbeat: float = field(default_factory=time.time)
    assigned_tasks: List[str] = field(default_factory=list)
    position: Optional[Position] = None
    priority: int = 0


@dataclass
class Task:
    """A task to be executed by the swarm."""

    task_id: str
    requirements: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)


class SwarmCoordinator:
    """Coordinates a swarm of agricultural robots.

    Handles agent registration, heartbeat monitoring, task distribution,
    and failure detection for multi-agent agricultural operations.
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        max_tasks_per_agent: int = 10,
        safety_radius: float = 2.0,
        max_avoidance_speed: float = 1.0,
    ):
        """Initialize the swarm coordinator.

        Args:
            heartbeat_timeout: Seconds without heartbeat before agent is
                marked as failed.
            max_tasks_per_agent: Maximum number of tasks per agent.
            safety_radius: Minimum safe distance between agents.
            max_avoidance_speed: Maximum speed of avoidance velocity.
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.max_tasks_per_agent = max_tasks_per_agent
        self._agents: Dict[str, Agent] = {}
        self._tasks: Dict[str, Task] = {}
        self._collision_avoidance = CollisionAvoidance(
            safety_radius=safety_radius,
            max_avoidance_speed=max_avoidance_speed,
        )
        self._leader_election = LeaderElection()
        self._task_reassignment = TaskReassignment()
        self._bus: Optional[EventBus] = None

    def register_agent(
        self,
        agent_id: str,
        capabilities: Optional[List[str]] = None,
        priority: int = 0,
    ) -> Agent:
        """Register a new agent with the swarm.

        Args:
            agent_id: Unique identifier for the agent.
            capabilities: List of capabilities (e.g., ['spray', 'scan']).
            priority: Agent priority for leader election.

        Returns:
            The registered Agent instance.

        Raises:
            ValueError: If agent_id is already registered.
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent '{agent_id}' already registered")
        agent = Agent(
            agent_id=agent_id,
            capabilities=capabilities or [],
            priority=priority,
        )
        self._agents[agent_id] = agent
        # Register with leader election and task reassignment
        self._leader_election.register_candidate(agent_id)
        self._task_reassignment.register_agent(agent_id, capabilities or [])
        logger.info("Registered agent %s with capabilities %s", agent_id, capabilities)
        return agent

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the swarm.

        Args:
            agent_id: The agent to remove.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info("Unregistered agent %s", agent_id)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID.

        Args:
            agent_id: The agent to look up.

        Returns:
            The Agent instance, or None if not found.
        """
        return self._agents.get(agent_id)

    def heartbeat(self, agent_id: str) -> None:
        """Record a heartbeat from an agent.

        Args:
            agent_id: The agent sending the heartbeat.

        Raises:
            ValueError: If the agent is not registered.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        self._agents[agent_id].last_heartbeat = time.time()
        logger.debug("Heartbeat from agent %s", agent_id)

    def check_health(self) -> List[str]:
        """Check health of all agents based on heartbeat timeout.

        When an agent is marked as failed, its tasks are reassigned to
        other active agents with matching capabilities.

        Returns:
            List of agent IDs that were marked as failed.
        """
        now = time.time()
        failed = []
        for agent_id, agent in self._agents.items():
            if agent.status == AgentStatus.ACTIVE:
                if now - agent.last_heartbeat > self.heartbeat_timeout:
                    agent.status = AgentStatus.FAILED
                    failed.append(agent_id)
                    logger.warning("Agent %s marked as failed (no heartbeat)", agent_id)
                    # Reassign tasks from failed agent
                    self._reassign_agent_tasks(agent_id)
        return failed

    def _reassign_agent_tasks(self, failed_agent_id: str) -> None:
        """Reassign all tasks from a failed agent to other active agents.

        Args:
            failed_agent_id: The agent that has failed.
        """
        failed_agent = self._agents[failed_agent_id]
        tasks_to_reassign = list(failed_agent.assigned_tasks)
        failed_agent.assigned_tasks.clear()

        for task_id in tasks_to_reassign:
            task = self._tasks.get(task_id)
            if task is None:
                continue
            task.status = TaskStatus.PENDING
            task.assigned_agent = None

            # Find a replacement agent
            active_agents = {
                aid: a
                for aid, a in self._agents.items()
                if a.status == AgentStatus.ACTIVE and aid != failed_agent_id
            }
            best_agent = self._find_best_agent(task, active_agents, {})
            if best_agent is not None:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = best_agent
                active_agents[best_agent].assigned_tasks.append(task_id)
                logger.info(
                    "Task %s reassigned from failed agent %s to agent %s",
                    task_id,
                    failed_agent_id,
                    best_agent,
                )
            else:
                logger.warning(
                    "No replacement found for task %s from failed agent %s",
                    task_id,
                    failed_agent_id,
                )

    def get_failed_agents(self) -> List[str]:
        """Get list of failed agent IDs.

        Returns:
            List of agent IDs with FAILED status.
        """
        return [aid for aid, a in self._agents.items() if a.status == AgentStatus.FAILED]

    def submit_task(
        self,
        task_id: str,
        requirements: Optional[List[str]] = None,
        priority: int = 0,
        depends_on: Optional[List[str]] = None,
    ) -> Task:
        """Submit a task to the swarm.

        Args:
            task_id: Unique task identifier.
            requirements: Required capabilities for the task.
            priority: Task priority (higher = more urgent).
            depends_on: List of task IDs that must complete before this task.

        Returns:
            The created Task instance.
        """
        task = Task(
            task_id=task_id,
            requirements=requirements or [],
            priority=priority,
            depends_on=depends_on or [],
        )
        self._tasks[task_id] = task
        logger.info(
            "Submitted task %s with requirements %s priority %d depends_on %s",
            task_id,
            requirements,
            priority,
            depends_on,
        )
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: The task to look up.

        Returns:
            The Task instance, or None if not found.
        """
        return self._tasks.get(task_id)

    def distribute_tasks(self) -> Dict[str, str]:
        """Distribute pending tasks to available agents.

        Matches tasks to agents based on capability requirements.
        Only distributes to ACTIVE agents. Tasks are processed in
        priority order (highest first). Tasks with unmet dependencies
        are skipped. Agents with capacity limits are not overloaded.
        Higher priority tasks can preempt lower priority ones.

        Returns:
            Dictionary mapping task_id to assigned agent_id.
        """
        assignments: Dict[str, str] = {}
        active_agents = {
            aid: a for aid, a in self._agents.items() if a.status == AgentStatus.ACTIVE
        }

        # Collect all tasks that need assignment (pending + preempted)
        all_tasks = list(self._tasks.values())

        # Sort by priority (highest first)
        all_tasks.sort(key=lambda t: t.priority, reverse=True)

        for task in all_tasks:
            # Skip completed and failed tasks
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                continue

            # Skip tasks with unmet dependencies
            if not self._dependencies_met(task):
                logger.debug("Task %s skipped: dependencies not met", task.task_id)
                continue

            # If task is already assigned, skip (preemption handled below)
            if task.status == TaskStatus.ASSIGNED:
                continue

            best_agent = self._find_best_agent(task, active_agents, assignments)
            if best_agent is not None:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = best_agent
                active_agents[best_agent].assigned_tasks.append(task.task_id)
                assignments[task.task_id] = best_agent
                if self._bus:
                    self._bus.publish(
                        DomainEvent(
                            event_type=EventType.TASK_ASSIGNED,
                            source="multi_agent.swarm",
                            payload={
                                "task_id": task.task_id,
                                "assigned_agent": best_agent,
                            },
                        )
                    )
                logger.info(
                    "Assigned task %s to agent %s (priority %d)",
                    task.task_id,
                    best_agent,
                    task.priority,
                )
            else:
                # Try to preempt a lower priority task
                preempted = self._try_preempt(task, active_agents, assignments)
                if preempted:
                    logger.info("Task %s preempted a lower priority task", task.task_id)

        return assignments

    def _try_preempt(
        self,
        high_priority_task: Task,
        active_agents: Dict[str, Agent],
        current_assignments: Dict[str, str],
    ) -> bool:
        """Try to preempt a lower priority task for a higher priority task.

        Args:
            high_priority_task: The high priority task needing assignment.
            active_agents: Currently active agents.
            current_assignments: Assignments made in this distribution round.

        Returns:
            True if preemption occurred.
        """
        # Find the lowest priority assigned task that can be preempted
        preemptable = []
        for t in self._tasks.values():
            if t.status != TaskStatus.ASSIGNED:
                continue
            if t.priority >= high_priority_task.priority:
                continue
            if t.assigned_agent is None:
                continue
            agent = active_agents.get(t.assigned_agent)
            if agent is None:
                continue
            if self._can_handle(agent, high_priority_task):
                preemptable.append(t)

        if not preemptable:
            return False

        # Preempt the lowest priority task
        victim = min(preemptable, key=lambda t: t.priority)
        old_agent = victim.assigned_agent
        if old_agent is None:
            return False

        # Remove victim from agent
        if old_agent in active_agents:
            agent = active_agents[old_agent]
            if victim.task_id in agent.assigned_tasks:
                agent.assigned_tasks.remove(victim.task_id)

        # Reset victim
        victim.status = TaskStatus.PENDING
        victim.assigned_agent = None

        # Assign high priority task
        high_priority_task.status = TaskStatus.ASSIGNED
        high_priority_task.assigned_agent = old_agent
        if old_agent in active_agents:
            active_agents[old_agent].assigned_tasks.append(high_priority_task.task_id)
        current_assignments[high_priority_task.task_id] = old_agent

        logger.info(
            "Task %s (priority %d) preempted task %s (priority %d) on agent %s",
            high_priority_task.task_id,
            high_priority_task.priority,
            victim.task_id,
            victim.priority,
            old_agent,
        )
        return True

    def _find_best_agent(
        self,
        task: Task,
        active_agents: Dict[str, Agent],
        current_assignments: Dict[str, str],
    ) -> Optional[str]:
        """Find the best agent for a task based on capabilities and load.

        Args:
            task: The task to assign.
            active_agents: Currently active agents.
            current_assignments: Assignments made in this distribution round.

        Returns:
            Best agent ID or None if no suitable agent found.
        """
        best_agent = None
        best_load = float("inf")

        for agent_id, agent in active_agents.items():
            if self._can_handle(agent, task):
                load = len(agent.assigned_tasks)
                # Respect capacity limit
                if load >= self.max_tasks_per_agent:
                    continue
                if load < best_load:
                    best_load = load
                    best_agent = agent_id

        return best_agent

    def _can_handle(self, agent: Agent, task: Task) -> bool:
        """Check if an agent has all required capabilities for a task.

        Args:
            agent: The agent to check.
            task: The task with requirements.

        Returns:
            True if the agent can handle the task.
        """
        return all(req in agent.capabilities for req in task.requirements)

    # ── Leader election integration ──────────────────────────────────

    def elect_leader(self) -> str:
        """Elect a leader from registered agents.

        Only active agents are considered as candidates.

        Returns:
            The elected leader's agent ID.

        Raises:
            ValueError: If there are no candidates.
        """
        self._leader_election = LeaderElection()
        for agent_id, agent in self._agents.items():
            if agent.status == AgentStatus.ACTIVE:
                self._leader_election.register_candidate(agent_id, priority=agent.priority)
        leader = self._leader_election.elect_leader()
        logger.info("Elected leader: %s", leader)
        return leader

    def get_leader(self) -> Optional[str]:
        """Get the current leader, or None if no leader elected."""
        return self._leader_election.leader

    # ── Collision avoidance integration ──────────────────────────────

    def set_agent_position(self, agent_id: str, position: Position) -> None:
        """Set the position of an agent.

        Args:
            agent_id: The agent to position.
            position: The agent's position.

        Raises:
            ValueError: If the agent is not registered.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        self._agents[agent_id].position = position

    def detect_collisions(self) -> List[Tuple[str, str]]:
        """Detect collisions between agents based on their positions.

        Returns:
            List of (agent_id_1, agent_id_2) tuples for colliding pairs.
        """
        positions = {aid: a.position for aid, a in self._agents.items() if a.position is not None}
        return self._collision_avoidance.detect_collisions(positions)

    # ── Task completion / failure reporting ──────────────────────────

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed.

        Args:
            task_id: The task to complete.

        Raises:
            ValueError: If the task is unknown or not assigned.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        if task.status != TaskStatus.ASSIGNED:
            raise ValueError(f"Task '{task_id}' is not assigned")
        task.status = TaskStatus.COMPLETED
        # Remove from agent's assigned tasks
        if task.assigned_agent and task.assigned_agent in self._agents:
            agent = self._agents[task.assigned_agent]
            if task_id in agent.assigned_tasks:
                agent.assigned_tasks.remove(task_id)
        if self._bus:
            self._bus.publish(
                DomainEvent(
                    event_type=EventType.TASK_COMPLETED,
                    source="multi_agent.swarm",
                    payload={
                        "task_id": task_id,
                        "assigned_agent": task.assigned_agent,
                    },
                )
            )
        logger.info("Task %s completed", task_id)

    def fail_task(self, task_id: str, retry: bool = True) -> None:
        """Mark a task as failed.

        Args:
            task_id: The task to fail.
            retry: If True, reassign the task to another agent.

        Raises:
            ValueError: If the task is unknown.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        task = self._tasks[task_id]
        old_agent = task.assigned_agent
        task.status = TaskStatus.FAILED
        # Remove from agent's assigned tasks
        if old_agent and old_agent in self._agents:
            agent = self._agents[old_agent]
            if task_id in agent.assigned_tasks:
                agent.assigned_tasks.remove(task_id)
        task.assigned_agent = None

        if retry:
            # Try to reassign to another active agent
            active_agents = {
                aid: a
                for aid, a in self._agents.items()
                if a.status == AgentStatus.ACTIVE and aid != old_agent
            }
            best_agent = self._find_best_agent(task, active_agents, {})
            if best_agent is not None:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = best_agent
                active_agents[best_agent].assigned_tasks.append(task_id)
                logger.info("Task %s reassigned to agent %s after failure", task_id, best_agent)
            else:
                task.status = TaskStatus.PENDING
                logger.warning("Task %s failed and no replacement found", task_id)
        else:
            logger.warning("Task %s failed (no retry)", task_id)

    # ── Agent status recovery ────────────────────────────────────────

    def set_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Set the status of an agent.

        Args:
            agent_id: The agent to update.
            status: The new status.

        Raises:
            ValueError: If the agent is not registered.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        self._agents[agent_id].status = status
        logger.info("Agent %s status set to %s", agent_id, status)

    def recover_agent(self, agent_id: str) -> None:
        """Recover a failed agent back to active status.

        Args:
            agent_id: The agent to recover.

        Raises:
            ValueError: If the agent is not registered or not failed.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        agent = self._agents[agent_id]
        if agent.status != AgentStatus.FAILED:
            raise ValueError(f"Agent '{agent_id}' is not failed")
        agent.status = AgentStatus.ACTIVE
        agent.last_heartbeat = time.time()
        agent.assigned_tasks.clear()
        logger.info("Agent %s recovered", agent_id)

    # ── Task dependencies ────────────────────────────────────────────

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks sorted by priority (highest first).

        Returns:
            List of pending tasks in priority order.
        """
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return sorted(pending, key=lambda t: t.priority, reverse=True)

    def _dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies of a task are completed.

        Args:
            task: The task to check.

        Returns:
            True if all dependencies are completed.
        """
        for dep_id in task.depends_on:
            dep_task = self._tasks.get(dep_id)
            if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

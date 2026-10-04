"""Multi-agent swarm coordination for agricultural robots."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import logging

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


@dataclass
class Task:
    """A task to be executed by the swarm."""
    task_id: str
    requirements: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None


class SwarmCoordinator:
    """Coordinates a swarm of agricultural robots.

    Handles agent registration, heartbeat monitoring, task distribution,
    and failure detection for multi-agent agricultural operations.
    """

    def __init__(self, heartbeat_timeout: float = 30.0):
        """Initialize the swarm coordinator.

        Args:
            heartbeat_timeout: Seconds without heartbeat before agent is
                marked as failed.
        """
        self.heartbeat_timeout = heartbeat_timeout
        self._agents: Dict[str, Agent] = {}
        self._tasks: Dict[str, Task] = {}

    def register_agent(self, agent_id: str, capabilities: Optional[List[str]] = None) -> Agent:
        """Register a new agent with the swarm.

        Args:
            agent_id: Unique identifier for the agent.
            capabilities: List of capabilities (e.g., ['spray', 'scan']).

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
        )
        self._agents[agent_id] = agent
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
        return failed

    def get_failed_agents(self) -> List[str]:
        """Get list of failed agent IDs.

        Returns:
            List of agent IDs with FAILED status.
        """
        return [aid for aid, a in self._agents.items() if a.status == AgentStatus.FAILED]

    def submit_task(self, task_id: str, requirements: Optional[List[str]] = None) -> Task:
        """Submit a task to the swarm.

        Args:
            task_id: Unique task identifier.
            requirements: Required capabilities for the task.

        Returns:
            The created Task instance.
        """
        task = Task(
            task_id=task_id,
            requirements=requirements or [],
        )
        self._tasks[task_id] = task
        logger.info("Submitted task %s with requirements %s", task_id, requirements)
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
        Only distributes to ACTIVE agents.

        Returns:
            Dictionary mapping task_id to assigned agent_id.
        """
        assignments: Dict[str, str] = {}
        active_agents = {
            aid: a for aid, a in self._agents.items()
            if a.status == AgentStatus.ACTIVE
        }

        for task_id, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            best_agent = self._find_best_agent(task, active_agents, assignments)
            if best_agent is not None:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent = best_agent
                active_agents[best_agent].assigned_tasks.append(task_id)
                assignments[task_id] = best_agent
                logger.info("Assigned task %s to agent %s", task_id, best_agent)

        return assignments

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

"""Test multi-agent swarm coordination for agricultural robots."""

import time

import pytest

from src.multi_agent.swarm import (
    AgentStatus,
    SwarmCoordinator,
    TaskStatus,
)


def test_swarm_register_agent():
    """Registering an agent adds it to the swarm."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray", "scan"])
    agent = coordinator.get_agent("drone_1")
    assert agent.agent_id == "drone_1"
    assert agent.status == AgentStatus.ACTIVE
    assert "spray" in agent.capabilities


def test_swarm_unregister_agent():
    """Unregistering an agent removes it from the swarm."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.unregister_agent("drone_1")
    assert coordinator.get_agent("drone_1") is None


def test_swarm_duplicate_registration_raises():
    """Registering the same agent twice raises ValueError."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    with pytest.raises(ValueError, match="already registered"):
        coordinator.register_agent("drone_1", capabilities=["scan"])


def test_swarm_heartbeat_updates_timestamp():
    """Heartbeat updates the agent's last heartbeat time."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    agent = coordinator.get_agent("drone_1")
    old_time = agent.last_heartbeat
    time.sleep(0.01)
    coordinator.heartbeat("drone_1")
    new_time = coordinator.get_agent("drone_1").last_heartbeat
    assert new_time > old_time


def test_swarm_heartbeat_unknown_agent_raises():
    """Heartbeat for unknown agent raises ValueError."""
    coordinator = SwarmCoordinator()
    with pytest.raises(ValueError, match="Unknown agent"):
        coordinator.heartbeat("nonexistent")


def test_swarm_detect_failed_agent():
    """Agent that hasn't sent heartbeat within timeout is marked failed."""
    coordinator = SwarmCoordinator(heartbeat_timeout=0.05)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    time.sleep(0.1)
    coordinator.check_health()
    agent = coordinator.get_agent("drone_1")
    assert agent.status == AgentStatus.FAILED


def test_swarm_active_agent_not_failed():
    """Agent with recent heartbeat stays active."""
    coordinator = SwarmCoordinator(heartbeat_timeout=10.0)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.check_health()
    agent = coordinator.get_agent("drone_1")
    assert agent.status == AgentStatus.ACTIVE


def test_swarm_submit_task():
    """Submitting a task adds it to the task queue."""
    coordinator = SwarmCoordinator()
    coordinator.submit_task("task_1", requirements=["spray"])
    task = coordinator.get_task("task_1")
    assert task.task_id == "task_1"
    assert task.status == TaskStatus.PENDING


def test_swarm_distribute_tasks():
    """Tasks are distributed to agents with matching capabilities."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["scan"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.submit_task("task_2", requirements=["scan"])
    assignments = coordinator.distribute_tasks()
    assert "task_1" in assignments
    assert "task_2" in assignments
    assert assignments["task_1"] == "drone_1"
    assert assignments["task_2"] == "drone_2"


def test_swarm_distribute_no_matching_agent():
    """Task with no matching agent stays unassigned."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["harvest"])
    assignments = coordinator.distribute_tasks()
    assert "task_1" not in assignments


def test_swarm_get_failed_agents():
    """get_failed_agents returns only failed agents."""
    coordinator = SwarmCoordinator(heartbeat_timeout=0.05)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["scan"])
    time.sleep(0.1)
    coordinator.check_health()
    failed = coordinator.get_failed_agents()
    assert "drone_1" in failed
    assert "drone_2" in failed

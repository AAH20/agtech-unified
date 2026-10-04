"""Test multi-agent swarm integration: fault tolerance, leader election, collision avoidance."""

import time

import pytest

from src.multi_agent.collision_avoidance import Position
from src.multi_agent.swarm import (
    AgentStatus,
    SwarmCoordinator,
    TaskStatus,
)

# ── MA-SW-001: Task reassignment on agent failure ─────────────────────


def test_swarm_reassign_tasks_on_agent_failure():
    """When an agent fails, its tasks are reassigned to other capable agents."""
    coordinator = SwarmCoordinator(heartbeat_timeout=1.0)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    # Verify initial assignment
    assert coordinator.get_task("task_1").assigned_agent == "drone_1"

    # Simulate agent failure
    time.sleep(1.1)
    # Keep drone_2 alive right before health check
    coordinator.heartbeat("drone_2")
    coordinator.check_health()
    assert coordinator.get_agent("drone_1").status == AgentStatus.FAILED

    # Tasks should be reassigned
    task = coordinator.get_task("task_1")
    assert task.status == TaskStatus.ASSIGNED
    assert task.assigned_agent == "drone_2"


def test_swarm_reassign_no_capable_agent():
    """When no other agent can handle the task, it stays unassigned."""
    coordinator = SwarmCoordinator(heartbeat_timeout=0.05)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["scan"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    time.sleep(0.1)
    coordinator.check_health()

    task = coordinator.get_task("task_1")
    # Task should be back to pending since no one else can handle it
    assert task.status == TaskStatus.PENDING
    assert task.assigned_agent is None


# ── MA-SW-002: Leader election integration ────────────────────────────


def test_swarm_leader_election_integration():
    """SwarmCoordinator elects a leader from registered agents."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["scan"])

    leader = coordinator.elect_leader()
    assert leader in ("drone_1", "drone_2")
    assert coordinator.get_leader() == leader


def test_swarm_leader_reelection_on_failure():
    """When the leader fails, a new leader is elected."""
    coordinator = SwarmCoordinator(heartbeat_timeout=1.0)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["scan"])

    first_leader = coordinator.elect_leader()
    assert first_leader is not None

    # Simulate leader failure
    time.sleep(1.1)
    # Keep non-leader alive right before health check
    if first_leader == "drone_1":
        coordinator.heartbeat("drone_2")
    else:
        coordinator.heartbeat("drone_1")
    coordinator.check_health()

    # Re-elect
    second_leader = coordinator.elect_leader()
    assert second_leader is not None
    assert second_leader != first_leader


def test_swarm_leader_election_no_agents_raises():
    """Electing a leader with no agents raises ValueError."""
    coordinator = SwarmCoordinator()
    with pytest.raises(ValueError, match="No candidates"):
        coordinator.elect_leader()


# ── MA-SW-003: Collision avoidance integration ────────────────────────


def test_swarm_collision_avoidance_integration():
    """SwarmCoordinator uses collision avoidance to prevent spatial conflicts."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["spray"])

    # Set positions close together
    coordinator.set_agent_position("drone_1", Position(x=0.0, y=0.0))
    coordinator.set_agent_position("drone_2", Position(x=1.0, y=0.0))

    # Check for collisions
    collisions = coordinator.detect_collisions()
    assert len(collisions) > 0


def test_swarm_collision_avoidance_no_collision_when_far():
    """No collision when agents are far apart."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["spray"])

    coordinator.set_agent_position("drone_1", Position(x=0.0, y=0.0))
    coordinator.set_agent_position("drone_2", Position(x=100.0, y=100.0))

    collisions = coordinator.detect_collisions()
    assert len(collisions) == 0


# ── MA-SW-021: Task completion reporting ───────────────────────────────


def test_swarm_complete_task():
    """complete_task() marks a task as completed."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    coordinator.complete_task("task_1")
    task = coordinator.get_task("task_1")
    assert task.status == TaskStatus.COMPLETED


def test_swarm_complete_task_not_assigned_raises():
    """Completing a task that is not assigned raises ValueError."""
    coordinator = SwarmCoordinator()
    coordinator.submit_task("task_1", requirements=["spray"])
    with pytest.raises(ValueError, match="not assigned"):
        coordinator.complete_task("task_1")


def test_swarm_complete_task_unknown_raises():
    """Completing an unknown task raises ValueError."""
    coordinator = SwarmCoordinator()
    with pytest.raises(ValueError, match="Unknown task"):
        coordinator.complete_task("nonexistent")


# ── MA-SW-022: Task failure reporting ─────────────────────────────────


def test_swarm_fail_task():
    """fail_task() marks a task as failed."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    coordinator.fail_task("task_1", retry=False)
    task = coordinator.get_task("task_1")
    assert task.status == TaskStatus.FAILED


def test_swarm_fail_task_with_retry():
    """fail_task() with retry reassigns the task to another agent."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    assert coordinator.get_task("task_1").assigned_agent == "drone_1"

    coordinator.fail_task("task_1", retry=True)
    task = coordinator.get_task("task_1")
    assert task.status == TaskStatus.ASSIGNED
    assert task.assigned_agent == "drone_2"


def test_swarm_fail_task_no_retry():
    """fail_task() without retry leaves task in FAILED state."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    coordinator.fail_task("task_1", retry=False)
    task = coordinator.get_task("task_1")
    assert task.status == TaskStatus.FAILED


# ── MA-SW-023: Agent status recovery ──────────────────────────────────


def test_swarm_set_agent_status():
    """set_agent_status() transitions agent state."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])

    coordinator.set_agent_status("drone_1", AgentStatus.BUSY)
    assert coordinator.get_agent("drone_1").status == AgentStatus.BUSY

    coordinator.set_agent_status("drone_1", AgentStatus.ACTIVE)
    assert coordinator.get_agent("drone_1").status == AgentStatus.ACTIVE


def test_swarm_recover_failed_agent():
    """A failed agent can be recovered back to active."""
    coordinator = SwarmCoordinator(heartbeat_timeout=0.05)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    time.sleep(0.1)
    coordinator.check_health()
    assert coordinator.get_agent("drone_1").status == AgentStatus.FAILED

    coordinator.recover_agent("drone_1")
    assert coordinator.get_agent("drone_1").status == AgentStatus.ACTIVE


def test_swarm_recover_agent_not_failed_raises():
    """Recovering a non-failed agent raises ValueError."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    with pytest.raises(ValueError, match="not failed"):
        coordinator.recover_agent("drone_1")


# ── MA-SW-012: Agent recovery mechanism ───────────────────────────────


def test_swarm_agent_recovery_resets_tasks():
    """Recovering an agent clears its task assignments."""
    coordinator = SwarmCoordinator(heartbeat_timeout=0.05)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.distribute_tasks()

    time.sleep(0.1)
    coordinator.check_health()

    coordinator.recover_agent("drone_1")
    agent = coordinator.get_agent("drone_1")
    assert agent.status == AgentStatus.ACTIVE
    assert len(agent.assigned_tasks) == 0


# ── MA-SW-006: Priority-based task scheduling ─────────────────────────


def test_swarm_task_priority():
    """Tasks with higher priority are assigned first."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])

    coordinator.submit_task("low_priority", requirements=["spray"], priority=1)
    coordinator.submit_task("high_priority", requirements=["spray"], priority=10)

    assignments = coordinator.distribute_tasks()
    # High priority task should be assigned
    assert "high_priority" in assignments


def test_swarm_task_priority_ordering():
    """Tasks are ordered by priority in the task queue."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["spray"])

    coordinator.submit_task("low", requirements=["spray"], priority=1)
    coordinator.submit_task("high", requirements=["spray"], priority=10)
    coordinator.submit_task("medium", requirements=["spray"], priority=5)

    pending = coordinator.get_pending_tasks()
    priorities = [t.priority for t in pending]
    assert priorities == sorted(priorities, reverse=True)


# ── MA-SW-010: Task dependencies ──────────────────────────────────────


def test_swarm_task_dependencies():
    """Tasks with dependencies are only assigned after dependencies complete."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray", "scan"])

    coordinator.submit_task("task_a", requirements=["spray"])
    coordinator.submit_task("task_b", requirements=["scan"], depends_on=["task_a"])

    assignments = coordinator.distribute_tasks()
    # task_a should be assigned, task_b should not (dependency not met)
    assert "task_a" in assignments
    assert "task_b" not in assignments


def test_swarm_task_dependencies_met():
    """Tasks with met dependencies are assigned."""
    coordinator = SwarmCoordinator()
    coordinator.register_agent("drone_1", capabilities=["spray", "scan"])

    coordinator.submit_task("task_a", requirements=["spray"])
    coordinator.submit_task("task_b", requirements=["scan"], depends_on=["task_a"])

    coordinator.distribute_tasks()
    coordinator.complete_task("task_a")

    # Now task_b should be assignable
    assignments = coordinator.distribute_tasks()
    assert "task_b" in assignments


# ── MA-SW-008: Resource constraints ───────────────────────────────────


def test_swarm_agent_capacity_limit():
    """Agents have a maximum task capacity."""
    coordinator = SwarmCoordinator(max_tasks_per_agent=2)
    coordinator.register_agent("drone_1", capabilities=["spray"])
    coordinator.register_agent("drone_2", capabilities=["spray"])

    coordinator.submit_task("task_1", requirements=["spray"])
    coordinator.submit_task("task_2", requirements=["spray"])
    coordinator.submit_task("task_3", requirements=["spray"])

    assignments = coordinator.distribute_tasks()
    # Only 2 tasks should be assigned to drone_1 (capacity limit)
    drone_1_tasks = [t for t, a in assignments.items() if a == "drone_1"]
    assert len(drone_1_tasks) <= 2


# ── MA-SW-005: Task preemption ────────────────────────────────────────


def test_swarm_task_preemption():
    """Higher priority tasks can preempt lower priority ones."""
    coordinator = SwarmCoordinator(max_tasks_per_agent=1)
    coordinator.register_agent("drone_1", capabilities=["spray"])

    coordinator.submit_task("low_priority", requirements=["spray"], priority=1)
    coordinator.distribute_tasks()

    assert coordinator.get_task("low_priority").status == TaskStatus.ASSIGNED

    # Submit high priority task
    coordinator.submit_task("high_priority", requirements=["spray"], priority=10)
    coordinator.distribute_tasks()

    # Low priority task should be preempted
    assert coordinator.get_task("low_priority").status == TaskStatus.PENDING
    assert coordinator.get_task("high_priority").status == TaskStatus.ASSIGNED

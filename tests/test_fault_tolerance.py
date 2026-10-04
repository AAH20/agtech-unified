"""Test multi-agent fault tolerance: task reassignment and leader election."""
import pytest
from src.multi_agent.fault_tolerance import (
    TaskReassignment,
    LeaderElection,
    AgentInfo,
    TaskInfo,
)


# ── TaskReassignment tests ───────────────────────────────────────────

def test_reassignment_creation():
    """TaskReassignment can be instantiated."""
    reassigner = TaskReassignment()
    assert reassigner is not None


def test_reassign_task_to_new_agent():
    """When an agent fails, its tasks are reassigned to another agent."""
    reassigner = TaskReassignment()
    reassigner.register_agent("agent_1", capabilities=["spray"])
    reassigner.register_agent("agent_2", capabilities=["spray"])
    reassigner.assign_task("task_1", "agent_1", requirements=["spray"])

    result = reassigner.reassign_failed_agent("agent_1")

    assert "task_1" in result
    assert result["task_1"] == "agent_2"


def test_reassign_no_available_agent():
    """When no other agent can handle the task, it goes to unassigned."""
    reassigner = TaskReassignment()
    reassigner.register_agent("agent_1", capabilities=["spray"])
    reassigner.assign_task("task_1", "agent_1", requirements=["spray"])

    result = reassigner.reassign_failed_agent("agent_1")

    assert result.get("task_1") is None


def test_reassign_multiple_tasks():
    """All tasks from a failed agent are reassigned."""
    reassigner = TaskReassignment()
    reassigner.register_agent("agent_1", capabilities=["spray", "scan"])
    reassigner.register_agent("agent_2", capabilities=["spray"])
    reassigner.register_agent("agent_3", capabilities=["scan"])
    reassigner.assign_task("task_1", "agent_1", requirements=["spray"])
    reassigner.assign_task("task_2", "agent_1", requirements=["scan"])

    result = reassigner.reassign_failed_agent("agent_1")

    assert result["task_1"] == "agent_2"
    assert result["task_2"] == "agent_3"


def test_reassign_unknown_agent_raises():
    """Reassigning an unknown agent raises ValueError."""
    reassigner = TaskReassignment()
    with pytest.raises(ValueError, match="Unknown agent"):
        reassigner.reassign_failed_agent("nonexistent")


# ── LeaderElection tests ─────────────────────────────────────────────

def test_leader_election_creation():
    """LeaderElection can be instantiated."""
    elector = LeaderElection()
    assert elector is not None


def test_elect_leader_from_candidates():
    """Leader is elected from registered candidates."""
    elector = LeaderElection()
    elector.register_candidate("agent_1")
    elector.register_candidate("agent_2")

    leader = elector.elect_leader()

    assert leader in ("agent_1", "agent_2")


def test_elect_leader_single_candidate():
    """With one candidate, that candidate becomes leader."""
    elector = LeaderElection()
    elector.register_candidate("agent_1")

    leader = elector.elect_leader()

    assert leader == "agent_1"


def test_elect_leader_no_candidates_raises():
    """Electing with no candidates raises ValueError."""
    elector = LeaderElection()
    with pytest.raises(ValueError, match="No candidates"):
        elector.elect_leader()


def test_leader_reelection_after_failure():
    """When the leader fails, a new leader is elected from remaining candidates."""
    elector = LeaderElection()
    elector.register_candidate("agent_1")
    elector.register_candidate("agent_2")
    elector.register_candidate("agent_3")

    first_leader = elector.elect_leader()
    elector.remove_candidate(first_leader)
    second_leader = elector.elect_leader()

    assert second_leader != first_leader
    assert second_leader in ("agent_1", "agent_2", "agent_3")

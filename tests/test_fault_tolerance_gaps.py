"""Test multi-agent fault tolerance: Bully algorithm, heartbeat, health tracking."""

import time

from src.multi_agent.fault_tolerance import (
    AgentInfo,
    LeaderElection,
    TaskReassignment,
)

# ── MA-FT-001: Bully algorithm for leader election ────────────────────


def test_bully_election_highest_priority_wins():
    """Bully algorithm elects the highest priority candidate."""
    elector = LeaderElection(algorithm="bully")
    elector.register_candidate("agent_1", priority=1)
    elector.register_candidate("agent_2", priority=5)
    elector.register_candidate("agent_3", priority=3)

    leader = elector.elect_leader()
    assert leader == "agent_2"


def test_bully_election_single_candidate():
    """Bully algorithm with one candidate elects that candidate."""
    elector = LeaderElection(algorithm="bully")
    elector.register_candidate("agent_1", priority=1)

    leader = elector.elect_leader()
    assert leader == "agent_1"


def test_bully_reelection_after_leader_failure():
    """Bully algorithm re-elects when the leader fails."""
    elector = LeaderElection(algorithm="bully")
    elector.register_candidate("agent_1", priority=1)
    elector.register_candidate("agent_2", priority=5)
    elector.register_candidate("agent_3", priority=3)

    first_leader = elector.elect_leader()
    assert first_leader == "agent_2"

    elector.remove_candidate(first_leader)
    second_leader = elector.elect_leader()
    assert second_leader == "agent_3"


# ── MA-FT-003: Heartbeat integration ──────────────────────────────────


def test_leader_heartbeat_monitoring():
    """Leader heartbeat is monitored for liveness."""
    elector = LeaderElection(algorithm="bully", heartbeat_timeout=0.05)
    elector.register_candidate("agent_1", priority=1)
    elector.register_candidate("agent_2", priority=5)

    leader = elector.elect_leader()
    assert leader == "agent_2"

    # Send heartbeat
    elector.heartbeat(leader)
    assert elector.is_leader_alive()

    # Wait for timeout
    time.sleep(0.1)
    assert not elector.is_leader_alive()


def test_leader_auto_reelection_on_timeout():
    """Leader is automatically re-elected when heartbeat times out."""
    elector = LeaderElection(algorithm="bully", heartbeat_timeout=0.05, auto_reelect=True)
    elector.register_candidate("agent_1", priority=1)
    elector.register_candidate("agent_2", priority=5)
    elector.register_candidate("agent_3", priority=3)

    first_leader = elector.elect_leader()
    assert first_leader == "agent_2"

    # Wait for heartbeat timeout
    time.sleep(0.1)
    elector.check_leader_health()

    # New leader should be elected
    new_leader = elector.leader
    assert new_leader is not None
    assert new_leader != first_leader


# ── MA-FT-005: Task reassignment validation ───────────────────────────


def test_reassignment_validates_replacement_health():
    """Reassignment only targets active agents."""
    reassigner = TaskReassignment()
    reassigner.register_agent("agent_1", capabilities=["spray"])
    reassigner.register_agent("agent_2", capabilities=["spray"])
    reassigner.assign_task("task_1", "agent_1", requirements=["spray"])

    # Mark agent_2 as inactive
    reassigner._agents["agent_2"].is_active = False

    result = reassigner.reassign_failed_agent("agent_1")
    # agent_2 is inactive, so task should be unassigned
    assert result["task_1"] is None


# ── MA-FT-007: Agent health tracking ──────────────────────────────────


def test_agent_health_score():
    """Agents have health scores that degrade with errors."""
    agent = AgentInfo(agent_id="agent_1", capabilities=["spray"])
    assert agent.health_score == 1.0

    agent.record_error()
    assert agent.health_score < 1.0

    agent.record_success()
    assert agent.health_score > 0.0


def test_agent_health_degradation_detection():
    """Agent health degradation is detected after multiple errors."""
    agent = AgentInfo(agent_id="agent_1", capabilities=["spray"])
    for _ in range(10):
        agent.record_error()

    assert agent.health_score < 0.5
    assert agent.is_degraded()


def test_agent_health_recovery():
    """Agent health recovers after successes."""
    agent = AgentInfo(agent_id="agent_1", capabilities=["spray"])
    for _ in range(10):
        agent.record_error()

    assert agent.is_degraded()

    for _ in range(20):
        agent.record_success()

    assert not agent.is_degraded()

"""Tests for deadlock resolution in fault tolerance."""

from src.multi_agent.fault_tolerance import (
    DeadlockDetector,
    DeadlockResolver,
    TaskReassignment,
)

# ── MA-FT-DL-001: Deadlock detection ──────────────────────────────────


def test_deadlock_detection_no_cycle():
    """No deadlock when there are no circular wait dependencies."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A2")
    detector.add_wait_edge("A2", "A3")
    assert not detector.detect_deadlock()


def test_deadlock_detection_simple_cycle():
    """Deadlock detected when A waits for B and B waits for A."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A2")
    detector.add_wait_edge("A2", "A1")
    assert detector.detect_deadlock()


def test_deadlock_detection_three_way_cycle():
    """Deadlock detected in a three-agent cycle."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A2")
    detector.add_wait_edge("A2", "A3")
    detector.add_wait_edge("A3", "A1")
    assert detector.detect_deadlock()


def test_deadlock_detection_self_loop():
    """Deadlock detected when agent waits for itself."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A1")
    assert detector.detect_deadlock()


def test_deadlock_detection_diamond_no_cycle():
    """No deadlock in diamond dependency (no cycle)."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A2")
    detector.add_wait_edge("A1", "A3")
    detector.add_wait_edge("A2", "A4")
    detector.add_wait_edge("A3", "A4")
    assert not detector.detect_deadlock()


# ── MA-FT-DL-002: Deadlock resolution ─────────────────────────────────


def test_deadlock_resolution_victim_selection():
    """Deadlock resolver selects a victim to break the cycle."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A1", "A2")
    resolver.add_wait_edge("A2", "A1")

    victim = resolver.resolve_deadlock()
    assert victim in ("A1", "A2")


def test_deadlock_resolution_removes_cycle():
    """After resolution, the deadlock cycle is broken."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A1", "A2")
    resolver.add_wait_edge("A2", "A1")

    victim = resolver.resolve_deadlock()
    # Remove the victim's outgoing edge to break the cycle
    resolver.remove_wait_edge(victim, "A1" if victim == "A2" else "A2")
    assert not resolver.detect_deadlock()


def test_deadlock_resolution_no_deadlock():
    """Resolver returns None when there's no deadlock."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A1", "A2")
    victim = resolver.resolve_deadlock()
    assert victim is None


# ── MA-FT-DL-003: Priority-based victim selection ─────────────────────


def test_deadlock_resolution_priority_victim():
    """Lower priority agent is selected as victim."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A1", "A2")
    resolver.add_wait_edge("A2", "A1")
    resolver.set_priority("A1", 10)
    resolver.set_priority("A2", 1)

    victim = resolver.resolve_deadlock()
    assert victim == "A2"  # Lower priority


def test_deadlock_resolution_equal_priority_deterministic():
    """With equal priorities, victim selection is deterministic."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A1", "A2")
    resolver.add_wait_edge("A2", "A1")

    victim1 = resolver.resolve_deadlock()
    victim2 = resolver.resolve_deadlock()
    assert victim1 == victim2


# ── MA-FT-DL-004: Deadlock detection with task reassignment ───────────


def test_deadlock_resolution_triggers_reassignment():
    """Deadlock resolution triggers task reassignment for victim."""
    resolver = DeadlockResolver()
    reassigner = TaskReassignment()
    reassigner.register_agent("A1", capabilities=["spray"])
    reassigner.register_agent("A2", capabilities=["spray"])
    reassigner.register_agent("A3", capabilities=["spray"])
    reassigner.assign_task("T1", "A1", requirements=["spray"])

    resolver.add_wait_edge("A1", "A2")
    resolver.add_wait_edge("A2", "A1")
    resolver.set_reassignment_callback(reassigner.reassign_failed_agent)

    victim = resolver.resolve_deadlock()
    assert victim in ("A1", "A2")


# ── MA-FT-DL-005: Wait-for graph management ───────────────────────────


def test_wait_for_graph_clear():
    """Clearing wait-for graph removes all edges."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A2")
    detector.add_wait_edge("A2", "A1")
    assert detector.detect_deadlock()

    detector.clear()
    assert not detector.detect_deadlock()


def test_wait_for_graph_remove_edge():
    """Removing a specific edge breaks the cycle."""
    detector = DeadlockDetector()
    detector.add_wait_edge("A1", "A2")
    detector.add_wait_edge("A2", "A1")
    assert detector.detect_deadlock()

    detector.remove_wait_edge("A1", "A2")
    assert not detector.detect_deadlock()


# ── MA-FT-DL-006: resolve_and_recover ────────────────────────────────


def test_resolve_and_recover_breaks_cycle():
    """resolve_and_recover removes victim edges and breaks the cycle."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "A")
    resolver.set_priority("A", 5)
    resolver.set_priority("B", 1)

    victim = resolver.resolve_and_recover()
    assert victim == "B"
    assert not resolver.detect_deadlock()


def test_resolve_and_recover_no_deadlock():
    """resolve_and_recover returns None when no deadlock exists."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    assert resolver.resolve_and_recover() is None


def test_resolve_and_recover_three_agent_cycle():
    """resolve_and_recover breaks a three-agent cycle."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "C")
    resolver.add_wait_edge("C", "A")
    resolver.set_priority("A", 10)
    resolver.set_priority("B", 3)
    resolver.set_priority("C", 7)

    victim = resolver.resolve_and_recover()
    assert victim == "B"
    assert not resolver.detect_deadlock()


def test_resolve_and_recover_triggers_callback():
    """resolve_and_recover invokes the reassignment callback."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "A")
    resolver.set_priority("A", 5)
    resolver.set_priority("B", 1)

    callback_calls = []
    resolver.set_reassignment_callback(lambda v: callback_calls.append(v))

    resolver.resolve_and_recover()
    assert callback_calls == ["B"]


def test_resolve_and_recover_self_loop():
    """resolve_and_recover handles self-loop deadlock."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "A")
    resolver.set_priority("A", 1)

    victim = resolver.resolve_and_recover()
    assert victim == "A"
    assert not resolver.detect_deadlock()


def test_resolve_and_recover_multiple_cycles():
    """resolve_and_recover breaks one cycle per call."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "A")
    resolver.add_wait_edge("C", "D")
    resolver.add_wait_edge("D", "C")
    resolver.set_priority("A", 1)
    resolver.set_priority("B", 5)
    resolver.set_priority("C", 2)
    resolver.set_priority("D", 8)

    # First call breaks one cycle
    victim1 = resolver.resolve_and_recover()
    assert victim1 in ("A", "C")

    # Second call breaks the remaining cycle
    victim2 = resolver.resolve_and_recover()
    assert victim2 in ("A", "C")
    assert victim2 != victim1
    assert not resolver.detect_deadlock()


# ── MA-FT-DL-007: get_cycle_path ─────────────────────────────────────


def test_get_cycle_path_no_deadlock():
    """get_cycle_path returns None when no deadlock exists."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "C")
    assert resolver.get_cycle_path() is None


def test_get_cycle_path_simple_cycle():
    """get_cycle_path returns the cycle for a two-agent deadlock."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "A")
    path = resolver.get_cycle_path()
    assert path is not None
    assert len(path) == 2
    assert set(path) == {"A", "B"}


def test_get_cycle_path_three_agent_cycle():
    """get_cycle_path returns the full cycle for three agents."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "C")
    resolver.add_wait_edge("C", "A")
    path = resolver.get_cycle_path()
    assert path is not None
    assert len(path) == 3
    assert set(path) == {"A", "B", "C"}


def test_get_cycle_path_self_loop():
    """get_cycle_path returns [agent] for self-loop."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "A")
    path = resolver.get_cycle_path()
    assert path == ["A"]


def test_get_cycle_path_continuity():
    """get_cycle_path returns a valid path (each node waits for next)."""
    resolver = DeadlockResolver()
    resolver.add_wait_edge("A", "B")
    resolver.add_wait_edge("B", "C")
    resolver.add_wait_edge("C", "A")
    path = resolver.get_cycle_path()
    assert path is not None
    for i in range(len(path)):
        current = path[i]
        next_node = path[(i + 1) % len(path)]
        assert next_node in resolver._wait_for.get(current, [])

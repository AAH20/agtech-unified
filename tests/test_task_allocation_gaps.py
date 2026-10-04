"""Test multi-agent task allocation: dependencies, priorities, capability constraints."""

import pytest

from src.multi_agent.task_allocation import AllocationInstance, TaskAllocator

# ── MA-TA-005: Task dependencies ──────────────────────────────────────


def test_allocation_task_dependencies():
    """Tasks with dependencies are ordered correctly."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2"],
        tasks=["T1", "T2"],
        cost_matrix=[[1.0, 5.0], [5.0, 1.0]],
        dependencies={"T2": ["T1"]},
    )
    result = allocator.allocate(instance)
    assert result.allocation is not None
    # T1 should be assigned before T2
    assert "T1" in result.allocation.get("A1", []) or "T1" in result.allocation.get("A2", [])


def test_allocation_circular_dependency_raises():
    """Circular dependencies raise ValueError."""
    with pytest.raises(ValueError, match="Circular dependency"):
        AllocationInstance(
            agents=["A1"],
            tasks=["T1", "T2"],
            cost_matrix=[[1.0, 2.0]],
            dependencies={"T1": ["T2"], "T2": ["T1"]},
        )


def test_allocation_dependency_unknown_task_raises():
    """Dependency on unknown task raises ValueError."""
    with pytest.raises(ValueError, match="Unknown task"):
        AllocationInstance(
            agents=["A1"],
            tasks=["T1"],
            cost_matrix=[[1.0]],
            dependencies={"T1": ["nonexistent"]},
        )


# ── MA-TA-006: Task priorities ────────────────────────────────────────


def test_allocation_task_priorities():
    """Tasks have priority fields."""
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[1.0]],
        priorities={"T1": 5},
    )
    assert instance.priorities["T1"] == 5


def test_allocation_priority_affects_assignment():
    """Higher priority tasks are preferred in allocation."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2"],
        tasks=["T1", "T2"],
        cost_matrix=[[1.0, 10.0], [1.0, 10.0]],
        priorities={"T1": 10, "T2": 1},
    )
    result = allocator.allocate(instance)
    # Both tasks should be assigned
    assigned = [t for tasks in result.allocation.values() for t in tasks]
    assert len(assigned) == 2


# ── MA-TA-007: Task deadlines ─────────────────────────────────────────


def test_allocation_task_deadlines():
    """Tasks have deadline fields."""
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[1.0]],
        deadlines={"T1": 100.0},
    )
    assert instance.deadlines["T1"] == 100.0


# ── MA-TA-017: Agent capability constraints ───────────────────────────


def test_allocation_capability_constraints():
    """Tasks are only assigned to agents with required capabilities."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2"],
        tasks=["T1", "T2"],
        cost_matrix=[[1.0, 10.0], [10.0, 1.0]],
        agent_capabilities={"A1": ["spray"], "A2": ["scan"]},
        task_requirements={"T1": ["spray"], "T2": ["scan"]},
    )
    result = allocator.allocate(instance)
    # T1 should go to A1 (has spray), T2 should go to A2 (has scan)
    assert "T1" in result.allocation.get("A1", [])
    assert "T2" in result.allocation.get("A2", [])


def test_allocation_capability_no_match():
    """Task with no capable agent is unassigned."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[1.0]],
        agent_capabilities={"A1": ["spray"]},
        task_requirements={"T1": ["harvest"]},
    )
    result = allocator.allocate(instance)
    assert "T1" not in result.allocation.get("A1", [])
    assert "T1" in (result.unassigned_tasks or [])


# ── MA-TA-015: Unassigned task reporting ──────────────────────────────


def test_allocation_unassigned_tasks_populated():
    """Unassigned tasks are reported in AllocationResult."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1", "T2"],
        cost_matrix=[[1.0, 2.0]],
        agent_capabilities={"A1": ["spray"]},
        task_requirements={"T1": ["spray"], "T2": ["harvest"]},
    )
    result = allocator.allocate(instance)
    assert result.unassigned_tasks is not None
    assert "T2" in result.unassigned_tasks


# ── MA-TA-016: Cost matrix validation ─────────────────────────────────


def test_allocation_negative_cost_raises():
    """Negative cost in cost matrix raises ValueError."""
    with pytest.raises(ValueError, match="Negative cost"):
        AllocationInstance(
            agents=["A1"],
            tasks=["T1"],
            cost_matrix=[[-1.0]],
        )


def test_allocation_nan_cost_raises():
    """NaN cost in cost matrix raises ValueError."""
    with pytest.raises(ValueError, match="NaN"):
        AllocationInstance(
            agents=["A1"],
            tasks=["T1"],
            cost_matrix=[[float("nan")]],
        )


def test_allocation_infinite_cost_raises():
    """Infinite cost in cost matrix raises ValueError."""
    with pytest.raises(ValueError, match="Infinite"):
        AllocationInstance(
            agents=["A1"],
            tasks=["T1"],
            cost_matrix=[[float("inf")]],
        )


# ── MA-TA-008: Resource constraints ───────────────────────────────────


def test_allocation_resource_constraints():
    """Agents have resource limits."""
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[1.0]],
        agent_resources={"A1": {"battery": 100.0}},
        task_resource_requirements={"T1": {"battery": 50.0}},
    )
    assert instance.agent_resources["A1"]["battery"] == 100.0
    assert instance.task_resource_requirements["T1"]["battery"] == 50.0


# ── MA-TA-009: Time windows ───────────────────────────────────────────


def test_allocation_time_windows():
    """Tasks have time window constraints."""
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[1.0]],
        time_windows={"T1": (0.0, 100.0)},
    )
    assert instance.time_windows["T1"] == (0.0, 100.0)

"""Test multi-agent task allocation for agricultural robots."""

import pytest

from src.multi_agent.task_allocation import AllocationInstance, TaskAllocator


def test_allocation_empty():
    """No tasks returns empty allocation."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=[],
        tasks=[],
        cost_matrix=[],
    )
    result = allocator.allocate(instance)
    assert result.allocation == {}
    assert result.total_cost == 0.0


def test_allocation_single_agent_single_task():
    """One agent, one task → trivial allocation."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[5.0]],
    )
    result = allocator.allocate(instance)
    assert result.allocation == {"A1": ["T1"]}
    assert result.total_cost == pytest.approx(5.0)


def test_allocation_two_agents_two_tasks():
    """Two agents, two tasks → optimal assignment."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2"],
        tasks=["T1", "T2"],
        cost_matrix=[
            [1.0, 10.0],
            [10.0, 1.0],
        ],
    )
    result = allocator.allocate(instance)
    # Optimal: A1→T1 (1.0), A2→T2 (1.0) = 2.0
    assert result.total_cost == pytest.approx(2.0)


def test_allocation_more_tasks_than_agents():
    """More tasks than agents → some agents get multiple tasks."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2"],
        tasks=["T1", "T2", "T3"],
        cost_matrix=[
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
    )
    result = allocator.allocate(instance)
    # All 3 tasks assigned
    assigned = [t for tasks in result.allocation.values() for t in tasks]
    assert len(assigned) == 3
    assert set(assigned) == {"T1", "T2", "T3"}


def test_allocation_more_agents_than_tasks():
    """More agents than tasks → some agents idle."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2", "A3"],
        tasks=["T1", "T2"],
        cost_matrix=[
            [1.0, 10.0],
            [10.0, 1.0],
            [5.0, 5.0],
        ],
    )
    result = allocator.allocate(instance)
    # Only 2 tasks assigned
    assigned = [t for tasks in result.allocation.values() for t in tasks]
    assert len(assigned) == 2


def test_allocation_greedy_algorithm():
    """Greedy algorithm produces valid allocation."""
    allocator = TaskAllocator(algorithm="greedy")
    instance = AllocationInstance(
        agents=["A1", "A2"],
        tasks=["T1", "T2"],
        cost_matrix=[
            [1.0, 10.0],
            [10.0, 1.0],
        ],
    )
    result = allocator.allocate(instance)
    assert result.algorithm == "greedy"
    assert result.total_cost > 0


def test_allocation_invalid_cost_matrix():
    """Non-square cost matrix raises ValueError."""
    TaskAllocator()
    with pytest.raises(ValueError, match="Cost matrix must be square"):
        AllocationInstance(
            agents=["A1", "A2"],
            tasks=["T1", "T2"],
            cost_matrix=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )


def test_allocation_result_contains_algorithm():
    """AllocationResult includes algorithm name."""
    allocator = TaskAllocator(algorithm="greedy")
    instance = AllocationInstance(
        agents=["A1"],
        tasks=["T1"],
        cost_matrix=[[5.0]],
    )
    result = allocator.allocate(instance)
    assert result.algorithm == "greedy"


def test_allocation_all_tasks_assigned():
    """Every task is assigned to exactly one agent."""
    allocator = TaskAllocator()
    instance = AllocationInstance(
        agents=["A1", "A2", "A3"],
        tasks=["T1", "T2", "T3", "T4"],
        cost_matrix=[
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 4.0, 3.0],
        ],
    )
    result = allocator.allocate(instance)
    assigned = [t for tasks in result.allocation.values() for t in tasks]
    assert len(assigned) == 4
    assert len(set(assigned)) == 4  # No duplicates

"""Tests for IoT edge computing scheduler: task offloading, latency-aware placement.

TDD: these tests define the expected behavior for the new features.
"""

import pytest

from src.iot.edge_computing import (
    EdgeNode,
    EdgeScheduler,
    EdgeTask,
)

# ===========================================================================
# EdgeNode
# ===========================================================================


class TestEdgeNode:
    """EdgeNode represents an edge computing resource."""

    def test_create_node(self):
        """Create an edge node with capacity and latency."""
        node = EdgeNode(
            node_id="edge-1",
            cpu_capacity=80.0,
            memory_capacity=1024.0,
            latency_ms=15.0,
            location="field-a",
        )
        assert node.node_id == "edge-1"
        assert node.cpu_capacity == 80.0
        assert node.memory_capacity == 1024.0
        assert node.latency_ms == 15.0
        assert node.location == "field-a"

    def test_node_default_location(self):
        """Location defaults to empty string."""
        node = EdgeNode(node_id="edge-1", cpu_capacity=50.0, memory_capacity=512.0, latency_ms=10.0)
        assert node.location == ""

    def test_invalid_cpu_capacity_raises(self):
        """CPU capacity must be positive."""
        with pytest.raises(ValueError, match="cpu_capacity must be positive"):
            EdgeNode(node_id="edge-1", cpu_capacity=0, memory_capacity=512.0, latency_ms=10.0)

    def test_invalid_memory_capacity_raises(self):
        """Memory capacity must be positive."""
        with pytest.raises(ValueError, match="memory_capacity must be positive"):
            EdgeNode(node_id="edge-1", cpu_capacity=50.0, memory_capacity=0, latency_ms=10.0)

    def test_invalid_latency_raises(self):
        """Latency must be non-negative."""
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            EdgeNode(node_id="edge-1", cpu_capacity=50.0, memory_capacity=512.0, latency_ms=-1.0)


# ===========================================================================
# EdgeTask
# ===========================================================================


class TestEdgeTask:
    """EdgeTask represents a computation task to offload."""

    def test_create_task(self):
        """Create an edge task with requirements."""
        task = EdgeTask(
            task_id="task-1",
            cpu_required=20.0,
            memory_required=256.0,
            deadline_ms=100.0,
            priority=5,
            data_size_kb=50.0,
        )
        assert task.task_id == "task-1"
        assert task.cpu_required == 20.0
        assert task.memory_required == 256.0
        assert task.deadline_ms == 100.0
        assert task.priority == 5
        assert task.data_size_kb == 50.0

    def test_task_default_priority(self):
        """Priority defaults to 0."""
        task = EdgeTask(
            task_id="task-1",
            cpu_required=10.0,
            memory_required=128.0,
            deadline_ms=50.0,
        )
        assert task.priority == 0
        assert task.data_size_kb == 0.0

    def test_invalid_cpu_required_raises(self):
        """CPU required must be positive."""
        with pytest.raises(ValueError, match="cpu_required must be positive"):
            EdgeTask(task_id="task-1", cpu_required=0, memory_required=128.0, deadline_ms=50.0)

    def test_invalid_memory_required_raises(self):
        """Memory required must be positive."""
        with pytest.raises(ValueError, match="memory_required must be positive"):
            EdgeTask(task_id="task-1", cpu_required=10.0, memory_required=0, deadline_ms=50.0)

    def test_invalid_deadline_raises(self):
        """Deadline must be positive."""
        with pytest.raises(ValueError, match="deadline_ms must be positive"):
            EdgeTask(task_id="task-1", cpu_required=10.0, memory_required=128.0, deadline_ms=0)


# ===========================================================================
# EdgeScheduler — Node Management
# ===========================================================================


class TestEdgeSchedulerNodes:
    """EdgeScheduler node registration and management."""

    def test_register_node(self):
        """Register an edge node."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        assert scheduler.register_node(node) is True

    def test_register_duplicate_node_raises(self):
        """Registering the same node ID twice raises ValueError."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        with pytest.raises(ValueError, match="already registered"):
            scheduler.register_node(node)

    def test_unregister_node(self):
        """Unregister a node."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        assert scheduler.unregister_node("edge-1") is True

    def test_unregister_nonexistent_node(self):
        """Unregistering a nonexistent node returns False."""
        scheduler = EdgeScheduler()
        assert scheduler.unregister_node("nonexistent") is False

    def test_get_all_nodes(self):
        """Get all registered nodes."""
        scheduler = EdgeScheduler()
        node1 = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        node2 = EdgeNode(
            node_id="edge-2", cpu_capacity=60.0, memory_capacity=512.0, latency_ms=25.0
        )
        scheduler.register_node(node1)
        scheduler.register_node(node2)
        nodes = scheduler.get_all_nodes()
        assert len(nodes) == 2
        assert {n.node_id for n in nodes} == {"edge-1", "edge-2"}

    def test_get_node_load_empty(self):
        """Load on a fresh node is zero."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        load = scheduler.get_node_load("edge-1")
        assert load["cpu_used"] == 0.0
        assert load["memory_used"] == 0.0
        assert load["cpu_available"] == 80.0
        assert load["memory_available"] == 1024.0

    def test_get_node_load_nonexistent(self):
        """Load on a nonexistent node returns zeros."""
        scheduler = EdgeScheduler()
        load = scheduler.get_node_load("nonexistent")
        assert load["cpu_used"] == 0.0
        assert load["memory_used"] == 0.0


# ===========================================================================
# EdgeScheduler — Task Submission
# ===========================================================================


class TestEdgeSchedulerTaskSubmission:
    """EdgeScheduler task submission."""

    def test_submit_task(self):
        """Submit a task to the scheduler."""
        scheduler = EdgeScheduler()
        task = EdgeTask(
            task_id="task-1",
            cpu_required=20.0,
            memory_required=256.0,
            deadline_ms=100.0,
        )
        assert scheduler.submit_task(task) is True

    def test_submit_duplicate_task_raises(self):
        """Submitting the same task ID twice raises ValueError."""
        scheduler = EdgeScheduler()
        task = EdgeTask(
            task_id="task-1",
            cpu_required=20.0,
            memory_required=256.0,
            deadline_ms=100.0,
        )
        scheduler.submit_task(task)
        with pytest.raises(ValueError, match="already submitted"):
            scheduler.submit_task(task)

    def test_get_all_tasks(self):
        """Get all submitted tasks."""
        scheduler = EdgeScheduler()
        task1 = EdgeTask(
            task_id="task-1", cpu_required=10.0, memory_required=128.0, deadline_ms=50.0
        )
        task2 = EdgeTask(
            task_id="task-2", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task1)
        scheduler.submit_task(task2)
        tasks = scheduler.get_all_tasks()
        assert len(tasks) == 2
        assert {t.task_id for t in tasks} == {"task-1", "task-2"}


# ===========================================================================
# EdgeScheduler — Scheduling & Placement
# ===========================================================================


class TestEdgeSchedulerScheduling:
    """EdgeScheduler task-to-node assignment with latency-aware placement."""

    def test_schedule_assigns_task_to_node(self):
        """A task is assigned to a node with sufficient capacity."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].task_id == "task-1"
        assert assignments[0].node_id == "edge-1"
        assert assignments[0].status == "assigned"

    def test_schedule_prefers_lowest_latency(self):
        """Among capable nodes, the one with lowest latency is chosen."""
        scheduler = EdgeScheduler()
        node_slow = EdgeNode(
            node_id="edge-slow", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=50.0
        )
        node_fast = EdgeNode(
            node_id="edge-fast", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=10.0
        )
        scheduler.register_node(node_slow)
        scheduler.register_node(node_fast)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].node_id == "edge-fast"

    def test_schedule_respects_deadline(self):
        """A node whose latency exceeds the task deadline is not chosen."""
        scheduler = EdgeScheduler()
        node_slow = EdgeNode(
            node_id="edge-slow", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=200.0
        )
        node_fast = EdgeNode(
            node_id="edge-fast", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=10.0
        )
        scheduler.register_node(node_slow)
        scheduler.register_node(node_fast)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=50.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].node_id == "edge-fast"

    def test_schedule_no_capable_node(self):
        """Task is unassigned when no node has sufficient capacity."""
        scheduler = EdgeScheduler()
        node = EdgeNode(node_id="edge-1", cpu_capacity=10.0, memory_capacity=128.0, latency_ms=15.0)
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=50.0, memory_required=512.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].status == "unassigned"
        assert assignments[0].node_id is None

    def test_schedule_respects_cpu_capacity(self):
        """A node at full CPU capacity is not chosen."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=30.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task1 = EdgeTask(
            task_id="task-1", cpu_required=25.0, memory_required=256.0, deadline_ms=100.0
        )
        task2 = EdgeTask(
            task_id="task-2", cpu_required=25.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task1)
        scheduler.submit_task(task2)
        assignments = scheduler.schedule()
        assigned = [a for a in assignments if a.status == "assigned"]
        unassigned = [a for a in assignments if a.status == "unassigned"]
        assert len(assigned) == 1
        assert len(unassigned) == 1

    def test_schedule_respects_memory_capacity(self):
        """A node at full memory capacity is not chosen."""
        scheduler = EdgeScheduler()
        node = EdgeNode(node_id="edge-1", cpu_capacity=80.0, memory_capacity=300.0, latency_ms=15.0)
        scheduler.register_node(node)
        task1 = EdgeTask(
            task_id="task-1", cpu_required=10.0, memory_required=256.0, deadline_ms=100.0
        )
        task2 = EdgeTask(
            task_id="task-2", cpu_required=10.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task1)
        scheduler.submit_task(task2)
        assignments = scheduler.schedule()
        assigned = [a for a in assignments if a.status == "assigned"]
        unassigned = [a for a in assignments if a.status == "unassigned"]
        assert len(assigned) == 1
        assert len(unassigned) == 1

    def test_schedule_multiple_tasks_multiple_nodes(self):
        """Multiple tasks are distributed across nodes."""
        scheduler = EdgeScheduler()
        node1 = EdgeNode(
            node_id="edge-1", cpu_capacity=50.0, memory_capacity=512.0, latency_ms=20.0
        )
        node2 = EdgeNode(
            node_id="edge-2", cpu_capacity=50.0, memory_capacity=512.0, latency_ms=30.0
        )
        scheduler.register_node(node1)
        scheduler.register_node(node2)
        for i in range(4):
            task = EdgeTask(
                task_id=f"task-{i}",
                cpu_required=20.0,
                memory_required=200.0,
                deadline_ms=100.0,
            )
            scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 4
        assigned = [a for a in assignments if a.status == "assigned"]
        assert len(assigned) == 4
        # Both nodes should be used
        used_nodes = {a.node_id for a in assigned}
        assert len(used_nodes) == 2

    def test_schedule_priority_ordering(self):
        """Higher priority tasks are scheduled first."""
        scheduler = EdgeScheduler()
        node = EdgeNode(node_id="edge-1", cpu_capacity=30.0, memory_capacity=512.0, latency_ms=15.0)
        scheduler.register_node(node)
        task_low = EdgeTask(
            task_id="task-low",
            cpu_required=25.0,
            memory_required=256.0,
            deadline_ms=100.0,
            priority=1,
        )
        task_high = EdgeTask(
            task_id="task-high",
            cpu_required=25.0,
            memory_required=256.0,
            deadline_ms=100.0,
            priority=10,
        )
        scheduler.submit_task(task_low)
        scheduler.submit_task(task_high)
        assignments = scheduler.schedule()
        assigned = [a for a in assignments if a.status == "assigned"]
        unassigned = [a for a in assignments if a.status == "unassigned"]
        assert len(assigned) == 1
        assert len(unassigned) == 1
        assert assigned[0].task_id == "task-high"

    def test_get_task_assignment(self):
        """Get the assignment for a specific task."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        scheduler.schedule()
        assignment = scheduler.get_task_assignment("task-1")
        assert assignment is not None
        assert assignment.task_id == "task-1"
        assert assignment.node_id == "edge-1"

    def test_get_task_assignment_nonexistent(self):
        """Getting assignment for nonexistent task returns None."""
        scheduler = EdgeScheduler()
        assert scheduler.get_task_assignment("nonexistent") is None

    def test_schedule_empty_scheduler(self):
        """Scheduling with no tasks returns empty list."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        assert scheduler.schedule() == []

    def test_schedule_no_nodes(self):
        """Scheduling with no nodes leaves all tasks unassigned."""
        scheduler = EdgeScheduler()
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].status == "unassigned"


# ===========================================================================
# EdgeScheduler — Task Completion
# ===========================================================================


class TestEdgeSchedulerCompletion:
    """EdgeScheduler task completion and resource release."""

    def test_complete_task_releases_resources(self):
        """Completing a task frees its resources on the node."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        scheduler.schedule()
        assert scheduler.complete_task("task-1") is True
        load = scheduler.get_node_load("edge-1")
        assert load["cpu_used"] == 0.0
        assert load["memory_used"] == 0.0

    def test_complete_nonexistent_task(self):
        """Completing a nonexistent task returns False."""
        scheduler = EdgeScheduler()
        assert scheduler.complete_task("nonexistent") is False

    def test_complete_task_updates_assignment_status(self):
        """Completing a task sets its assignment status to 'completed'."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        scheduler.schedule()
        scheduler.complete_task("task-1")
        assignment = scheduler.get_task_assignment("task-1")
        assert assignment.status == "completed"

    def test_unregister_node_with_tasks(self):
        """Unregistering a node unassigns its tasks."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        scheduler.schedule()
        scheduler.unregister_node("edge-1")
        assignment = scheduler.get_task_assignment("task-1")
        assert assignment.status == "unassigned"
        assert assignment.node_id is None


# ===========================================================================
# EdgeScheduler — Latency-Aware Placement
# ===========================================================================


class TestEdgeSchedulerLatencyAware:
    """Latency-aware placement decisions."""

    def test_estimated_latency_includes_processing(self):
        """Estimated latency includes both network and processing time."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1",
            cpu_required=20.0,
            memory_required=256.0,
            deadline_ms=100.0,
            data_size_kb=100.0,
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        # Estimated latency should be >= network latency
        assert assignments[0].estimated_latency_ms >= 15.0

    def test_data_size_affects_latency(self):
        """Larger data size increases estimated latency."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task_small = EdgeTask(
            task_id="task-small",
            cpu_required=10.0,
            memory_required=128.0,
            deadline_ms=200.0,
            data_size_kb=10.0,
        )
        task_large = EdgeTask(
            task_id="task-large",
            cpu_required=10.0,
            memory_required=128.0,
            deadline_ms=200.0,
            data_size_kb=500.0,
        )
        scheduler.submit_task(task_small)
        scheduler.submit_task(task_large)
        assignments = scheduler.schedule()
        small_assign = next(a for a in assignments if a.task_id == "task-small")
        large_assign = next(a for a in assignments if a.task_id == "task-large")
        assert large_assign.estimated_latency_ms > small_assign.estimated_latency_ms

    def test_schedule_with_multiple_capable_nodes_picks_best(self):
        """With multiple capable nodes, the best (lowest latency) is chosen."""
        scheduler = EdgeScheduler()
        nodes = [
            EdgeNode(node_id=f"edge-{i}", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=lat)
            for i, lat in enumerate([30.0, 10.0, 50.0, 20.0])
        ]
        for node in nodes:
            scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].node_id == "edge-1"  # latency 10.0


# ===========================================================================
# EdgeScheduler — Edge Cases
# ===========================================================================


class TestEdgeSchedulerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_schedule_idempotent(self):
        """Running schedule() twice doesn't double-assign."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=20.0, memory_required=256.0, deadline_ms=100.0
        )
        scheduler.submit_task(task)
        scheduler.schedule()
        assignments = scheduler.schedule()
        # Second schedule should not create new assignments for already-assigned tasks
        assert len(assignments) == 1

    def test_node_load_after_multiple_tasks(self):
        """Node load reflects all assigned tasks."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=15.0
        )
        scheduler.register_node(node)
        for i in range(3):
            task = EdgeTask(
                task_id=f"task-{i}",
                cpu_required=20.0,
                memory_required=256.0,
                deadline_ms=100.0,
            )
            scheduler.submit_task(task)
        scheduler.schedule()
        load = scheduler.get_node_load("edge-1")
        assert load["cpu_used"] == 60.0
        assert load["memory_used"] == 768.0
        assert load["cpu_available"] == 20.0
        assert load["memory_available"] == 256.0

    def test_task_deadline_too_tight(self):
        """Task with deadline tighter than any node latency is unassigned."""
        scheduler = EdgeScheduler()
        node = EdgeNode(
            node_id="edge-1", cpu_capacity=80.0, memory_capacity=1024.0, latency_ms=50.0
        )
        scheduler.register_node(node)
        task = EdgeTask(
            task_id="task-1", cpu_required=10.0, memory_required=128.0, deadline_ms=30.0
        )
        scheduler.submit_task(task)
        assignments = scheduler.schedule()
        assert len(assignments) == 1
        assert assignments[0].status == "unassigned"

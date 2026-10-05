"""IoT edge computing scheduler: task offloading and latency-aware placement.

Provides EdgeNode (computing resource), EdgeTask (computation unit),
and EdgeScheduler (task-to-node assignment with latency awareness).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EdgeNode:
    """Edge computing node with capacity and latency characteristics."""

    node_id: str
    cpu_capacity: float
    memory_capacity: float
    latency_ms: float
    location: str = ""

    def __post_init__(self) -> None:
        if self.cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be positive")
        if self.memory_capacity <= 0:
            raise ValueError("memory_capacity must be positive")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass
class EdgeTask:
    """Computation task to be offloaded to an edge node."""

    task_id: str
    cpu_required: float
    memory_required: float
    deadline_ms: float
    priority: int = 0
    data_size_kb: float = 0.0

    def __post_init__(self) -> None:
        if self.cpu_required <= 0:
            raise ValueError("cpu_required must be positive")
        if self.memory_required <= 0:
            raise ValueError("memory_required must be positive")
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")


@dataclass
class TaskAssignment:
    """Assignment of a task to an edge node."""

    task_id: str
    node_id: Optional[str]
    status: str  # "assigned", "unassigned", "completed"
    estimated_latency_ms: float = 0.0


class EdgeScheduler:
    """Latency-aware edge computing scheduler.

    Assigns tasks to edge nodes based on:
    - Resource capacity (CPU, memory)
    - Network latency (lower is better)
    - Task deadline (latency must be within deadline)
    - Task priority (higher priority scheduled first)
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, EdgeNode] = {}
        self._tasks: Dict[str, EdgeTask] = {}
        self._assignments: Dict[str, TaskAssignment] = {}
        self._node_cpu_used: Dict[str, float] = {}
        self._node_memory_used: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Node Management
    # ------------------------------------------------------------------

    def register_node(self, node: EdgeNode) -> bool:
        """Register an edge node with the scheduler."""
        with self._lock:
            if node.node_id in self._nodes:
                raise ValueError(f"Node '{node.node_id}' already registered")
            self._nodes[node.node_id] = node
            self._node_cpu_used[node.node_id] = 0.0
            self._node_memory_used[node.node_id] = 0.0
            return True

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node and unassign its tasks."""
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            del self._node_cpu_used[node_id]
            del self._node_memory_used[node_id]
            # Unassign tasks on this node
            for task_id, assignment in self._assignments.items():
                if assignment.node_id == node_id and assignment.status == "assigned":
                    assignment.status = "unassigned"
                    assignment.node_id = None
            return True

    def get_all_nodes(self) -> List[EdgeNode]:
        """Get all registered nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_node_load(self, node_id: str) -> Dict[str, float]:
        """Get current resource usage for a node."""
        with self._lock:
            if node_id not in self._nodes:
                return {
                    "cpu_used": 0.0,
                    "memory_used": 0.0,
                    "cpu_available": 0.0,
                    "memory_available": 0.0,
                }
            node = self._nodes[node_id]
            cpu_used = self._node_cpu_used.get(node_id, 0.0)
            mem_used = self._node_memory_used.get(node_id, 0.0)
            return {
                "cpu_used": cpu_used,
                "memory_used": mem_used,
                "cpu_available": node.cpu_capacity - cpu_used,
                "memory_available": node.memory_capacity - mem_used,
            }

    # ------------------------------------------------------------------
    # Task Management
    # ------------------------------------------------------------------

    def submit_task(self, task: EdgeTask) -> bool:
        """Submit a task to the scheduler."""
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError(f"Task '{task.task_id}' already submitted")
            self._tasks[task.task_id] = task
            self._assignments[task.task_id] = TaskAssignment(
                task_id=task.task_id,
                node_id=None,
                status="unassigned",
            )
            return True

    def get_all_tasks(self) -> List[EdgeTask]:
        """Get all submitted tasks."""
        with self._lock:
            return list(self._tasks.values())

    def get_task_assignment(self, task_id: str) -> Optional[TaskAssignment]:
        """Get the assignment for a specific task."""
        with self._lock:
            return self._assignments.get(task_id)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(self) -> List[TaskAssignment]:
        """Schedule all unassigned tasks to edge nodes.

        Uses latency-aware placement: among capable nodes, picks the one
        with the lowest estimated latency that meets the task deadline.
        Higher priority tasks are scheduled first.
        """
        with self._lock:
            # Get unassigned tasks sorted by priority (descending)
            unassigned_tasks = [
                task
                for task in self._tasks.values()
                if self._assignments[task.task_id].status == "unassigned"
            ]
            unassigned_tasks.sort(key=lambda t: t.priority, reverse=True)

            for task in unassigned_tasks:
                best_node = self._find_best_node(task)
                if best_node is not None:
                    self._assign_task(task, best_node)
                # else: remains unassigned

            return list(self._assignments.values())

    def _find_best_node(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Find the best node for a task based on latency and capacity."""
        best_node: Optional[EdgeNode] = None
        best_latency: float = float("inf")

        for node in self._nodes.values():
            # Check capacity
            cpu_avail = node.cpu_capacity - self._node_cpu_used.get(node.node_id, 0.0)
            mem_avail = node.memory_capacity - self._node_memory_used.get(node.node_id, 0.0)
            if cpu_avail < task.cpu_required or mem_avail < task.memory_required:
                continue

            # Check deadline
            estimated_latency = self._estimate_latency(task, node)
            if estimated_latency > task.deadline_ms:
                continue

            # Pick lowest latency
            if estimated_latency < best_latency:
                best_latency = estimated_latency
                best_node = node

        return best_node

    def _estimate_latency(self, task: EdgeTask, node: EdgeNode) -> float:
        """Estimate total latency for a task on a node.

        Includes network latency and a processing component based on
        data size and CPU requirement.
        """
        # Network latency (round-trip estimate)
        network_latency = node.latency_ms * 2
        # Processing estimate: data transfer + computation
        processing_latency = (task.data_size_kb / 100.0) + (task.cpu_required / 10.0)
        return network_latency + processing_latency

    def _assign_task(self, task: EdgeTask, node: EdgeNode) -> None:
        """Assign a task to a node and update resource usage."""
        self._node_cpu_used[node.node_id] = (
            self._node_cpu_used.get(node.node_id, 0.0) + task.cpu_required
        )
        self._node_memory_used[node.node_id] = (
            self._node_memory_used.get(node.node_id, 0.0) + task.memory_required
        )
        estimated_latency = self._estimate_latency(task, node)
        self._assignments[task.task_id] = TaskAssignment(
            task_id=task.task_id,
            node_id=node.node_id,
            status="assigned",
            estimated_latency_ms=estimated_latency,
        )

    # ------------------------------------------------------------------
    # Task Completion
    # ------------------------------------------------------------------

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed and release its resources."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            assignment = self._assignments.get(task_id)
            if assignment is None or assignment.status != "assigned":
                return False
            node_id = assignment.node_id
            if node_id is not None and node_id in self._nodes:
                task = self._tasks[task_id]
                self._node_cpu_used[node_id] = max(
                    0.0, self._node_cpu_used.get(node_id, 0.0) - task.cpu_required
                )
                self._node_memory_used[node_id] = max(
                    0.0, self._node_memory_used.get(node_id, 0.0) - task.memory_required
                )
            assignment.status = "completed"
            return True

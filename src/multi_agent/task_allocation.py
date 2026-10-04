"""Multi-agent task allocation using greedy and Hungarian algorithms."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AllocationInstance:
    """Task allocation problem instance."""

    agents: List[str]
    tasks: List[str]
    cost_matrix: List[List[float]]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    priorities: Dict[str, int] = field(default_factory=dict)
    deadlines: Dict[str, float] = field(default_factory=dict)
    agent_capabilities: Dict[str, List[str]] = field(default_factory=dict)
    task_requirements: Dict[str, List[str]] = field(default_factory=dict)
    agent_resources: Dict[str, Dict[str, float]] = field(default_factory=dict)
    task_resource_requirements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    time_windows: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.agents)
        m = len(self.tasks)
        if len(self.cost_matrix) != n:
            raise ValueError("Cost matrix must be square")
        for row in self.cost_matrix:
            if len(row) != m:
                raise ValueError("Cost matrix must be square")

        # Validate cost matrix values
        for i, row in enumerate(self.cost_matrix):
            for j, cost in enumerate(row):
                if cost < 0:
                    raise ValueError(f"Negative cost at [{i}][{j}]: {cost}")
                if math.isnan(cost):
                    raise ValueError(f"NaN cost at [{i}][{j}]")
                if math.isinf(cost):
                    raise ValueError(f"Infinite cost at [{i}][{j}]: {cost}")

        # Validate dependencies
        for task_id, deps in self.dependencies.items():
            if task_id not in self.tasks:
                raise ValueError(f"Unknown task in dependencies: {task_id}")
            for dep in deps:
                if dep not in self.tasks:
                    raise ValueError(f"Unknown task in dependencies of {task_id}: {dep}")

        # Check for circular dependencies
        self._check_circular_dependencies()

    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies using DFS."""
        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(task_id: str) -> bool:
            """Returns True if circular dependency found."""
            if visited.get(task_id, 0) == 1:
                return True  # Back edge = cycle
            if visited.get(task_id, 0) == 2:
                return False
            visited[task_id] = 1
            for dep in self.dependencies.get(task_id, []):
                if dfs(dep):
                    return True
            visited[task_id] = 2
            return False

        for task_id in self.tasks:
            if visited.get(task_id, 0) == 0:
                if dfs(task_id):
                    raise ValueError("Circular dependency detected")


@dataclass
class AllocationResult:
    """Task allocation result."""

    allocation: Dict[str, List[str]]
    total_cost: float
    algorithm: str
    unassigned_tasks: Optional[List[str]] = None


class TaskAllocator:
    """Multi-agent task allocator."""

    def __init__(self, algorithm: str = "greedy"):
        self.algorithm = algorithm

    def allocate(self, instance: AllocationInstance) -> AllocationResult:
        """Allocate tasks to agents."""
        if not instance.tasks:
            return AllocationResult(
                allocation={a: [] for a in instance.agents},
                total_cost=0.0,
                algorithm=self.algorithm,
            )

        if self.algorithm == "greedy":
            return self._greedy(instance)
        elif self.algorithm == "hungarian":
            return self._hungarian(instance)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _greedy(self, instance: AllocationInstance) -> AllocationResult:
        """Greedy task allocation — assign each task to cheapest agent."""
        n = len(instance.agents)
        m = len(instance.tasks)
        allocation = {a: [] for a in instance.agents}
        total_cost = 0.0
        unassigned: List[str] = []

        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(
            range(m),
            key=lambda j: instance.priorities.get(instance.tasks[j], 0),
            reverse=True,
        )

        for j in sorted_tasks:
            task_id = instance.tasks[j]

            # Check capability constraints
            if not self._can_assign(instance, j, allocation):
                unassigned.append(task_id)
                continue

            best_agent = 0
            best_cost = instance.cost_matrix[0][j]
            for i in range(1, n):
                if instance.cost_matrix[i][j] < best_cost:
                    # Check capability constraints
                    if self._can_assign(instance, j, allocation, agent_idx=i):
                        best_cost = instance.cost_matrix[i][j]
                        best_agent = i

            allocation[instance.agents[best_agent]].append(task_id)
            total_cost += best_cost

        return AllocationResult(
            allocation=allocation,
            total_cost=total_cost,
            algorithm="greedy",
            unassigned_tasks=unassigned if unassigned else None,
        )

    def _can_assign(
        self,
        instance: AllocationInstance,
        task_idx: int,
        allocation: Dict[str, List[str]],
        agent_idx: Optional[int] = None,
    ) -> bool:
        """Check if a task can be assigned to an agent based on constraints.

        Args:
            instance: The allocation instance.
            task_idx: Index of the task.
            allocation: Current allocation state.
            agent_idx: Index of the agent to check, or None to check all.

        Returns:
            True if the task can be assigned.
        """
        task_id = instance.tasks[task_idx]
        requirements = instance.task_requirements.get(task_id, [])

        agents_to_check = [agent_idx] if agent_idx is not None else range(len(instance.agents))

        for i in agents_to_check:
            agent_id = instance.agents[i]
            capabilities = instance.agent_capabilities.get(agent_id, [])

            # Check capability match
            if requirements and not all(req in capabilities for req in requirements):
                continue

            # Check resource constraints
            if not self._check_resources(instance, agent_id, task_id):
                continue

            return True

        return False

    def _check_resources(
        self,
        instance: AllocationInstance,
        agent_id: str,
        task_id: str,
    ) -> bool:
        """Check if an agent has sufficient resources for a task.

        Args:
            instance: The allocation instance.
            agent_id: The agent to check.
            task_id: The task to check.

        Returns:
            True if the agent has sufficient resources.
        """
        agent_res = instance.agent_resources.get(agent_id, {})
        task_req = instance.task_resource_requirements.get(task_id, {})

        for resource, required in task_req.items():
            available = agent_res.get(resource, 0.0)
            if available < required:
                return False

        return True

    def _hungarian(self, instance: AllocationInstance) -> AllocationResult:
        """Hungarian algorithm for optimal assignment (square matrix only)."""
        n = len(instance.agents)
        m = len(instance.tasks)

        if n != m:
            # Fall back to greedy for non-square
            return self._greedy(instance)

        # Hungarian algorithm implementation
        cost = [row[:] for row in instance.cost_matrix]
        u = [0.0] * (n + 1)
        v = [0.0] * (m + 1)
        p = [0] * (m + 1)
        way = [0] * (m + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (m + 1)
            used = [False] * (m + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # Build result
        allocation = {a: [] for a in instance.agents}
        total_cost = 0.0
        for j in range(1, m + 1):
            if p[j] > 0:
                agent_idx = p[j] - 1
                allocation[instance.agents[agent_idx]].append(instance.tasks[j - 1])
                total_cost += instance.cost_matrix[agent_idx][j - 1]

        return AllocationResult(
            allocation=allocation,
            total_cost=total_cost,
            algorithm="hungarian",
        )

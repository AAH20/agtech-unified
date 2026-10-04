"""Multi-agent task allocation using greedy and Hungarian algorithms."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AllocationInstance:
    """Task allocation problem instance."""
    agents: List[str]
    tasks: List[str]
    cost_matrix: List[List[float]]

    def __post_init__(self):
        n = len(self.agents)
        m = len(self.tasks)
        if len(self.cost_matrix) != n:
            raise ValueError("Cost matrix must be square")
        for row in self.cost_matrix:
            if len(row) != m:
                raise ValueError("Cost matrix must be square")


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

        # For each task, find the cheapest agent
        for j in range(m):
            best_agent = 0
            best_cost = instance.cost_matrix[0][j]
            for i in range(1, n):
                if instance.cost_matrix[i][j] < best_cost:
                    best_cost = instance.cost_matrix[i][j]
                    best_agent = i
            allocation[instance.agents[best_agent]].append(instance.tasks[j])
            total_cost += best_cost

        return AllocationResult(
            allocation=allocation,
            total_cost=total_cost,
            algorithm="greedy",
        )

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
            minv = [float('inf')] * (m + 1)
            used = [False] * (m + 1)

            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
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

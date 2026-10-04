"""Task allocation example: assign tasks to agents using Hungarian algorithm."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi_agent.task_allocation import AllocationInstance, TaskAllocator


def main():
    agents = ["robot-1", "robot-2", "robot-3"]
    tasks = ["weed-A", "weed-B", "weed-C"]
    cost_matrix = [
        [10, 5, 8],
        [7, 12, 6],
        [9, 8, 11],
    ]
    instance = AllocationInstance(agents, tasks, cost_matrix)
    result = TaskAllocator("hungarian").allocate(instance)
    print(f"Assignments: {result.assignments}")
    print(f"Total Cost: {result.total_cost}")


if __name__ == "__main__":
    main()

"""Swarm coordination example: assign tasks to agricultural robots."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi_agent.swarm import Agent, Task, SwarmCoordinator


def main():
    agents = [
        Agent(id="robot-1", position=(0, 0), capacity=10),
        Agent(id="robot-2", position=(5, 5), capacity=10),
        Agent(id="robot-3", position=(10, 0), capacity=10),
    ]
    tasks = [
        Task(id="weed-1", position=(2, 3), priority=1),
        Task(id="weed-2", position=(7, 8), priority=2),
        Task(id="weed-3", position=(12, 2), priority=1),
    ]
    coordinator = SwarmCoordinator(agents)
    assignments = coordinator.assign_tasks(tasks)
    for task_id, agent_id in assignments.items():
        print(f"Task {task_id} -> Agent {agent_id}")


if __name__ == "__main__":
    main()

# Tutorial 03: Swarm Coordination

## Overview

Learn how to coordinate multiple agricultural robots using swarm intelligence.

## Basic Swarm Setup

```python
from src.multi_agent.swarm import Agent, Task, SwarmCoordinator

# Create agents
agents = [
    Agent(id="robot-1", position=(0, 0), capacity=10),
    Agent(id="robot-2", position=(5, 5), capacity=10),
    Agent(id="robot-3", position=(10, 0), capacity=10),
]

# Create tasks
tasks = [
    Task(id="weed-1", position=(2, 3), priority=1),
    Task(id="weed-2", position=(7, 8), priority=2),
    Task(id="weed-3", position=(12, 2), priority=1),
]

# Coordinate
coordinator = SwarmCoordinator(agents)
assignments = coordinator.assign_tasks(tasks)
for task_id, agent_id in assignments.items():
    print(f"Task {task_id} -> Agent {agent_id}")
```

## Byzantine Consensus

```python
from src.multi_agent.consensus import ByzantineConsensus

# 4 nodes, 1 Byzantine fault tolerated
consensus = ByzantineConsensus(num_nodes=4, max_faults=1)
result = consensus.propose(value="harvest-field-A")
print(f"Consensus: {result.status}, Value: {result.value}")
```

## Next Steps

- [Tutorial 04: IoT Data Pipeline](04-iot-data-pipeline.md)
- [Tutorial 05: Digital Twin Simulation](05-digital-twin-simulation.md)

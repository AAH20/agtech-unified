"""Multi-agent orchestrator: subscribes to event_bus and coordinates swarm.

Listens for domain events (sensor readings, alerts) and triggers swarm
task creation and dispatch through SwarmCoordinator.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.multi_agent.swarm import Agent, SwarmCoordinator, Task, TaskStatus

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates multi-agent coordination based on domain events.

    Subscribes to SENSOR_READING_RECEIVED and ALERT_TRIGGERED events,
    creates tasks from them, and dispatches to registered swarm agents.
    """

    def __init__(self, bus: EventBus, coordinator: Optional[SwarmCoordinator] = None) -> None:
        self._bus = bus
        self._coordinator = coordinator or SwarmCoordinator()
        self._tasks: Dict[str, Task] = {}
        self._task_counter = 0

        # Subscribe to relevant events
        self._bus.subscribe(EventType.SENSOR_READING_RECEIVED, self._on_sensor_reading)
        self._bus.subscribe(EventType.ALERT_TRIGGERED, self._on_alert)

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def coordinator(self) -> SwarmCoordinator:
        return self._coordinator

    def register_agent(self, agent_id: str, capabilities: Optional[List[str]] = None) -> Agent:
        """Register an agent with the swarm."""
        return self._coordinator.register_agent(agent_id, capabilities)

    def get_agents(self) -> List[Agent]:
        """Return all registered agents."""
        return list(self._coordinator._agents.values())

    def submit_task(self, task_id: str, requirements: Optional[List[str]] = None) -> Task:
        """Submit a task to the swarm."""
        task = self._coordinator.submit_task(task_id, requirements)
        self._tasks[task_id] = task
        return task

    def get_pending_tasks(self) -> List[Task]:
        """Return all pending tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def dispatch_tasks(self) -> Dict[str, str]:
        """Dispatch pending tasks to available agents."""
        return self._coordinator.distribute_tasks()

    def _on_sensor_reading(self, event: DomainEvent) -> None:
        """Handle sensor reading events by creating monitoring tasks."""
        payload = event.payload
        metric = payload.get("metric", "unknown")
        value = payload.get("value", 0.0)

        self._task_counter += 1
        task_id = f"task_sensor_{self._task_counter}"
        requirements = ["scan"]

        if metric == "soil_moisture" and value < 0.3:
            requirements.append("spray")

        self.submit_task(task_id, requirements=requirements)
        logger.info("Created task %s from sensor reading: %s=%s", task_id, metric, value)

    def _on_alert(self, event: DomainEvent) -> None:
        """Handle alert events by creating urgent tasks."""
        payload = event.payload
        metric = payload.get("metric", "unknown")

        self._task_counter += 1
        task_id = f"task_alert_{self._task_counter}"
        requirements = ["scan", "spray"]

        self.submit_task(task_id, requirements=requirements)
        logger.info("Created urgent task %s from alert: %s", task_id, metric)

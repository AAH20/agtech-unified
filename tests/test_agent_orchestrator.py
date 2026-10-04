"""Tests for multi-agent orchestrator: event_bus → swarm coordination."""

from src.integration.event_bus import EventBus, EventType
from src.multi_agent.agent_orchestrator import AgentOrchestrator


class TestAgentOrchestrator:
    """AgentOrchestrator subscribes to events and coordinates swarm tasks."""

    def test_subscribe_to_sensor_events(self):
        """Orchestrator subscribes to SENSOR_READING_RECEIVED events."""
        bus = EventBus()
        AgentOrchestrator(bus=bus)

        assert bus.subscriber_count(EventType.SENSOR_READING_RECEIVED) >= 1

    def test_sensor_event_triggers_task(self):
        """A sensor reading event triggers swarm task creation."""
        bus = EventBus()
        orch = AgentOrchestrator(bus=bus)
        orch.register_agent("drone_1", capabilities=["spray", "scan"])

        from src.integration.event_bus import DomainEvent

        bus.publish(
            DomainEvent(
                event_type=EventType.SENSOR_READING_RECEIVED,
                source="iot.orchestrator",
                payload={"sensor_id": "s1", "value": 0.15, "metric": "soil_moisture"},
            )
        )

        tasks = orch.get_pending_tasks()
        assert len(tasks) >= 1

    def test_register_agent(self):
        """Agents can be registered with capabilities."""
        bus = EventBus()
        orch = AgentOrchestrator(bus=bus)

        agent = orch.register_agent("drone_1", capabilities=["spray"])

        assert agent.agent_id == "drone_1"
        assert "spray" in agent.capabilities

    def test_get_registered_agents(self):
        """Registered agents can be listed."""
        bus = EventBus()
        orch = AgentOrchestrator(bus=bus)
        orch.register_agent("drone_1", capabilities=["spray"])
        orch.register_agent("drone_2", capabilities=["scan"])

        agents = orch.get_agents()
        assert len(agents) == 2

    def test_dispatch_tasks(self):
        """Tasks are dispatched to capable agents."""
        bus = EventBus()
        orch = AgentOrchestrator(bus=bus)
        orch.register_agent("drone_1", capabilities=["spray"])
        orch.submit_task("task_1", requirements=["spray"])

        assignments = orch.dispatch_tasks()

        assert "task_1" in assignments
        assert assignments["task_1"] == "drone_1"

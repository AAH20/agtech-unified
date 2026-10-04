"""Tests for digital twin orchestrator: simulation → event_bus."""

from src.digital_twin.simulator import SimulationState
from src.digital_twin.twin_orchestrator import TwinOrchestrator
from src.integration.event_bus import EventBus, EventType


class TestTwinOrchestrator:
    """TwinOrchestrator publishes simulation results to the event bus."""

    def test_run_simulation_publishes_event(self):
        """Running a simulation publishes SIMULATION_COMPLETED."""
        bus = EventBus()
        orch = TwinOrchestrator(bus=bus)
        initial = SimulationState(
            soil_moisture=0.5, temperature=25.0, crop_height=0.1, nutrient_level=0.5
        )

        result = orch.run_simulation(initial, days=10)

        events = bus.get_history(EventType.SIMULATION_COMPLETED)
        assert len(events) == 1
        assert events[0].source == "digital_twin.orchestrator"
        assert events[0].payload["days_simulated"] == 10
        assert result.days_simulated == 10

    def test_subscribe_to_simulation_completed(self):
        """Handlers can subscribe to SIMULATION_COMPLETED events."""
        bus = EventBus()
        orch = TwinOrchestrator(bus=bus)
        received = []
        orch.subscribe(EventType.SIMULATION_COMPLETED, received.append)

        initial = SimulationState(
            soil_moisture=0.5, temperature=25.0, crop_height=0.1, nutrient_level=0.5
        )
        orch.run_simulation(initial, days=5)

        assert len(received) == 1

    def test_simulation_result_in_payload(self):
        """The event payload contains the simulation result summary."""
        bus = EventBus()
        orch = TwinOrchestrator(bus=bus)
        initial = SimulationState(
            soil_moisture=0.5, temperature=25.0, crop_height=0.1, nutrient_level=0.5
        )

        orch.run_simulation(initial, days=7)

        event = bus.get_history(EventType.SIMULATION_COMPLETED)[0]
        assert "final_state" in event.payload
        assert "total_growth" in event.payload

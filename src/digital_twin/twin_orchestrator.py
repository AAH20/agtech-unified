"""Digital twin orchestrator: publishes simulation results to the event bus.

Runs simulations via DigitalTwin and publishes SIMULATION_COMPLETED events
so decision_support and other modules can react to simulation outcomes.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.digital_twin.simulator import DigitalTwin, SimulationResult, SimulationState
from src.integration.event_bus import DomainEvent, EventBus, EventType

logger = logging.getLogger(__name__)


class TwinOrchestrator:
    """Orchestrates digital twin simulations and publishes results.

    Wraps DigitalTwin to run simulations and automatically publish
    SIMULATION_COMPLETED events with the final state and growth metrics.
    """

    def __init__(self, bus: EventBus, twin: Optional[DigitalTwin] = None) -> None:
        self._bus = bus
        self._twin = twin or DigitalTwin()

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def twin(self) -> DigitalTwin:
        return self._twin

    def run_simulation(self, initial_state: SimulationState, days: int) -> SimulationResult:
        """Run a simulation and publish the result to the event bus.

        Args:
            initial_state: Starting simulation state.
            days: Number of days to simulate.

        Returns:
            The SimulationResult from the digital twin.
        """
        result = self._twin.simulate(initial_state, days)

        event = DomainEvent(
            event_type=EventType.SIMULATION_COMPLETED,
            source="digital_twin.orchestrator",
            payload={
                "days_simulated": result.days_simulated,
                "algorithm": result.algorithm,
                "total_growth": result.total_growth,
                "final_state": {
                    "soil_moisture": result.final_state.soil_moisture,
                    "temperature": result.final_state.temperature,
                    "crop_height": result.final_state.crop_height,
                    "nutrient_level": result.final_state.nutrient_level,
                },
            },
        )
        self._bus.publish(event)
        return result

    def subscribe(self, event_type: str, handler) -> None:
        """Subscribe a handler to an event type on the bus."""
        self._bus.subscribe(event_type, handler)

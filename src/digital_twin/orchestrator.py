"""Unified digital twin orchestrator facade.

Coordinates all subsystems (simulator, water balance, nutrient cycling,
knowledge graph, ontology, event bus, sensor ingestion) into a cohesive
digital twin with a single entry point.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.digital_twin.event_engine import DiscreteEventSimulator, Event, EventType
from src.digital_twin.sensor_ingestion import SensorDataIngestion, SensorReading, StateReconciler
from src.digital_twin.shared_state import DigitalTwinState, StateSnapshot
from src.integration.event_bus import DomainEvent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the digital twin orchestrator."""

    enable_water_balance: bool = True
    enable_nutrient_cycling: bool = True
    enable_event_driven: bool = True
    enable_sensor_ingestion: bool = True
    enable_knowledge_graph: bool = True
    enable_ontology: bool = True
    auto_reconcile_sensors: bool = True
    publish_simulation_events: bool = True


class DigitalTwinOrchestrator:
    """Unified facade for the agricultural digital twin.

    Coordinates all subsystems and provides a single API for
    simulation, sensor ingestion, event handling, and state management.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._run_id = str(uuid.uuid4())
        self._created_at = time.time()

        # Core state
        self._state = DigitalTwinState()
        self._history: List[StateSnapshot] = []

        # Submodules (lazy-loaded)
        self._simulator = None
        self._water_balance = None
        self._nutrient_cycling = None
        self._knowledge_graph = None
        self._ontology = None
        self._event_bus = None
        self._event_simulator = None
        self._sensor_ingestion = None
        self._state_reconciler = None

        # Initialize enabled submodules
        self._init_submodules()

    def _init_submodules(self) -> None:
        """Initialize enabled submodules."""
        if self.config.enable_water_balance:
            from src.digital_twin.water_balance import WaterBalanceModel

            self._water_balance = WaterBalanceModel()

        if self.config.enable_nutrient_cycling:
            from src.digital_twin.nutrient_cycling import NutrientCyclingModel

            self._nutrient_cycling = NutrientCyclingModel()

        if self.config.enable_knowledge_graph:
            from src.digital_twin.knowledge_graph import AgriKnowledgeGraph

            self._knowledge_graph = AgriKnowledgeGraph()

        if self.config.enable_ontology:
            from src.digital_twin.ontology import AGROVOCOntology

            self._ontology = AGROVOCOntology()

        if self.config.enable_event_driven:
            self._event_simulator = DiscreteEventSimulator()

        if self.config.enable_sensor_ingestion:
            self._sensor_ingestion = SensorDataIngestion()
            self._state_reconciler = StateReconciler()

        # Event bus
        from src.integration.event_bus import EventBus

        self._event_bus = EventBus()

    @property
    def state(self) -> DigitalTwinState:
        """Current digital twin state."""
        return self._state

    @property
    def run_id(self) -> str:
        """Unique run identifier."""
        return self._run_id

    def get_state(self) -> DigitalTwinState:
        """Get current state."""
        return self._state

    def set_state(self, state: DigitalTwinState) -> None:
        """Set current state."""
        self._state = state

    def get_submodule(self, name: str) -> Any:
        """Get a submodule by name.

        Args:
            name: One of 'simulator', 'water_balance', 'nutrient_cycling',
                  'knowledge_graph', 'ontology', 'event_bus', 'event_simulator',
                  'sensor_ingestion'.

        Returns:
            The requested submodule instance.
        """
        module_map = {
            "simulator": self._simulator,
            "water_balance": self._water_balance,
            "nutrient_cycling": self._nutrient_cycling,
            "knowledge_graph": self._knowledge_graph,
            "ontology": self._ontology,
            "event_bus": self._event_bus,
            "event_simulator": self._event_simulator,
            "sensor_ingestion": self._sensor_ingestion,
        }
        return module_map.get(name)

    def step(self) -> None:
        """Advance simulation by one day."""
        sim_state = self._state.to_simulation_state()
        if self._simulator is None:
            from src.digital_twin.simulator import DigitalTwin

            self._simulator = DigitalTwin()
        result = self._simulator.simulate(sim_state, days=1)
        self._state = DigitalTwinState.from_simulation_state(result.final_state)
        self._state.day += 1
        self._record_snapshot()

    def step_n(self, days: int) -> None:
        """Advance simulation by N days."""
        for _ in range(days):
            self.step()

    def simulate(self, days: int, events: Optional[List[Event]] = None) -> Dict[str, Any]:
        """Run full simulation.

        Args:
            days: Number of days to simulate.
            events: Optional list of events to inject.

        Returns:
            Dict with simulation results.
        """
        from src.digital_twin.simulator import DigitalTwin

        if self._simulator is None:
            self._simulator = DigitalTwin()

        sim_state = self._state.to_simulation_state()
        result = self._simulator.simulate(sim_state, days)

        self._state = DigitalTwinState.from_simulation_state(result.final_state)
        self._state.day += days

        # Publish event
        if self.config.publish_simulation_events and self._event_bus:
            self._event_bus.publish(
                DomainEvent(
                    event_type="simulation_completed",
                    source="digital_twin.orchestrator",
                    payload={
                        "days_simulated": days,
                        "final_height": result.final_state.crop_height,
                    },
                )
            )

        return {
            "days_simulated": days,
            "final_state": self._state.to_dict(),
            "total_growth": result.total_growth,
            "history_length": len(result.history),
        }

    def add_sensor_reading(self, reading: SensorReading) -> None:
        """Add sensor reading and reconcile state.

        Args:
            reading: Sensor reading to ingest.
        """
        if self._sensor_ingestion is None:
            return

        self._sensor_ingestion.ingest(reading)

        if self.config.auto_reconcile_sensors and self._state_reconciler:
            state_dict = self._state.to_dict()
            self._state_reconciler.reconcile(state_dict, reading)
            self._state = DigitalTwinState.from_dict(state_dict)

    def publish_event(self, event: Event) -> None:
        """Publish event to the event bus.

        Args:
            event: Event to publish.
        """
        if self._event_bus:
            domain_event = DomainEvent(
                event_type=event.event_type.value,
                source="digital_twin.orchestrator",
                payload=event.payload,
            )
            self._event_bus.publish(domain_event)

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to events.

        Args:
            event_type: Type of event to subscribe to.
            handler: Handler function.
        """
        if self._event_bus:
            self._event_bus.subscribe(event_type.value, handler)

    def reset(self) -> None:
        """Reset orchestrator state."""
        self._state = DigitalTwinState()
        self._history.clear()
        self._run_id = str(uuid.uuid4())

    def get_history(self) -> List[StateSnapshot]:
        """Get simulation history."""
        return list(self._history)

    def export_state(self) -> Dict[str, Any]:
        """Export current state to dict."""
        return self._state.to_dict()

    def import_state(self, data: Dict[str, Any]) -> None:
        """Import state from dict."""
        self._state = DigitalTwinState.from_dict(data)

    def run_scenario(self, name: str, days: int) -> Dict[str, Any]:
        """Run a named scenario.

        Args:
            name: Scenario name.
            days: Number of days to simulate.

        Returns:
            Scenario results.
        """
        # Save current state
        saved_state = self._state.clone()

        # Apply scenario parameters
        scenarios = {
            "default": {},
            "high_input": {"nitrogen": 100.0, "phosphorus": 20.0, "potassium": 80.0},
            "drought": {"soil_moisture": 0.1},
            "optimal": {"soil_moisture": 0.4, "temperature": 25.0, "nitrogen": 80.0},
        }
        params = scenarios.get(name, {})
        for key, value in params.items():
            setattr(self._state, key, value)

        result = self.simulate(days)
        result["scenario"] = name

        # Restore state
        self._state = saved_state
        return result

    def compare_scenarios(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two scenario results.

        Args:
            result1: First scenario result.
            result2: Second scenario result.

        Returns:
            Comparison dict.
        """
        s1 = result1.get("final_state", {})
        s2 = result2.get("final_state", {})
        return {
            "height_diff": s1.get("crop_height", 0) - s2.get("crop_height", 0),
            "moisture_diff": s1.get("soil_moisture", 0) - s2.get("soil_moisture", 0),
            "growth_diff": result1.get("total_growth", 0) - result2.get("total_growth", 0),
        }

    def health_check(self) -> Dict[str, Any]:
        """Health check returns status."""
        return {
            "status": "healthy",
            "run_id": self._run_id,
            "uptime": time.time() - self._created_at,
            "submodules": {
                "water_balance": self._water_balance is not None,
                "nutrient_cycling": self._nutrient_cycling is not None,
                "knowledge_graph": self._knowledge_graph is not None,
                "ontology": self._ontology is not None,
                "event_driven": self._event_simulator is not None,
                "sensor_ingestion": self._sensor_ingestion is not None,
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics."""
        return {
            "days_simulated": self._state.day,
            "current_height": self._state.crop_height,
            "current_moisture": self._state.soil_moisture,
            "nutrient_level": self._state.nutrient_level,
            "stress_factor": self._state.combined_stress_factor(),
            "history_length": len(self._history),
        }

    def _record_snapshot(self) -> None:
        """Record state snapshot."""
        snapshot = StateSnapshot(
            state=self._state.clone(),
            day=self._state.day,
        )
        self._history.append(snapshot)

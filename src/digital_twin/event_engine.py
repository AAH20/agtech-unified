"""Event-driven simulation engine for agricultural digital twin.

Provides discrete event simulation with priority queue, event handlers,
and support for irrigation, fertilization, pest outbreaks, and other
management interventions.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of simulation events."""

    IRRIGATION = "irrigation"
    FERTILIZATION = "fertilization"
    PEST_OUTBREAK = "pest_outbreak"
    DISEASE_OUTBREAK = "disease_outbreak"
    WEED_CONTROL = "weed_control"
    HARVEST = "harvest"
    WEATHER_EXTREME = "weather_extreme"
    GROWTH_STAGE_CHANGE = "growth_stage_change"
    SENSOR_READING = "sensor_reading"
    SIMULATION_COMPLETED = "simulation_completed"
    CUSTOM = "custom"


@dataclass
class Event:
    """A discrete simulation event."""

    event_type: EventType
    day: int
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1 = highest, 9 = lowest

    def __lt__(self, other: Event) -> bool:
        """Compare events for priority queue ordering."""
        if self.day != other.day:
            return self.day < other.day
        return self.priority < other.priority


@dataclass
class SimulationEvent:
    """Event with a handler callback."""

    event_type: EventType
    day: int
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    payload: Dict[str, Any] = field(default_factory=dict)

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this event's handler to the state."""
        return self.handler(state)


class EventHandler:
    """Base class for event handlers."""

    def __init__(self, event_type: EventType):
        self.event_type = event_type

    def handle(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an event. Override in subclasses."""
        return state


class EventQueue:
    """Priority queue for simulation events."""

    def __init__(self):
        self._queue: List[tuple] = []
        self._counter = 0  # Tie-breaker for stable ordering

    def push(self, event: Event) -> None:
        """Add event to queue."""
        heapq.heappush(self._queue, (event.day, event.priority, self._counter, event))
        self._counter += 1

    def pop(self) -> Optional[Event]:
        """Remove and return next event."""
        if not self._queue:
            return None
        return heapq.heappop(self._queue)[3]

    def peek(self) -> Optional[Event]:
        """View next event without removing."""
        if not self._queue:
            return None
        return self._queue[0][3]

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def size(self) -> int:
        """Number of events in queue."""
        return len(self._queue)

    def clear(self) -> None:
        """Remove all events."""
        self._queue.clear()

    def events_for_day(self, day: int) -> List[Event]:
        """Get all events scheduled for a specific day."""
        return [item[3] for item in self._queue if item[0] == day]


class DiscreteEventSimulator:
    """Discrete event simulator with priority queue.

    Advances simulation day by day, processing scheduled events
    and updating state accordingly.
    """

    def __init__(self):
        self._queue = EventQueue()
        self._event_history: List[Event] = []
        self._handlers: Dict[EventType, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default event handlers."""
        self._handlers[EventType.IRRIGATION] = self._handle_irrigation
        self._handlers[EventType.FERTILIZATION] = self._handle_fertilization
        self._handlers[EventType.PEST_OUTBREAK] = self._handle_pest_outbreak
        self._handlers[EventType.DISEASE_OUTBREAK] = self._handle_disease_outbreak
        self._handlers[EventType.WEED_CONTROL] = self._handle_weed_control
        self._handlers[EventType.WEATHER_EXTREME] = self._handle_weather_extreme

    def schedule(self, event: Event) -> None:
        """Schedule an event."""
        self._queue.push(event)

    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register a custom event handler."""
        self._handlers[event_type] = handler

    def simulate(
        self,
        days: int,
        initial_state: Dict[str, Any],
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run discrete event simulation.

        Args:
            days: Number of days to simulate.
            initial_state: Starting state dict.
            callback: Optional callback(day, state) called each day.

        Returns:
            Dict with final_state, days_simulated, and event_history.
        """
        state = dict(initial_state)
        self._event_history.clear()

        for day in range(days):
            # Process events for this day
            for event in self._queue.events_for_day(day):
                handler = self._handlers.get(event.event_type)
                if handler:
                    state = handler(event, state)
                self._event_history.append(event)

            # Default daily dynamics
            state = self._daily_update(state)

            if callback:
                callback(day, state)

        return {
            "final_state": state,
            "days_simulated": days,
            "event_history": list(self._event_history),
        }

    def _daily_update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Default daily state update."""
        # Soil moisture depletes
        stress = state.get("stress_factor", 0.5)
        state["soil_moisture"] = max(0.0, state.get("soil_moisture", 0.3) - 0.02 * stress)

        # Pest pressure evolves
        pest = state.get("pest_pressure", 0.0)
        if pest > 0:
            # Logistic growth of pest population
            state["pest_pressure"] = min(1.0, pest + 0.05 * pest * (1 - pest))

        return state

    def _handle_irrigation(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle irrigation event."""
        amount_mm = event.payload.get("amount_mm", 10.0)
        # Convert mm to m³/m³ (simplified)
        root_depth = state.get("root_depth", 0.5)
        mm_to_m3m3 = 1.0 / (root_depth * 1000.0)
        state["soil_moisture"] = min(1.0, state.get("soil_moisture", 0.3) + amount_mm * mm_to_m3m3)
        return state

    def _handle_fertilization(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fertilization event."""
        n = event.payload.get("nitrogen", 0.0)
        p = event.payload.get("phosphorus", 0.0)
        k = event.payload.get("potassium", 0.0)
        state["nitrogen"] = state.get("nitrogen", 0.0) + n
        state["phosphorus"] = state.get("phosphorus", 0.0) + p
        state["potassium"] = state.get("potassium", 0.0) + k
        # Update aggregate nutrient level
        state["nutrient_level"] = min(
            1.0,
            (state["nitrogen"] / 100.0 + state["phosphorus"] / 20.0 + state["potassium"] / 80.0)
            / 3.0,
        )
        return state

    def _handle_pest_outbreak(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pest outbreak event."""
        severity = event.payload.get("severity", 0.5)
        state["pest_pressure"] = min(1.0, state.get("pest_pressure", 0.0) + severity)
        return state

    def _handle_disease_outbreak(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle disease outbreak event."""
        severity = event.payload.get("severity", 0.5)
        state["disease_pressure"] = min(1.0, state.get("disease_pressure", 0.0) + severity)
        return state

    def _handle_weed_control(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weed control event."""
        effectiveness = event.payload.get("effectiveness", 0.8)
        state["weed_pressure"] = max(0.0, state.get("weed_pressure", 0.0) * (1 - effectiveness))
        return state

    def _handle_weather_extreme(self, event: Event, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle extreme weather event."""
        event_subtype = event.payload.get("subtype", "heat")
        if event_subtype == "heat":
            state["temperature"] = event.payload.get("temperature", 40.0)
        elif event_subtype == "frost":
            state["temperature"] = event.payload.get("temperature", -2.0)
        elif event_subtype == "heavy_rain":
            amount = event.payload.get("amount_mm", 50.0)
            state["soil_moisture"] = min(1.0, state.get("soil_moisture", 0.3) + amount / 500.0)
        return state

    def get_event_history(self) -> List[Event]:
        """Return history of processed events."""
        return list(self._event_history)

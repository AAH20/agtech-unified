"""Cross-module integration layer: event bus, unified optimizer, shared farm state."""

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.integration.farm_state import FarmState
from src.integration.unified_optimizer import (
    OptimizationProblem,
    OptimizationResult,
    SolverType,
    UnifiedOptimizer,
)

__all__ = [
    "EventType",
    "DomainEvent",
    "EventBus",
    "FarmState",
    "SolverType",
    "OptimizationProblem",
    "OptimizationResult",
    "UnifiedOptimizer",
]

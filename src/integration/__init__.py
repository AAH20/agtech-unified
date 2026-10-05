"""Cross-module integration layer: event bus, unified optimizer, shared farm state."""

from src.integration.event_bus import DomainEvent, EventBus, EventType
from src.integration.farm_state import FarmState
from src.integration.union_find import UnionFind

# UnifiedOptimizer imported lazily — pulls in GPU solvers which need torch
try:
    from src.integration.unified_optimizer import (
        OptimizationProblem,
        OptimizationResult,
        SolverType,
        UnifiedOptimizer,
    )
except ImportError:
    OptimizationProblem = None
    OptimizationResult = None
    SolverType = None
    UnifiedOptimizer = None

__all__ = [
    "EventType",
    "DomainEvent",
    "EventBus",
    "FarmState",
    "SolverType",
    "OptimizationProblem",
    "OptimizationResult",
    "UnifiedOptimizer",
    "UnionFind",
]

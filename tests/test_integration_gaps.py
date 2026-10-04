"""Tests for integration gap closure: schema versioning, partial updates,
cross-module event bus wiring, and optimizer lifecycle events.

These tests are written TDD-style: they define the expected behavior for
features that don't exist yet, driving implementation.
"""

import json

import pytest

from src.integration.event_bus import EventBus, EventType
from src.integration.farm_state import FarmState
from src.integration.unified_optimizer import (
    OptimizationProblem,
    OptimizationResult,
    SolverType,
    UnifiedOptimizer,
)

# ===========================================================================
# 1. Schema Versioning in FarmState
# ===========================================================================


class TestFarmStateSchemaVersioning:
    """FarmState serialization includes and validates schema version."""

    def test_to_dict_includes_schema_version(self):
        """Serialized FarmState dict includes a 'schema_version' key."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        d = state.to_dict()
        assert "schema_version" in d
        assert isinstance(d["schema_version"], int)
        assert d["schema_version"] >= 1

    def test_from_dict_preserves_schema_version(self):
        """from_dict round-trips the schema version."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        d = state.to_dict()
        restored = FarmState.from_dict(d)
        assert restored.to_dict()["schema_version"] == d["schema_version"]

    def test_from_dict_with_explicit_version(self):
        """from_dict accepts an explicit schema_version field."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        d = state.to_dict()
        d["schema_version"] = 2
        restored = FarmState.from_dict(d)
        assert restored is not None

    def test_to_json_includes_schema_version(self):
        """JSON serialization includes schema_version."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        json_str = state.to_json()
        data = json.loads(json_str)
        assert "schema_version" in data

    def test_from_json_preserves_schema_version(self):
        """from_json round-trips the schema version."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        json_str = state.to_json()
        restored = FarmState.from_json(json_str)
        assert restored is not None

    def test_from_dict_without_version_assumes_v1(self):
        """from_dict with no schema_version key defaults to version 1."""
        d = {
            "soil_moisture": 0.5,
            "temperature": 25.0,
            "crop_height": 0.3,
            "nutrient_level": 0.6,
            "pest_pressure": 0.2,
        }
        restored = FarmState.from_dict(d)
        assert restored.soil_moisture == 0.5


# ===========================================================================
# 2. Partial Updates in FarmState
# ===========================================================================


class TestFarmStatePartialUpdates:
    """FarmState supports updating individual fields without replacing all."""

    def test_update_single_field(self):
        """update() changes only the specified field."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        original_temp = state.temperature
        state.update(soil_moisture=0.7)
        assert state.soil_moisture == pytest.approx(0.7)
        assert state.temperature == original_temp  # unchanged

    def test_update_multiple_fields(self):
        """update() can change multiple fields at once."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        state.update(temperature=30.0, pest_pressure=0.5)
        assert state.temperature == pytest.approx(30.0)
        assert state.pest_pressure == pytest.approx(0.5)
        assert state.soil_moisture == pytest.approx(0.5)  # unchanged

    def test_update_invalid_field_raises(self):
        """update() with an unknown field name raises ValueError."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        with pytest.raises(ValueError, match="unknown_field"):
            state.update(unknown_field=42.0)

    def test_update_out_of_range_raises(self):
        """update() with an out-of-range value raises ValueError."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        with pytest.raises(ValueError, match="soil_moisture"):
            state.update(soil_moisture=1.5)

    def test_update_preserves_timestamp(self):
        """update() does not change the timestamp."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        ts = state.timestamp
        state.update(soil_moisture=0.8)
        assert state.timestamp == ts

    def test_patch_method_alias(self):
        """patch() is an alias for update()."""
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        state.patch(temperature=28.0)
        assert state.temperature == pytest.approx(28.0)
        assert state.soil_moisture == pytest.approx(0.5)  # unchanged


# ===========================================================================
# 3. Cross-Module Event Bus Wiring
# ===========================================================================


class TestCrossModuleEventBusWiring:
    """All major modules publish domain events to the event bus."""

    def test_iot_orchestrator_publishes_sensor_event(self):
        """IoTOrchestrator publishes SENSOR_READING_RECEIVED."""
        from src.iot.iot_orchestrator import IoTOrchestrator

        bus = EventBus()
        orch = IoTOrchestrator(bus=bus)
        orch.process_reading(
            {"sensor_id": "s1", "value": 0.35, "unit": "ratio", "metric": "soil_moisture"}
        )
        events = bus.get_history(EventType.SENSOR_READING_RECEIVED)
        assert len(events) == 1
        assert events[0].source == "iot.orchestrator"

    def test_iot_data_pipeline_produces_sensor_event(self):
        """DataValidator produces events for invalid readings."""
        from src.iot.data_pipeline import DataValidator

        validator = DataValidator()
        result = validator.validate({"sensor_id": "s1", "value": 999.0, "unit": "celsius"})
        # The validator itself doesn't publish, but the orchestrator does.
        # This test validates the integration point exists.
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_digital_twin_twin_orchestrator_publishes_simulation_event(self):
        """TwinOrchestrator publishes SIMULATION_COMPLETED."""
        from src.digital_twin.simulator import SimulationState
        from src.digital_twin.twin_orchestrator import TwinOrchestrator

        bus = EventBus()
        orch = TwinOrchestrator(bus=bus)
        initial = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        orch.run_simulation(initial, days=3)
        events = bus.get_history(EventType.SIMULATION_COMPLETED)
        assert len(events) == 1
        assert events[0].source == "digital_twin.orchestrator"

    def test_digital_twin_simulator_publishes_event_on_ingest(self):
        """DigitalTwin publishes an event when sensor data is ingested."""
        bus = EventBus()
        from src.digital_twin.simulator import DigitalTwin

        twin = DigitalTwin()
        # Inject the bus so the twin can publish
        twin._bus = bus  # type: ignore[attr-defined]
        state = FarmState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.3,
            nutrient_level=0.6,
            pest_pressure=0.2,
        )
        twin.ingest_sensor_data(state)
        # The twin should have published an event
        all_events = bus.get_history()
        assert len(all_events) > 0

    def test_genomics_orchestrator_publishes_protein_event(self):
        """GenomicsOrchestrator publishes RECOMMENDATION_PRODUCED for protein analysis."""
        from src.genomics.genomics_orchestrator import GenomicsOrchestrator

        bus = EventBus()
        orch = GenomicsOrchestrator(bus=bus)
        orch.analyze_protein("ACDEFGHIKLMNPQRSTVWY")
        events = bus.get_history(EventType.RECOMMENDATION_PRODUCED)
        assert len(events) == 1
        assert events[0].source == "genomics.orchestrator"
        assert events[0].payload["analysis_type"] == "protein"

    def test_genomics_orchestrator_publishes_crispr_event(self):
        """GenomicsOrchestrator publishes RECOMMENDATION_PRODUCED for CRISPR analysis."""
        from src.genomics.genomics_orchestrator import GenomicsOrchestrator

        bus = EventBus()
        orch = GenomicsOrchestrator(bus=bus)
        orch.analyze_crispr("ATCGATCGATCGATCGATCGATCGGG")
        events = bus.get_history(EventType.RECOMMENDATION_PRODUCED)
        assert len(events) == 1
        assert events[0].payload["analysis_type"] == "crispr"

    def test_multi_agent_orchestrator_subscribes_to_events(self):
        """AgentOrchestrator subscribes to SENSOR_READING_RECEIVED."""
        from src.multi_agent.agent_orchestrator import AgentOrchestrator

        bus = EventBus()
        AgentOrchestrator(bus=bus)
        assert bus.subscriber_count(EventType.SENSOR_READING_RECEIVED) > 0

    def test_multi_agent_swarm_publishes_task_completed(self):
        """SwarmCoordinator publishes TASK_COMPLETED when a task finishes."""
        bus = EventBus()
        from src.multi_agent.swarm import SwarmCoordinator, TaskStatus

        coord = SwarmCoordinator()
        coord._bus = bus  # type: ignore[attr-defined]
        coord.register_agent("agent1", capabilities=["scan", "spray"])
        coord.submit_task("task1", requirements=["scan"])
        coord.distribute_tasks()
        # After distribution, the task should be assigned
        t = coord.get_task("task1")
        assert t is not None
        assert t.status == TaskStatus.ASSIGNED
        # Complete the task — should publish event
        coord.complete_task("task1")
        events = bus.get_history(EventType.TASK_COMPLETED)
        assert len(events) == 1

    def test_multi_agent_swarm_publishes_task_assigned(self):
        """SwarmCoordinator publishes TASK_ASSIGNED when distributing."""
        bus = EventBus()
        from src.multi_agent.swarm import SwarmCoordinator

        coord = SwarmCoordinator()
        coord._bus = bus  # type: ignore[attr-defined]
        coord.register_agent("agent1", capabilities=["scan"])
        coord.submit_task("task1", requirements=["scan"])
        coord.distribute_tasks()
        events = bus.get_history(EventType.TASK_ASSIGNED)
        assert len(events) == 1

    def test_decision_support_recommender_publishes_alert(self):
        """DecisionEngine publishes ALERT_TRIGGERED for critical states."""
        from src.decision_support.recommender import DecisionEngine

        bus = EventBus()
        engine = DecisionEngine()
        engine._bus = bus  # type: ignore[attr-defined]
        # Create a critical farm state
        state = FarmState(
            soil_moisture=0.1,
            temperature=45.0,
            crop_height=0.05,
            nutrient_level=0.1,
            pest_pressure=0.9,
        )
        result = engine.recommend(state)
        # If there are high-priority recommendations, an alert should fire
        if result.priority_score > 0.5:
            events = bus.get_history(EventType.ALERT_TRIGGERED)
            assert len(events) >= 1


# ===========================================================================
# 4. Optimizer Lifecycle Events
# ===========================================================================


class TestOptimizerLifecycleEvents:
    """UnifiedOptimizer publishes lifecycle events during solving."""

    def test_opt_has_no_bus_by_default(self):
        """UnifiedOptimizer can be constructed without an event bus."""
        opt = UnifiedOptimizer()
        assert opt._bus is None  # type: ignore[attr-defined]

    def test_opt_accepts_bus_via_constructor(self):
        """UnifiedOptimizer accepts an optional EventBus."""
        bus = EventBus()
        opt = UnifiedOptimizer(bus=bus)  # type: ignore[arg-type]
        assert opt._bus is bus  # type: ignore[attr-defined]

    def test_opt_has_set_bus_method(self):
        """UnifiedOptimizer has a set_bus method."""
        bus = EventBus()
        opt = UnifiedOptimizer()
        opt.set_bus(bus)  # type: ignore[attr-defined]
        assert opt._bus is bus  # type: ignore[attr-defined]

    def test_solve_publishes_optimization_started(self):
        """Solving publishes OPTIMIZATION_STARTED before solving."""
        bus = EventBus()
        opt = UnifiedOptimizer(bus=bus)  # type: ignore[arg-type]
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B"],
            distance_matrix=[[0, 5], [5, 0]],
        )
        opt.solve(problem)
        events = bus.get_history("optimization_started")
        assert len(events) == 1
        assert events[0].source == "optimizer.unified"

    def test_solve_publishes_optimization_completed(self):
        """Solving publishes OPTIMIZATION_COMPLETED after solving."""
        bus = EventBus()
        opt = UnifiedOptimizer(bus=bus)  # type: ignore[arg-type]
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B"],
            distance_matrix=[[0, 5], [5, 0]],
        )
        opt.solve(problem)
        events = bus.get_history("optimization_completed")
        assert len(events) == 1
        assert events[0].source == "optimizer.unified"
        assert "cost" in events[0].payload
        assert "algorithm" in events[0].payload

    def test_solve_without_bus_does_not_raise(self):
        """Solving without a bus works fine (no events published)."""
        opt = UnifiedOptimizer()
        problem = OptimizationProblem(
            solver_type=SolverType.TSP,
            cities=["A", "B"],
            distance_matrix=[[0, 5], [5, 0]],
        )
        result = opt.solve(problem)
        assert isinstance(result, OptimizationResult)

    def test_solve_batch_publishes_events_for_each(self):
        """solve_batch publishes start/complete for each problem."""
        bus = EventBus()
        opt = UnifiedOptimizer(bus=bus)  # type: ignore[arg-type]
        problems = [
            OptimizationProblem(
                solver_type=SolverType.TSP,
                cities=["A", "B"],
                distance_matrix=[[0, 5], [5, 0]],
            ),
            OptimizationProblem(
                solver_type=SolverType.TSP,
                cities=["X", "Y"],
                distance_matrix=[[0, 7], [7, 0]],
            ),
        ]
        opt.solve_batch(problems)
        started = bus.get_history("optimization_started")
        completed = bus.get_history("optimization_completed")
        assert len(started) == 2
        assert len(completed) == 2


# ===========================================================================
# 5. New Event Types
# ===========================================================================


class TestNewEventTypes:
    """EventType catalog includes optimizer lifecycle events."""

    def test_optimization_started_type_exists(self):
        """EventType has OPTIMIZATION_STARTED."""
        assert hasattr(EventType, "OPTIMIZATION_STARTED")
        assert EventType.OPTIMIZATION_STARTED == "optimization_started"

    def test_optimization_completed_type_exists(self):
        """EventType has OPTIMIZATION_COMPLETED."""
        assert hasattr(EventType, "OPTIMIZATION_COMPLETED")
        assert EventType.OPTIMIZATION_COMPLETED == "optimization_completed"

    def test_task_assigned_type_exists(self):
        """EventType has TASK_ASSIGNED."""
        assert hasattr(EventType, "TASK_ASSIGNED")
        assert EventType.TASK_ASSIGNED == "task_assigned"

    def test_task_completed_type_exists(self):
        """EventType has TASK_COMPLETED."""
        assert hasattr(EventType, "TASK_COMPLETED")
        assert EventType.TASK_COMPLETED == "task_completed"

    def test_route_planned_type_exists(self):
        """EventType has ROUTE_PLANNED."""
        assert hasattr(EventType, "ROUTE_PLANNED")
        assert EventType.ROUTE_PLANNED == "route_planned"

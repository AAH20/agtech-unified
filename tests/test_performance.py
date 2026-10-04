"""Performance benchmark tests for AgTech Unified pipeline components.

Tests that each module completes within acceptable time bounds
and that the full pipeline meets latency requirements.
"""

import math
import time

from src.decision_support.recommender import DecisionEngine, FarmState
from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.iot.sensor_placement import PlacementInstance, SensorPlacement
from src.multi_agent.task_allocation import AllocationInstance, TaskAllocator
from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.vrp import VRPInstance, VRPSolver
from src.path_planning.coverage import CoverageInstance, CoveragePlanner


class TestSensorPlacementPerformance:
    """Performance benchmarks for sensor placement optimization."""

    def test_placement_small_instance_speed(self):
        """Small placement instance (10 sensors, 20 targets) completes quickly."""
        placement = SensorPlacement()
        instance = PlacementInstance(
            sensor_positions=[(i * 2.0, j * 2.0) for i in range(5) for j in range(2)],
            target_positions=[(i * 1.5, j * 1.5) for i in range(5) for j in range(4)],
            coverage_radius=5.0,
        )
        start = time.perf_counter()
        result = placement.optimize(instance)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0  # Should complete in under 1 second
        assert result.coverage_ratio > 0

    def test_placement_medium_instance_speed(self):
        """Medium placement instance (50 sensors, 100 targets) completes quickly."""
        placement = SensorPlacement()
        sensors = [(i * 3.0, j * 3.0) for i in range(10) for j in range(5)]
        targets = [(i * 2.0, j * 2.0) for i in range(10) for j in range(10)]
        instance = PlacementInstance(
            sensor_positions=sensors,
            target_positions=targets,
            coverage_radius=8.0,
        )
        start = time.perf_counter()
        result = placement.optimize(instance)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert result.coverage_ratio > 0


class TestDigitalTwinPerformance:
    """Performance benchmarks for digital twin simulation."""

    def test_simulation_30_days_speed(self):
        """30-day simulation completes quickly."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        start = time.perf_counter()
        result = twin.simulate(state, days=30)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
        assert result.days_simulated == 30

    def test_simulation_365_days_speed(self):
        """Full-year (365-day) simulation completes quickly."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.6,
            temperature=22.0,
            crop_height=0.05,
            nutrient_level=0.6,
        )
        start = time.perf_counter()
        result = twin.simulate(state, days=365)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0
        assert result.days_simulated == 365
        assert len(result.history) == 365


class TestDecisionEnginePerformance:
    """Performance benchmarks for decision engine."""

    def test_decision_single_state_speed(self):
        """Single decision completes in milliseconds."""
        engine = DecisionEngine()
        state = FarmState(
            soil_moisture=0.3,
            temperature=28.0,
            crop_height=0.4,
            nutrient_level=0.5,
            pest_pressure=0.2,
        )
        start = time.perf_counter()
        result = engine.recommend(state)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1  # Should be near-instant
        assert result.priority_score >= 0.0

    def test_decision_batch_100_states_speed(self):
        """100 decisions complete quickly."""
        engine = DecisionEngine()
        states = [
            FarmState(
                soil_moisture=0.1 + i * 0.005,
                temperature=20.0 + i * 0.1,
                crop_height=0.1 + i * 0.005,
                nutrient_level=0.3 + i * 0.003,
                pest_pressure=i * 0.005,
            )
            for i in range(100)
        ]
        start = time.perf_counter()
        results = [engine.recommend(s) for s in states]
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
        assert len(results) == 100


class TestFullPipelinePerformance:
    """End-to-end pipeline performance benchmarks."""

    def test_full_pipeline_latency(self):
        """Complete sensor→decision pipeline completes within latency budget."""
        start = time.perf_counter()

        # Sensor data ingestion
        farm_state = FarmState(
            soil_moisture=0.15,
            temperature=35.0,
            crop_height=0.3,
            nutrient_level=0.4,
            pest_pressure=0.2,
        )

        # Decision engine
        engine = DecisionEngine()
        decision = engine.recommend(farm_state)

        elapsed = time.perf_counter() - start
        assert elapsed < 0.5  # Full pipeline under 500ms
        assert len(decision.actions) > 0

    def test_simulation_to_decision_latency(self):
        """Digital twin + decision engine completes within latency budget."""
        start = time.perf_counter()

        # Simulate 14 days
        twin = DigitalTwin()
        sim = twin.simulate(
            SimulationState(
                soil_moisture=0.2,
                temperature=30.0,
                crop_height=0.1,
                nutrient_level=0.3,
            ),
            days=14,
        )

        # Decision on final state
        engine = DecisionEngine()
        decision = engine.recommend(
            FarmState(
                soil_moisture=sim.final_state.soil_moisture,
                temperature=30.0,
                crop_height=sim.final_state.crop_height,
                nutrient_level=sim.final_state.nutrient_level,
                pest_pressure=0.1,
            )
        )

        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
        assert decision.priority_score >= 0.0

    def test_tsp_solver_performance(self):
        """TSP solver completes within time budget for medium instance."""
        solver = TSPSolver(algorithm="christofides")
        n = 20
        coords = [(i * 1.0, (i % 5) * 2.0) for i in range(n)]
        dist = [
            [
                math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
                for j in range(n)
            ]
            for i in range(n)
        ]
        instance = TSPInstance(cities=[f"C{i}" for i in range(n)], distance_matrix=dist)

        start = time.perf_counter()
        result = solver.solve(instance)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
        assert len(result.tour) == n
        assert result.cost > 0

    def test_vrp_solver_performance(self):
        """VRP solver completes within time budget for medium instance."""
        solver = VRPSolver()
        n = 15
        customers = [(i * 2.0, (i % 3) * 3.0) for i in range(n)]
        demands = [10.0 + i for i in range(n)]
        dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    dist[i][j] = 0.0
                elif i == 0:
                    dist[i][j] = math.sqrt(customers[j - 1][0] ** 2 + customers[j - 1][1] ** 2)
                elif j == 0:
                    dist[i][j] = math.sqrt(customers[i - 1][0] ** 2 + customers[i - 1][1] ** 2)
                else:
                    dist[i][j] = math.sqrt(
                        (customers[i - 1][0] - customers[j - 1][0]) ** 2
                        + (customers[i - 1][1] - customers[j - 1][1]) ** 2
                    )
        instance = VRPInstance(
            depot=(0, 0),
            customers=customers,
            demands=demands,
            vehicle_capacity=50.0,
            distance_matrix=dist,
        )

        start = time.perf_counter()
        result = solver.solve(instance)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0
        assert len(result.routes) > 0
        assert result.total_cost > 0

    def test_task_allocation_performance(self):
        """Task allocation completes within time budget."""
        allocator = TaskAllocator(algorithm="hungarian")
        n = 10
        agents = [f"agent_{i}" for i in range(n)]
        tasks = [f"task_{i}" for i in range(n)]
        cost_matrix = [[float((i + j) % 7 + 1) for j in range(n)] for i in range(n)]
        instance = AllocationInstance(agents=agents, tasks=tasks, cost_matrix=cost_matrix)

        start = time.perf_counter()
        result = allocator.allocate(instance)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
        assert result.total_cost > 0

    def test_coverage_planning_performance(self):
        """Coverage planning completes within time budget."""
        planner = CoveragePlanner()
        instance = CoverageInstance(
            field_boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
            swath_width=2.0,
            start_point=(0, 0),
        )

        start = time.perf_counter()
        result = planner.plan(instance)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
        assert result.coverage_ratio > 0.8
        assert result.total_distance > 0

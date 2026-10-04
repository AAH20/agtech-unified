"""Fault injection and recovery tests for AgTech Unified.

Tests system resilience by injecting faults (invalid inputs,
edge cases, boundary conditions) and verifying graceful
recovery or appropriate error handling.
"""

import pytest

from src.decision_support.recommender import DecisionEngine, FarmState
from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.genomics.crispr import CRISPRDesigner
from src.genomics.protein import ProteinAnalyzer
from src.iot.sensor_placement import PlacementInstance, SensorPlacement
from src.multi_agent.task_allocation import AllocationInstance, TaskAllocator
from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.vrp import VRPInstance, VRPSolver
from src.path_planning.coverage import CoverageInstance, CoveragePlanner


class TestSensorPlacementFaultTolerance:
    """Fault injection tests for sensor placement."""

    def test_placement_rejects_zero_radius(self):
        """Zero coverage radius is rejected with clear error."""
        with pytest.raises(ValueError, match="Coverage radius must be positive"):
            PlacementInstance(
                sensor_positions=[(0.0, 0.0)],
                target_positions=[(1.0, 1.0)],
                coverage_radius=0.0,
            )

    def test_placement_rejects_negative_radius(self):
        """Negative coverage radius is rejected."""
        with pytest.raises(ValueError, match="Coverage radius must be positive"):
            PlacementInstance(
                sensor_positions=[(0.0, 0.0)],
                target_positions=[(1.0, 1.0)],
                coverage_radius=-5.0,
            )

    def test_placement_handles_empty_inputs(self):
        """Empty sensor/target lists return empty result, not crash."""
        placement = SensorPlacement()
        result = placement.optimize(
            PlacementInstance(
                sensor_positions=[],
                target_positions=[],
                coverage_radius=5.0,
            )
        )
        assert result.selected_sensors == []
        assert result.coverage_ratio == 0.0

    def test_placement_handles_uncoverable_targets(self):
        """Targets beyond all sensor ranges return partial coverage."""
        placement = SensorPlacement()
        result = placement.optimize(
            PlacementInstance(
                sensor_positions=[(0.0, 0.0)],
                target_positions=[(0.0, 0.0), (1000.0, 1000.0)],
                coverage_radius=1.0,
            )
        )
        assert result.coverage_ratio == 0.5
        assert 0 in result.covered_targets
        assert 1 not in result.covered_targets

    def test_placement_rejects_unknown_algorithm(self):
        """Unknown algorithm name raises ValueError."""
        placement = SensorPlacement(algorithm="quantum")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            placement.optimize(
                PlacementInstance(
                    sensor_positions=[(0.0, 0.0)],
                    target_positions=[(1.0, 0.0)],
                    coverage_radius=5.0,
                )
            )


class TestDigitalTwinFaultTolerance:
    """Fault injection tests for digital twin."""

    def test_simulation_rejects_negative_days(self):
        """Negative simulation days raises ValueError."""
        twin = DigitalTwin()
        with pytest.raises(ValueError, match="Days must be non-negative"):
            twin.simulate(
                SimulationState(
                    soil_moisture=0.5,
                    temperature=25.0,
                    crop_height=0.1,
                    nutrient_level=0.5,
                ),
                days=-5,
            )

    def test_simulation_handles_zero_days(self):
        """Zero days returns initial state unchanged."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )
        result = twin.simulate(state, days=0)
        assert result.days_simulated == 0
        assert result.final_state.crop_height == 0.1
        assert result.total_growth == 0.0

    def test_simulation_handles_extreme_temperature(self):
        """Extreme temperature (beyond 45°C) produces zero growth."""
        twin = DigitalTwin()
        result = twin.simulate(
            SimulationState(
                soil_moisture=0.8,
                temperature=50.0,
                crop_height=0.5,
                nutrient_level=0.8,
            ),
            days=10,
        )
        # Temp factor is 0 beyond 45°C, so no growth
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_simulation_handles_zero_moisture(self):
        """Zero soil moisture produces zero growth (water stress)."""
        twin = DigitalTwin()
        result = twin.simulate(
            SimulationState(
                soil_moisture=0.0,
                temperature=25.0,
                crop_height=0.5,
                nutrient_level=0.8,
            ),
            days=10,
        )
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)

    def test_simulation_recovers_from_near_zero_moisture(self):
        """Near-zero moisture (0.05) still allows minimal growth."""
        twin = DigitalTwin()
        result = twin.simulate(
            SimulationState(
                soil_moisture=0.05,
                temperature=25.0,
                crop_height=0.5,
                nutrient_level=0.8,
            ),
            days=5,
        )
        # Water factor is 0 below 0.1, so no growth
        assert result.total_growth == pytest.approx(0.0, abs=1e-6)


class TestDecisionEngineFaultTolerance:
    """Fault injection tests for decision engine."""

    def test_decision_handles_all_zeros(self):
        """All-zero farm state triggers critical alerts without crash."""
        engine = DecisionEngine()
        result = engine.recommend(
            FarmState(
                soil_moisture=0.0,
                temperature=0.0,
                crop_height=0.0,
                nutrient_level=0.0,
                pest_pressure=0.0,
            )
        )
        assert result.priority_score > 0.5
        assert len(result.recommendations) >= 2

    def test_decision_handles_extreme_values(self):
        """Extreme but valid values produce appropriate recommendations."""
        engine = DecisionEngine()
        result = engine.recommend(
            FarmState(
                soil_moisture=0.01,
                temperature=44.0,
                crop_height=0.01,
                nutrient_level=0.01,
                pest_pressure=0.99,
            )
        )
        assert result.priority_score == 1.0  # Capped at 1.0
        assert len(result.recommendations) >= 3

    def test_decision_handles_healthy_state(self):
        """Healthy farm state produces no urgent recommendations."""
        engine = DecisionEngine()
        result = engine.recommend(
            FarmState(
                soil_moisture=0.7,
                temperature=22.0,
                crop_height=0.8,
                nutrient_level=0.8,
                pest_pressure=0.05,
            )
        )
        assert result.priority_score < 0.2
        assert len(result.recommendations) == 0


class TestTaskAllocationFaultTolerance:
    """Fault injection tests for task allocation."""

    def test_allocation_rejects_non_square_matrix(self):
        """Non-square cost matrix raises ValueError."""
        with pytest.raises(ValueError, match="Cost matrix must be square"):
            AllocationInstance(
                agents=["A1", "A2"],
                tasks=["T1", "T2"],
                cost_matrix=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )

    def test_allocation_handles_empty_tasks(self):
        """Empty task list returns empty allocation."""
        allocator = TaskAllocator()
        result = allocator.allocate(
            AllocationInstance(
                agents=[],
                tasks=[],
                cost_matrix=[],
            )
        )
        assert result.total_cost == 0.0
        assert result.allocation == {}

    def test_allocation_rejects_unknown_algorithm(self):
        """Unknown algorithm raises ValueError."""
        allocator = TaskAllocator(algorithm="genetic")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            allocator.allocate(
                AllocationInstance(
                    agents=["A1"],
                    tasks=["T1"],
                    cost_matrix=[[1.0]],
                )
            )


class TestVRPFaultTolerance:
    """Fault injection tests for VRP solver."""

    def test_vrp_rejects_zero_capacity(self):
        """Zero vehicle capacity raises ValueError."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            VRPInstance(
                depot=(0.0, 0.0),
                customers=[(1.0, 0.0)],
                demands=[10.0],
                vehicle_capacity=0.0,
                distance_matrix=[[0.0, 1.0], [1.0, 0.0]],
            )

    def test_vrp_rejects_demand_exceeding_capacity(self):
        """Demand exceeding vehicle capacity raises ValueError."""
        with pytest.raises(ValueError, match="Demand exceeds vehicle capacity"):
            VRPInstance(
                depot=(0.0, 0.0),
                customers=[(1.0, 0.0)],
                demands=[200.0],
                vehicle_capacity=100.0,
                distance_matrix=[[0.0, 1.0], [1.0, 0.0]],
            )

    def test_vrp_handles_empty_customers(self):
        """Empty customer list returns empty routes."""
        solver = VRPSolver()
        result = solver.solve(
            VRPInstance(
                depot=(0.0, 0.0),
                customers=[],
                demands=[],
                vehicle_capacity=100.0,
                distance_matrix=[],
            )
        )
        assert result.routes == []
        assert result.total_cost == 0.0


class TestTSPFaultTolerance:
    """Fault injection tests for TSP solver."""

    def test_tsp_rejects_non_square_matrix(self):
        """Non-square distance matrix raises ValueError."""
        with pytest.raises(ValueError, match="Distance matrix must be square"):
            TSPInstance(
                cities=["A", "B"],
                distance_matrix=[[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]],
            )

    def test_tsp_handles_empty_cities(self):
        """Empty city list returns empty tour."""
        solver = TSPSolver()
        result = solver.solve(TSPInstance(cities=[], distance_matrix=[]))
        assert result.tour == []
        assert result.cost == 0.0

    def test_tsp_rejects_unknown_algorithm(self):
        """Unknown algorithm raises ValueError."""
        solver = TSPSolver(algorithm="brute_force")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            solver.solve(
                TSPInstance(
                    cities=["A", "B"],
                    distance_matrix=[[0.0, 1.0], [1.0, 0.0]],
                )
            )


class TestCoverageFaultTolerance:
    """Fault injection tests for coverage planner."""

    def test_coverage_rejects_small_boundary(self):
        """Boundary with < 3 points raises ValueError."""
        with pytest.raises(ValueError, match="Field boundary must have at least 3 points"):
            CoverageInstance(
                field_boundary=[(0.0, 0.0), (1.0, 1.0)],
                swath_width=2.0,
                start_point=(0.0, 0.0),
            )

    def test_coverage_rejects_zero_swath(self):
        """Zero swath width raises ValueError."""
        with pytest.raises(ValueError, match="Swath width must be positive"):
            CoverageInstance(
                field_boundary=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                swath_width=0.0,
                start_point=(0.0, 0.0),
            )

    def test_coverage_rejects_unknown_algorithm(self):
        """Unknown algorithm raises ValueError."""
        planner = CoveragePlanner(algorithm="spiral")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            planner.plan(
                CoverageInstance(
                    field_boundary=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
                    swath_width=2.0,
                    start_point=(0.0, 0.0),
                )
            )


class TestGenomicsFaultTolerance:
    """Fault injection tests for genomics modules."""

    def test_protein_rejects_empty_sequence(self):
        """Empty protein sequence raises ValueError."""
        analyzer = ProteinAnalyzer()
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            analyzer.analyze("")

    def test_protein_rejects_invalid_amino_acid(self):
        """Invalid amino acid character raises ValueError."""
        analyzer = ProteinAnalyzer()
        with pytest.raises(ValueError, match="Invalid amino acid"):
            analyzer.analyze("ACDEFGHIKLMNPQRSTVWYZ")

    def test_crispr_rejects_empty_sequence(self):
        """Empty DNA sequence raises ValueError."""
        designer = CRISPRDesigner()
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            designer.design_guide("")

    def test_crispr_rejects_no_pam_site(self):
        """Sequence without PAM site raises ValueError."""
        designer = CRISPRDesigner()
        with pytest.raises(ValueError, match="No PAM site found"):
            designer.design_guide("ATATATATATATATATATAT")


class TestSystemRecovery:
    """Tests for system recovery after faults."""

    def test_pipeline_recovers_after_invalid_sensor_data(self):
        """Pipeline recovers when sensor data is corrected after invalid input."""
        # First: invalid sensor data is rejected
        with pytest.raises(ValueError):
            PlacementInstance(
                sensor_positions=[(0.0, 0.0)],
                target_positions=[(1.0, 1.0)],
                coverage_radius=-1.0,
            )

        # Then: corrected data works fine
        placement = SensorPlacement()
        result = placement.optimize(
            PlacementInstance(
                sensor_positions=[(0.0, 0.0)],
                target_positions=[(1.0, 1.0)],
                coverage_radius=5.0,
            )
        )
        assert result.coverage_ratio == 1.0

    def test_pipeline_recovers_after_simulation_error(self):
        """System recovers after simulation error with valid retry."""
        twin = DigitalTwin()
        state = SimulationState(
            soil_moisture=0.5,
            temperature=25.0,
            crop_height=0.1,
            nutrient_level=0.5,
        )

        # First: invalid days rejected
        with pytest.raises(ValueError):
            twin.simulate(state, days=-1)

        # Then: valid simulation works
        result = twin.simulate(state, days=5)
        assert result.days_simulated == 5
        assert result.final_state.crop_height > 0.1

    def test_decision_engine_recovers_after_extreme_state(self):
        """Decision engine handles extreme state then processes normal state."""
        engine = DecisionEngine()

        # First: extreme state
        extreme = engine.recommend(
            FarmState(
                soil_moisture=0.0,
                temperature=45.0,
                crop_height=0.0,
                nutrient_level=0.0,
                pest_pressure=1.0,
            )
        )
        assert extreme.priority_score == 1.0

        # Then: normal state
        normal = engine.recommend(
            FarmState(
                soil_moisture=0.6,
                temperature=25.0,
                crop_height=0.5,
                nutrient_level=0.6,
                pest_pressure=0.1,
            )
        )
        assert normal.priority_score < 0.3

    def test_vrp_recovers_after_capacity_error(self):
        """VRP recovers after capacity error with valid input."""
        # First: invalid capacity rejected
        with pytest.raises(ValueError):
            VRPInstance(
                depot=(0.0, 0.0),
                customers=[(1.0, 0.0)],
                demands=[200.0],
                vehicle_capacity=100.0,
                distance_matrix=[[0.0, 1.0], [1.0, 0.0]],
            )

        # Then: valid instance works
        solver = VRPSolver()
        result = solver.solve(
            VRPInstance(
                depot=(0.0, 0.0),
                customers=[(1.0, 0.0)],
                demands=[50.0],
                vehicle_capacity=100.0,
                distance_matrix=[[0.0, 1.0], [1.0, 0.0]],
            )
        )
        assert len(result.routes) == 1
        assert result.total_cost > 0

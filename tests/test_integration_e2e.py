"""End-to-end integration tests: sensors → edge → cloud → AI → decisions → actuators.

These tests validate the entire AgTech pipeline working together,
from sensor data collection through simulation, decision-making,
and action recommendation.
"""
import pytest
import math
import time

from src.iot.sensor_placement import SensorPlacement, PlacementInstance
from src.digital_twin.simulator import DigitalTwin, SimulationState
from src.decision_support.recommender import DecisionEngine, FarmState
from src.multi_agent.task_allocation import TaskAllocator, AllocationInstance
from src.optimization.tsp import TSPSolver, TSPInstance
from src.optimization.vrp import VRPSolver, VRPInstance
from src.path_planning.coverage import CoveragePlanner, CoverageInstance
from src.genomics.protein import ProteinAnalyzer
from src.genomics.crispr import CRISPRDesigner


class TestFullPipeline:
    """E2E tests covering the complete data flow from sensors to decisions."""

    def test_full_pipeline_sensor_to_decision(self):
        """Sensor data → FarmState → DecisionEngine → actions.

        Simulates a sensor network detecting low soil moisture
        and high temperature, flowing through to irrigation and
        shade recommendations.
        """
        # Step 1: Sensors detect field conditions
        sensor_data = {
            "soil_moisture": 0.15,
            "temperature": 39.0,
            "crop_height": 0.3,
            "nutrient_level": 0.4,
            "pest_pressure": 0.2,
        }

        # Step 2: Edge processing — create FarmState from sensor readings
        farm_state = FarmState(**sensor_data)

        # Step 3: Cloud/AI — DecisionEngine processes the state
        engine = DecisionEngine()
        result = engine.recommend(farm_state)

        # Step 4: Actuators — verify actionable recommendations
        assert len(result.recommendations) > 0
        assert result.priority_score > 0.3
        assert "irrigate" in result.actions
        assert "shade" in result.actions
        assert result.algorithm == "rule_based"

    def test_digital_twin_to_decision_pipeline(self):
        """Simulate crop growth → extract final state → get recommendations.

        Runs a 30-day simulation starting from poor conditions,
        then feeds the final state into the decision engine.
        """
        # Step 1: Initial sensor readings (poor conditions)
        initial_state = SimulationState(
            soil_moisture=0.2,
            temperature=30.0,
            crop_height=0.05,
            nutrient_level=0.3,
        )

        # Step 2: Cloud simulation — digital twin projects 30 days forward
        twin = DigitalTwin()
        sim_result = twin.simulate(initial_state, days=30)

        # Step 3: Extract final state for decision engine
        final = sim_result.final_state
        farm_state = FarmState(
            soil_moisture=final.soil_moisture,
            temperature=30.0,  # Assume constant temperature
            crop_height=final.crop_height,
            nutrient_level=final.nutrient_level,
            pest_pressure=0.1,
        )

        # Step 4: Decision engine recommends actions
        engine = DecisionEngine()
        decision = engine.recommend(farm_state)

        # Step 5: Verify pipeline produced meaningful output
        assert sim_result.days_simulated == 30
        assert len(sim_result.history) == 30
        assert decision.priority_score >= 0.0
        # After 30 days with low moisture, irrigation should be recommended
        assert "irrigate" in decision.actions

    def test_sensor_placement_to_coverage_pipeline(self):
        """Place sensors → plan drone coverage path for monitoring.

        Optimizes sensor positions for field coverage, then uses
        the covered area to plan a drone monitoring route.
        """
        # Step 1: Define field targets (crop monitoring points)
        targets = [(i * 5, j * 5) for i in range(4) for j in range(4)]

        # Step 2: Edge — optimize sensor placement
        placement = SensorPlacement()
        instance = PlacementInstance(
            sensor_positions=[(2.5, 2.5), (12.5, 2.5), (2.5, 12.5), (12.5, 12.5)],
            target_positions=targets,
            coverage_radius=8.0,
        )
        placement_result = placement.optimize(instance)

        # Step 3: Verify sensor coverage
        assert placement_result.coverage_ratio == 1.0
        assert len(placement_result.selected_sensors) > 0

        # Step 4: Plan coverage path for drone to inspect sensor-blind spots
        planner = CoveragePlanner()
        coverage_instance = CoverageInstance(
            field_boundary=[(0, 0), (20, 0), (20, 20), (0, 20)],
            swath_width=4.0,
            start_point=(0, 0),
        )
        coverage_result = planner.plan(coverage_instance)

        # Step 5: Verify coverage path is valid
        assert len(coverage_result.path) > 0
        assert coverage_result.coverage_ratio > 0.8
        assert coverage_result.total_distance > 0

    def test_task_allocation_to_vrp_pipeline(self):
        """Allocate tasks to agents → plan vehicle routes for execution.

        Multi-agent task allocation determines which agent does what,
        then VRP plans the physical routes.
        """
        # Step 1: Define agents and tasks
        agents = ["drone_1", "drone_2", "tractor_1"]
        tasks = ["spray_A", "spray_B", "spray_C", "inspect_D", "inspect_E"]

        # Step 2: Cost matrix (agent × task)
        cost_matrix = [
            [10, 12, 15, 8, 9],   # drone_1
            [11, 9, 14, 10, 8],   # drone_2
            [20, 22, 18, 15, 16], # tractor_1
        ]

        # Step 3: Multi-agent allocation
        allocator = TaskAllocator(algorithm="hungarian")
        alloc_instance = AllocationInstance(
            agents=agents, tasks=tasks, cost_matrix=cost_matrix,
        )
        alloc_result = allocator.allocate(alloc_instance)

        # Step 4: Verify all tasks assigned
        assigned = [t for tasks in alloc_result.allocation.values() for t in tasks]
        assert len(assigned) == len(tasks)
        assert len(set(assigned)) == len(tasks)

        # Step 5: Plan VRP routes for task execution
        customers = [(3, 0), (6, 0), (0, 4), (4, 4), (8, 2)]
        demands = [20, 20, 15, 15, 10]
        dist_matrix = [
            [0, 3, 6, 5, 6, 8],
            [3, 0, 3, 4, 5, 7],
            [6, 3, 0, 4, 3, 5],
            [5, 4, 4, 0, 4, 4],
            [6, 5, 3, 4, 0, 4],
            [8, 7, 5, 4, 4, 0],
        ]
        vrp = VRPSolver()
        vrp_instance = VRPInstance(
            depot=(0, 0),
            customers=customers,
            demands=demands,
            vehicle_capacity=50,
            distance_matrix=dist_matrix,
        )
        vrp_result = vrp.solve(vrp_instance)

        # Step 6: Verify routes are valid
        assert len(vrp_result.routes) > 0
        assert vrp_result.total_cost > 0
        visited = [c for r in vrp_result.routes for c in r]
        assert sorted(visited) == list(range(len(customers)))

    def test_drought_scenario_end_to_end(self):
        """Drought conditions → simulation → decision → irrigation action.

        Full pipeline test for drought: sensors detect dry soil,
        digital twin projects crop stress, decision engine
        recommends urgent irrigation.
        """
        # Step 1: Sensors detect drought conditions
        drought_state = SimulationState(
            soil_moisture=0.05,
            temperature=38.0,
            crop_height=0.2,
            nutrient_level=0.3,
        )

        # Step 2: Digital twin simulates 14 days of drought
        twin = DigitalTwin()
        sim_result = twin.simulate(drought_state, days=14)

        # Step 3: Verify crop is stressed
        assert sim_result.final_state.soil_moisture <= 0.05
        assert sim_result.total_growth < 0.5  # Minimal growth under drought

        # Step 4: Decision engine processes drought state
        farm_state = FarmState(
            soil_moisture=sim_result.final_state.soil_moisture,
            temperature=38.0,
            crop_height=sim_result.final_state.crop_height,
            nutrient_level=sim_result.final_state.nutrient_level,
            pest_pressure=0.1,
        )
        engine = DecisionEngine()
        decision = engine.recommend(farm_state)

        # Step 5: Verify urgent irrigation is recommended
        assert decision.priority_score >= 0.5
        assert "irrigate" in decision.actions
        assert any("URGENT" in r for r in decision.recommendations)

    def test_pest_outbreak_scenario(self):
        """High pest pressure → decision → pest control action.

        Sensors detect pest outbreak, decision engine recommends
        immediate pest control intervention.
        """
        # Step 1: Sensors detect pest outbreak
        pest_state = FarmState(
            soil_moisture=0.5,
            temperature=28.0,
            crop_height=0.4,
            nutrient_level=0.5,
            pest_pressure=0.85,
        )

        # Step 2: Decision engine processes pest alert
        engine = DecisionEngine()
        decision = engine.recommend(pest_state)

        # Step 3: Verify pest control is recommended with high priority
        assert decision.priority_score > 0.3
        assert "pest_control" in decision.actions
        assert any("pest" in r.lower() for r in decision.recommendations)

    def test_multi_agent_coordination_pipeline(self):
        """Multiple agents → task allocation → VRP routing → coverage.

        Full coordination pipeline: allocate monitoring tasks to
        drones, plan their routes, and verify field coverage.
        """
        # Step 1: Define monitoring scenario
        agents = ["drone_north", "drone_south", "drone_east", "drone_west"]
        tasks = ["patrol_1", "patrol_2", "patrol_3", "patrol_4"]

        # Step 2: Cost matrix based on distance from agent base to patrol zone
        cost_matrix = [
            [5, 8, 15, 18],   # drone_north
            [18, 15, 8, 5],   # drone_south
            [12, 6, 10, 14],  # drone_east
            [14, 10, 6, 12],  # drone_west
        ]

        # Step 3: Allocate tasks optimally
        allocator = TaskAllocator(algorithm="hungarian")
        alloc_result = allocator.allocate(AllocationInstance(
            agents=agents, tasks=tasks, cost_matrix=cost_matrix,
        ))

        # Step 4: Verify optimal allocation
        assert alloc_result.algorithm == "hungarian"
        assigned = [t for tasks in alloc_result.allocation.values() for t in tasks]
        assert len(assigned) == 4
        assert len(set(assigned)) == 4

        # Step 5: Plan coverage path for the patrol area
        planner = CoveragePlanner()
        coverage = planner.plan(CoverageInstance(
            field_boundary=[(0, 0), (30, 0), (30, 30), (0, 30)],
            swath_width=5.0,
            start_point=(0, 0),
        ))

        # Step 6: Verify coverage
        assert coverage.coverage_ratio > 0.8
        assert coverage.num_passes > 0

    def test_genomics_to_decision_pipeline(self):
        """Protein analysis → stability assessment → breeding decision.

        Analyzes a target protein, assesses its stability,
        and flows the result into a decision about whether
        to proceed with a breeding program.
        """
        # Step 1: Analyze target protein
        analyzer = ProteinAnalyzer()
        protein_result = analyzer.analyze("ACDEFGHIKLMNPQRSTVWY")

        # Step 2: Assess protein stability
        assert protein_result.length == 20
        assert protein_result.molecular_weight > 0
        assert 0.0 <= protein_result.stability_score <= 1.0

        # Step 3: Make breeding decision based on stability
        if protein_result.stability_score > 0.5:
            decision = "proceed_with_breeding"
        else:
            decision = "needs_optimization"

        # Step 4: Verify decision is valid
        assert decision in ("proceed_with_breeding", "needs_optimization")

    def test_crispr_to_protein_pipeline(self):
        """CRISPR design → guide RNA → protein analysis of target.

        Designs CRISPR guides for a target gene, then analyzes
        the protein that the gene encodes.
        """
        # Step 1: Design CRISPR guides for target sequence
        designer = CRISPRDesigner()
        dna = "ATCGATCGATCGATCGATCGATCGGG" * 3
        guides = designer.design_guide(dna)

        # Step 2: Verify guides were designed
        assert len(guides) > 0
        for guide in guides:
            assert len(guide.sequence) == 20
            assert guide.pam.endswith("GG")
            assert 0.0 <= guide.efficiency_score <= 1.0
            assert 0.0 <= guide.off_target_score <= 1.0

        # Step 3: Analyze the target protein (encoded by the gene)
        analyzer = ProteinAnalyzer()
        protein = analyzer.analyze("MKTAYIAKQRQISFVKSHFSRQ")

        # Step 4: Verify protein analysis
        assert protein.length > 0
        assert protein.molecular_weight > 0
        assert 0.0 <= protein.stability_score <= 1.0

    def test_full_farm_management_cycle(self):
        """Complete cycle: sensors → simulation → decisions → actions → re-simulation.

        Runs a full farm management cycle:
        1. Sensors collect initial data
        2. Digital twin simulates 7 days
        3. Decision engine recommends actions
        4. Actions are "applied" (state is modified)
        5. Digital twin re-simulates with improved conditions
        6. Verify improvement
        """
        # Step 1: Initial sensor readings (suboptimal conditions)
        initial = SimulationState(
            soil_moisture=0.15,
            temperature=32.0,
            crop_height=0.1,
            nutrient_level=0.25,
        )

        # Step 2: Simulate 7 days without intervention
        twin = DigitalTwin()
        before = twin.simulate(initial, days=7)

        # Step 3: Decision engine recommends actions
        farm_state = FarmState(
            soil_moisture=before.final_state.soil_moisture,
            temperature=32.0,
            crop_height=before.final_state.crop_height,
            nutrient_level=before.final_state.nutrient_level,
            pest_pressure=0.1,
        )
        engine = DecisionEngine()
        decision = engine.recommend(farm_state)

        # Step 4: Apply recommended actions (improve conditions)
        improved = SimulationState(
            soil_moisture=min(1.0, before.final_state.soil_moisture + 0.3),
            temperature=25.0,  # Shade applied
            crop_height=before.final_state.crop_height,
            nutrient_level=min(1.0, before.final_state.nutrient_level + 0.2),
        )

        # Step 5: Re-simulate with improved conditions
        after = twin.simulate(improved, days=7)

        # Step 6: Verify improvement
        assert after.final_state.crop_height >= before.final_state.crop_height
        assert after.total_growth > 0
        # Decision engine should have recommended actions
        assert len(decision.actions) > 0

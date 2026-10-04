"""End-to-end farm example: integrate all modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.vrp import VRPInstance, VRPSolver
from src.iot.sensor_placement import PlacementInstance, SensorPlacement
from src.digital_twin.simulator import SimulationState, DigitalTwin
from src.decision_support.recommender import FarmState, DecisionEngine
from src.multi_agent.swarm import Agent, Task, SwarmCoordinator


def main():
    print("=== End-to-End Farm Simulation ===")

    # 1. Plan routes
    cities = ["Farm", "Field-A", "Field-B", "Field-C"]
    dist = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    tsp_result = TSPSolver("christofides").solve(TSPInstance(cities, dist))
    print(f"Route: {tsp_result.tour}")

    # 2. Place sensors
    sensors = [(0, 0), (5, 5), (10, 0)]
    targets = [(1, 1), (4, 4), (9, 1), (5, 0)]
    placement = SensorPlacement("greedy").optimize(PlacementInstance(sensors, targets, 3.0))
    print(f"Sensors: {placement.selected_sensors}")

    # 3. Simulate crop growth
    state = SimulationState(soil_moisture=0.6, temperature=25.0, crop_height=0.1, nutrient_level=0.8)
    twin_result = DigitalTwin("logistic_growth").simulate(state, days=30)
    print(f"Crop height: {twin_result.final_state.crop_height:.2f}m")

    # 4. Get recommendations
    farm_state = FarmState(soil_moisture=0.15, temperature=40.0, crop_height=0.05, nutrient_level=0.1, pest_pressure=0.8)
    recs = DecisionEngine("rule_based").recommend(farm_state)
    print(f"Recommendations: {len(recs.recommendations)}")

    # 5. Coordinate robots
    agents = [Agent(id="robot-1", position=(0, 0), capacity=10)]
    tasks = [Task(id="weed-1", position=(2, 3), priority=1)]
    coordinator = SwarmCoordinator(agents)
    assignments = coordinator.assign_tasks(tasks)
    print(f"Assignments: {assignments}")

    print("=== Simulation Complete ===")


if __name__ == "__main__":
    main()

"""Quickstart: run TSP, VRP, sensor placement, digital twin, and decision support in sequence."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optimization.tsp import TSPInstance, TSPSolver
from optimization.vrp import VRPInstance, VRPSolver
from iot.sensor_placement import PlacementInstance, SensorPlacement
from digital_twin.simulator import SimulationState, DigitalTwin
from decision_support.recommender import FarmState, DecisionEngine


def run_tsp():
    cities = ["Farm", "Field-A", "Field-B", "Field-C"]
    dist = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    result = TSPSolver("christofides").solve(TSPInstance(cities, dist))
    print(f"TSP: tour={result.tour}, cost={result.cost:.1f}")


def run_vrp():
    depot = (0.0, 0.0)
    customers = [(1.0, 2.0), (3.0, 1.0), (2.0, 4.0)]
    demands = [5.0, 8.0, 6.0]
    dist = [[0, 2.2, 3.2, 4.5], [2.2, 0, 2.8, 3.6], [3.2, 2.8, 0, 3.2], [4.5, 3.6, 3.2, 0]]
    result = VRPSolver("savings").solve(VRPInstance(depot, customers, demands, 15.0, dist))
    print(f"VRP: routes={result.routes}, cost={result.total_cost:.1f}, vehicles={result.num_vehicles}")


def run_sensor_placement():
    sensors = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    targets = [(1.0, 1.0), (4.0, 4.0), (9.0, 1.0), (5.0, 0.0)]
    result = SensorPlacement("greedy").optimize(PlacementInstance(sensors, targets, 3.0))
    print(f"Sensors: selected={result.selected_sensors}, coverage={result.coverage_ratio:.0%}")


def run_digital_twin():
    state = SimulationState(soil_moisture=0.6, temperature=25.0, crop_height=0.1, nutrient_level=0.8)
    result = DigitalTwin("logistic_growth").simulate(state, days=30)
    print(f"Twin: days={result.days_simulated}, final_height={result.final_state.crop_height:.2f}m")


def run_decision_support():
    state = FarmState(soil_moisture=0.15, temperature=40.0, crop_height=0.05, nutrient_level=0.1, pest_pressure=0.8)
    result = DecisionEngine("rule_based").recommend(state)
    print(f"Decisions: {len(result.recommendations)} recommendations, priority={result.priority_score:.1f}")


if __name__ == "__main__":
    run_tsp()
    run_vrp()
    run_sensor_placement()
    run_digital_twin()
    run_decision_support()
    print("All modules ran successfully.")

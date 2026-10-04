"""Test VRP solver with savings algorithm (Clarke-Wright)."""
import pytest
import math
from src.optimization.vrp import VRPSolver, VRPInstance


def test_vrp_empty():
    """VRP with no customers returns empty routes."""
    solver = VRPSolver()
    instance = VRPInstance(
        depot=(0, 0),
        customers=[],
        demands=[],
        vehicle_capacity=100,
        distance_matrix=[],
    )
    result = solver.solve(instance)
    assert result.routes == []
    assert result.total_cost == 0.0


def test_vrp_single_customer():
    """VRP with one customer returns single route."""
    solver = VRPSolver()
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(3, 4)],
        demands=[10],
        vehicle_capacity=100,
        distance_matrix=[[0, 5], [5, 0]],
    )
    result = solver.solve(instance)
    assert len(result.routes) == 1
    assert len(result.routes[0]) == 1
    assert result.total_cost == pytest.approx(10.0)  # 5 out + 5 back


def test_vrp_capacity_constraint():
    """VRP respects vehicle capacity — splits into multiple routes."""
    solver = VRPSolver()
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(1, 0), (2, 0), (3, 0)],
        demands=[60, 60, 60],
        vehicle_capacity=100,
        distance_matrix=[
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ],
    )
    result = solver.solve(instance)
    # Each customer needs own route (60+60 > 100)
    assert len(result.routes) == 3
    for route in result.routes:
        total_demand = sum(60 for _ in route)
        assert total_demand <= 100


def test_vrp_two_customers_one_route():
    """Two customers that fit in one vehicle share a route."""
    solver = VRPSolver()
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(3, 0), (6, 0)],
        demands=[30, 30],
        vehicle_capacity=100,
        distance_matrix=[
            [0, 3, 6],
            [3, 0, 3],
            [6, 3, 0],
        ],
    )
    result = solver.solve(instance)
    assert len(result.routes) == 1
    assert len(result.routes[0]) == 2


def test_vrp_savings_algorithm():
    """Clarke-Wright savings produces valid routes."""
    solver = VRPSolver(algorithm="savings")
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(1, 0), (0, 1), (-1, 0), (0, -1)],
        demands=[10, 10, 10, 10],
        vehicle_capacity=100,
        distance_matrix=[
            [0, 1, 1.41, 1.41, 1],
            [1, 0, 1.41, 2, 1.41],
            [1.41, 1.41, 0, 1.41, 2],
            [1.41, 2, 1.41, 0, 1.41],
            [1, 1.41, 2, 1.41, 0],
        ],
    )
    result = solver.solve(instance)
    assert len(result.routes) >= 1
    # All customers visited exactly once
    visited = [c for r in result.routes for c in r]
    assert sorted(visited) == [0, 1, 2, 3]


def test_vrp_route_cost_calculation():
    """Route cost includes depot→customer→depot."""
    solver = VRPSolver()
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(3, 4)],
        demands=[10],
        vehicle_capacity=100,
        distance_matrix=[[0, 5], [5, 0]],
    )
    result = solver.solve(instance)
    # Cost = depot→cust (5) + cust→depot (5) = 10
    assert result.total_cost == pytest.approx(10.0)


def test_vrp_invalid_capacity():
    """Zero or negative capacity raises ValueError."""
    solver = VRPSolver()
    with pytest.raises(ValueError, match="Capacity must be positive"):
        instance = VRPInstance(
            depot=(0, 0),
            customers=[(1, 0)],
            demands=[10],
            vehicle_capacity=0,
            distance_matrix=[[0, 1], [1, 0]],
        )


def test_vrp_demand_exceeds_capacity():
    """Customer demand exceeding capacity raises ValueError."""
    solver = VRPSolver()
    with pytest.raises(ValueError, match="Demand exceeds vehicle capacity"):
        instance = VRPInstance(
            depot=(0, 0),
            customers=[(1, 0)],
            demands=[200],
            vehicle_capacity=100,
            distance_matrix=[[0, 1], [1, 0]],
        )


def test_vrp_result_contains_algorithm():
    """VRPResult includes algorithm name."""
    solver = VRPSolver(algorithm="savings")
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(1, 0)],
        demands=[10],
        vehicle_capacity=100,
        distance_matrix=[[0, 1], [1, 0]],
    )
    result = solver.solve(instance)
    assert result.algorithm == "savings"


def test_vrp_multiple_vehicles():
    """VRP uses multiple vehicles when needed."""
    solver = VRPSolver()
    instance = VRPInstance(
        depot=(0, 0),
        customers=[(1, 0), (2, 0), (3, 0), (4, 0)],
        demands=[80, 80, 80, 80],
        vehicle_capacity=100,
        distance_matrix=[
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ],
    )
    result = solver.solve(instance)
    # Each customer needs own route (80+80 > 100)
    assert len(result.routes) == 4
    assert result.num_vehicles == 4

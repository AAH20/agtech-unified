"""Property-based tests using hypothesis for TSP/VRP/consensus invariants."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.multi_agent.consensus import ByzantineConsensus, ConsensusStatus
from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.vrp import VRPInstance, VRPSolver

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def metric_tsp_instances(draw, min_cities=2, max_cities=8):
    """Generate random metric TSP instances using Euclidean distances."""
    n = draw(st.integers(min_value=min_cities, max_value=max_cities))
    coords = draw(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            ),
            min_size=n,
            max_size=n,
        )
    )
    dist = [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(n)
        ]
        for i in range(n)
    ]
    cities = [f"city_{i}" for i in range(n)]
    return TSPInstance(cities=cities, distance_matrix=dist)


@st.composite
def vrp_instances(draw, min_customers=1, max_customers=6):
    """Generate random VRP instances with valid capacity constraints."""
    n = draw(st.integers(min_value=min_customers, max_value=max_customers))
    capacity = draw(st.floats(min_value=10, max_value=100, allow_nan=False, allow_infinity=False))
    demands = draw(
        st.lists(
            st.floats(min_value=1, max_value=capacity, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    coords = [(0.0, 0.0)] + draw(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=50, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0, max_value=50, allow_nan=False, allow_infinity=False),
            ),
            min_size=n,
            max_size=n,
        )
    )
    total = n + 1
    dist = [
        [
            math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            for j in range(total)
        ]
        for i in range(total)
    ]
    return VRPInstance(
        depot=(0.0, 0.0),
        customers=coords[1:],
        demands=demands,
        vehicle_capacity=capacity,
        distance_matrix=dist,
    )


@st.composite
def consensus_configs(draw):
    """Generate valid consensus configurations (n >= 3f+1)."""
    f = draw(st.integers(min_value=0, max_value=3))
    n = draw(st.integers(min_value=3 * f + 1, max_value=3 * f + 5))
    return f, n


# ---------------------------------------------------------------------------
# TSP Property Tests
# ---------------------------------------------------------------------------


class TestTSPProperties:
    """Property-based invariants for TSP solver."""

    @given(instance=metric_tsp_instances())
    @settings(max_examples=50, deadline=None)
    def test_tour_visits_each_city_exactly_once(self, instance):
        """Tour must be a permutation of all cities."""
        solver = TSPSolver(algorithm="christofides")
        result = solver.solve(instance)
        assert len(result.tour) == len(instance.cities)
        assert set(result.tour) == set(instance.cities)

    @given(instance=metric_tsp_instances())
    @settings(max_examples=50, deadline=None)
    def test_tour_cost_equals_sum_of_edges(self, instance):
        """Tour cost must equal sum of consecutive edge distances."""
        solver = TSPSolver(algorithm="christofides")
        result = solver.solve(instance)
        idx = {city: i for i, city in enumerate(instance.cities)}
        tour_idx = [idx[c] for c in result.tour]
        expected_cost = sum(
            instance.distance_matrix[tour_idx[i]][tour_idx[(i + 1) % len(tour_idx)]]
            for i in range(len(tour_idx))
        )
        assert result.cost == pytest.approx(expected_cost, rel=1e-9)

    @given(instance=metric_tsp_instances(min_cities=3))
    @settings(max_examples=50, deadline=None)
    def test_christofides_cost_bounded(self, instance):
        """Christofides produces a valid tour with finite non-negative cost."""
        solver = TSPSolver(algorithm="christofides")
        result = solver.solve(instance)
        assert result.cost >= 0
        assert math.isfinite(result.cost)
        # Cost should be at most n * max_edge (loose upper bound)
        n = len(instance.cities)
        max_edge = max(instance.distance_matrix[i][j] for i in range(n) for j in range(n))
        assert result.cost <= n * max_edge + 1e-9

    @given(instance=metric_tsp_instances())
    @settings(max_examples=50, deadline=None)
    def test_nearest_neighbor_produces_valid_tour(self, instance):
        """Nearest neighbor must produce a valid permutation."""
        solver = TSPSolver(algorithm="nearest_neighbor")
        result = solver.solve(instance)
        assert len(result.tour) == len(instance.cities)
        assert set(result.tour) == set(instance.cities)
        assert result.cost >= 0


# ---------------------------------------------------------------------------
# VRP Property Tests
# ---------------------------------------------------------------------------


class TestVRPProperties:
    """Property-based invariants for VRP solver."""

    @given(instance=vrp_instances())
    @settings(max_examples=50, deadline=None)
    def test_all_customers_visited_exactly_once(self, instance):
        """Every customer must appear in exactly one route."""
        solver = VRPSolver()
        result = solver.solve(instance)
        visited = [c for route in result.routes for c in route]
        assert sorted(visited) == list(range(len(instance.customers)))

    @given(instance=vrp_instances())
    @settings(max_examples=50, deadline=None)
    def test_route_demand_within_capacity(self, instance):
        """Total demand per route must not exceed vehicle capacity."""
        solver = VRPSolver()
        result = solver.solve(instance)
        for route in result.routes:
            total_demand = sum(instance.demands[c] for c in route)
            assert total_demand <= instance.vehicle_capacity + 1e-9

    @given(instance=vrp_instances())
    @settings(max_examples=50, deadline=None)
    def test_total_cost_equals_sum_of_route_costs(self, instance):
        """Total cost must equal sum of individual route costs."""
        solver = VRPSolver()
        result = solver.solve(instance)
        dist = instance.distance_matrix
        expected = 0.0
        for route in result.routes:
            expected += dist[0][route[0] + 1]
            for k in range(len(route) - 1):
                expected += dist[route[k] + 1][route[k + 1] + 1]
            expected += dist[route[-1] + 1][0]
        assert result.total_cost == pytest.approx(expected, rel=1e-9)

    @given(instance=vrp_instances())
    @settings(max_examples=50, deadline=None)
    def test_num_vehicles_matches_routes(self, instance):
        """num_vehicles must equal number of non-empty routes."""
        solver = VRPSolver()
        result = solver.solve(instance)
        assert result.num_vehicles == len(result.routes)


# ---------------------------------------------------------------------------
# Consensus Property Tests
# ---------------------------------------------------------------------------


class TestConsensusProperties:
    """Property-based invariants for Byzantine consensus."""

    @given(config=consensus_configs())
    @settings(max_examples=50, deadline=None)
    def test_quorum_is_2f_plus_1(self, config):
        """Quorum must always be 2f+1."""
        f, n = config
        consensus = ByzantineConsensus(node_id="n1", total_nodes=n, max_faulty=f)
        assert consensus.quorum == 2 * f + 1

    @given(config=consensus_configs())
    @settings(max_examples=50, deadline=None)
    def test_n_geq_3f_plus_1_constraint(self, config):
        """System must satisfy n >= 3f+1."""
        f, n = config
        consensus = ByzantineConsensus(node_id="n1", total_nodes=n, max_faulty=f)
        assert consensus.total_nodes >= 3 * consensus.max_faulty + 1

    @given(
        config=consensus_configs(),
        votes=st.lists(st.text(min_size=1, max_size=5), min_size=0, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_tally_never_commits_without_quorum(self, config, votes):
        """Consensus must NOT commit unless a value has >= 2f+1 votes."""
        f, n = config
        consensus = ByzantineConsensus(node_id="n1", total_nodes=n, max_faulty=f)
        result = consensus.tally_votes(votes)
        if result.status == ConsensusStatus.COMMITTED:
            # If committed, the winning value must have >= quorum votes
            count = votes.count(result.value)
            assert count >= consensus.quorum

    @given(
        config=consensus_configs(),
        votes=st.lists(st.text(min_size=1, max_size=5), min_size=0, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_tally_commits_with_sufficient_votes(self, config, votes):
        """Consensus MUST commit when a value has >= 2f+1 votes."""
        f, n = config
        consensus = ByzantineConsensus(node_id="n1", total_nodes=n, max_faulty=f)
        result = consensus.tally_votes(votes)
        # Check if any value has >= quorum votes
        from collections import Counter

        counts = Counter(votes)
        max_count = max(counts.values()) if counts else 0
        if max_count >= consensus.quorum:
            assert result.status == ConsensusStatus.COMMITTED
        else:
            assert result.status == ConsensusStatus.PENDING

    @given(config=consensus_configs())
    @settings(max_examples=50, deadline=None)
    def test_empty_votes_returns_pending(self, config):
        """Empty vote list must return PENDING."""
        f, n = config
        consensus = ByzantineConsensus(node_id="n1", total_nodes=n, max_faulty=f)
        result = consensus.tally_votes([])
        assert result.status == ConsensusStatus.PENDING
        assert result.votes_received == 0

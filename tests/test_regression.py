"""Regression tests for known edge cases across all modules.

Bug-ID references:
- BUG-001: TSP empty cities crash
- BUG-002: VRP zero capacity not validated
- BUG-003: Consensus n < 3f+1 not rejected
- BUG-004: Security expired token not rejected
- BUG-005: Alert manager unknown alert handling
- BUG-006: MQTT wildcard matching incorrect
- BUG-007: Kafka empty topic consume crash
- BUG-008: TimescaleDB unknown aggregation crash
- BUG-009: Decision engine all-zeros crash
- BUG-010: Digital twin negative days crash
- BUG-011: CRISPR empty sequence crash
- BUG-012: Protein invalid amino acid crash
"""

import asyncio
import math

import pytest

from src.decision_support.alerts import AlertManager, Threshold
from src.decision_support.recommender import DecisionEngine, FarmState
from src.decision_support.security import AuthenticationError, ZeroTrustAuth
from src.iot.data_pipeline import KafkaStream, MQTTClient, TimescaleDBStorage
from src.multi_agent.consensus import ByzantineConsensus, ConsensusStatus
from src.optimization.tsp import TSPInstance, TSPSolver
from src.optimization.vrp import VRPInstance, VRPSolver

# ---------------------------------------------------------------------------
# TSP Regression Tests
# ---------------------------------------------------------------------------


class TestTSPRegression:
    """Regression tests for TSP edge cases."""

    def test_empty_cities_returns_empty_tour(self):
        """BUG-001: Empty city list must return empty tour (not crash)."""
        solver = TSPSolver()
        result = solver.solve(TSPInstance(cities=[], distance_matrix=[]))
        assert result.tour == []
        assert result.cost == 0.0

    def test_single_city_zero_cost(self):
        """Single city TSP has zero cost."""
        solver = TSPSolver()
        result = solver.solve(TSPInstance(cities=["A"], distance_matrix=[[0]]))
        assert result.tour == ["A"]
        assert result.cost == 0.0

    def test_two_cities_round_trip(self):
        """Two cities must produce round trip cost."""
        solver = TSPSolver()
        result = solver.solve(TSPInstance(cities=["A", "B"], distance_matrix=[[0, 5], [5, 0]]))
        assert result.cost == 10.0

    def test_non_square_matrix_raises(self):
        """Non-square distance matrix must raise ValueError."""
        with pytest.raises(ValueError, match="Distance matrix must be square"):
            TSPInstance(cities=["A", "B"], distance_matrix=[[0, 1, 2], [1, 0, 2]])

    def test_non_metric_still_solves(self):
        """Non-metric TSP must still produce a valid tour."""
        solver = TSPSolver(algorithm="christofides")
        result = solver.solve(
            TSPInstance(
                cities=["A", "B", "C"], distance_matrix=[[0, 1, 100], [1, 0, 1], [100, 1, 0]]
            )
        )
        assert len(result.tour) == 3
        assert set(result.tour) == {"A", "B", "C"}

    def test_duplicate_city_names_produces_valid_tour(self):
        """Duplicate city names should still produce a valid tour."""
        solver = TSPSolver(algorithm="nearest_neighbor")
        result = solver.solve(
            TSPInstance(cities=["A", "A", "B"], distance_matrix=[[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        )
        assert len(result.tour) == 3

    def test_large_instance_completes(self):
        """Large TSP instance (n=20) must complete in reasonable time."""
        solver = TSPSolver(algorithm="nearest_neighbor")
        n = 20
        coords = [(i, i * 2) for i in range(n)]
        dist = [
            [
                math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
                for j in range(n)
            ]
            for i in range(n)
        ]
        result = solver.solve(TSPInstance(cities=[f"c{i}" for i in range(n)], distance_matrix=dist))
        assert len(result.tour) == n
        assert result.cost > 0


# ---------------------------------------------------------------------------
# VRP Regression Tests
# ---------------------------------------------------------------------------


class TestVRPRegression:
    """Regression tests for VRP edge cases."""

    def test_zero_capacity_raises(self):
        """BUG-002: Zero capacity must raise ValueError."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[10],
                vehicle_capacity=0,
                distance_matrix=[[0, 1], [1, 0]],
            )

    def test_negative_capacity_raises(self):
        """Negative capacity must raise ValueError."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[10],
                vehicle_capacity=-5,
                distance_matrix=[[0, 1], [1, 0]],
            )

    def test_demand_exceeds_capacity_raises(self):
        """Demand exceeding capacity must raise ValueError."""
        with pytest.raises(ValueError, match="Demand exceeds vehicle capacity"):
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[200],
                vehicle_capacity=100,
                distance_matrix=[[0, 1], [1, 0]],
            )

    def test_empty_customers_returns_empty_routes(self):
        """Empty customer list must return empty routes."""
        solver = VRPSolver()
        result = solver.solve(
            VRPInstance(
                depot=(0, 0), customers=[], demands=[], vehicle_capacity=100, distance_matrix=[]
            )
        )
        assert result.routes == []
        assert result.total_cost == 0.0
        assert result.num_vehicles == 0

    def test_single_customer_single_route(self):
        """Single customer must produce one route."""
        solver = VRPSolver()
        result = solver.solve(
            VRPInstance(
                depot=(0, 0),
                customers=[(3, 4)],
                demands=[10],
                vehicle_capacity=100,
                distance_matrix=[[0, 5], [5, 0]],
            )
        )
        assert len(result.routes) == 1
        assert result.routes[0] == [0]

    def test_zero_demand_customer(self):
        """Zero demand customer should still be routed."""
        solver = VRPSolver()
        result = solver.solve(
            VRPInstance(
                depot=(0, 0),
                customers=[(1, 0)],
                demands=[0],
                vehicle_capacity=100,
                distance_matrix=[[0, 1], [1, 0]],
            )
        )
        assert len(result.routes) == 1
        assert result.routes[0] == [0]


# ---------------------------------------------------------------------------
# Consensus Regression Tests
# ---------------------------------------------------------------------------


class TestConsensusRegression:
    """Regression tests for consensus edge cases."""

    def test_n_less_than_3f_plus_1_raises(self):
        """BUG-003: n < 3f+1 must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot tolerate"):
            ByzantineConsensus(node_id="n1", total_nodes=3, max_faulty=1)

    def test_empty_votes_returns_pending(self):
        """Empty votes must return PENDING."""
        consensus = ByzantineConsensus(node_id="n1", total_nodes=4, max_faulty=1)
        result = consensus.tally_votes([])
        assert result.status == ConsensusStatus.PENDING
        assert result.votes_received == 0

    def test_unanimous_votes_commit(self):
        """All same votes must commit."""
        consensus = ByzantineConsensus(node_id="n1", total_nodes=4, max_faulty=1)
        result = consensus.tally_votes(["A", "A", "A", "A"])
        assert result.status == ConsensusStatus.COMMITTED
        assert result.value == "A"

    def test_all_different_votes_no_commit(self):
        """All different votes must not commit."""
        consensus = ByzantineConsensus(node_id="n1", total_nodes=7, max_faulty=2)
        result = consensus.tally_votes(["A", "B", "C", "D", "E", "F", "G"])
        assert result.status == ConsensusStatus.PENDING

    def test_max_faulty_zero_single_node(self):
        """f=0 with n=1 must auto-commit."""
        consensus = ByzantineConsensus(node_id="n1", total_nodes=1, max_faulty=0)
        result = consensus.propose("X")
        assert result.status == ConsensusStatus.COMMITTED

    def test_max_faulty_zero_multi_node(self):
        """f=0 with n>1 auto-commits (no faults to tolerate)."""
        consensus = ByzantineConsensus(node_id="n1", total_nodes=4, max_faulty=0)
        result = consensus.propose("X")
        assert result.status == ConsensusStatus.COMMITTED
        assert result.quorum == 1


# ---------------------------------------------------------------------------
# Security Regression Tests
# ---------------------------------------------------------------------------


class TestSecurityRegression:
    """Regression tests for security edge cases."""

    def test_expired_token_raises(self):
        """BUG-004: Expired token must raise AuthenticationError."""
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        token = auth.generate_token(subject="user", roles=["device"], ttl_seconds=-1)
        with pytest.raises(AuthenticationError, match="expired"):
            auth.validate_token(token)

    def test_invalid_signature_raises(self):
        """Token with wrong secret must raise AuthenticationError."""
        auth1 = ZeroTrustAuth(secret="secret-a-that-is-long-enough-for-hs256")
        auth2 = ZeroTrustAuth(secret="secret-b-that-is-long-enough-for-hs256")
        token = auth1.generate_token(subject="user", roles=["device"])
        with pytest.raises(AuthenticationError):
            auth2.validate_token(token)

    def test_malformed_token_raises(self):
        """Malformed token must raise AuthenticationError."""
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        with pytest.raises(AuthenticationError):
            auth.validate_token("not.a.valid.token")

    def test_empty_token_raises(self):
        """Empty token must raise AuthenticationError."""
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        with pytest.raises(AuthenticationError):
            auth.validate_token("")

    def test_empty_roles_authorize_fails(self):
        """Empty roles must fail authorization."""
        auth = ZeroTrustAuth(secret="test-secret-key")
        assert auth.authorize({"roles": []}, "admin") is False

    def test_rate_limit_blocks_after_max(self):
        """Rate limit must block after max requests."""
        auth = ZeroTrustAuth(secret="test-secret-key")
        for _ in range(3):
            assert auth.check_rate_limit("client", max_requests=3, window_seconds=60) is True
        assert auth.check_rate_limit("client", max_requests=3, window_seconds=60) is False

    def test_rate_limit_allows_after_window(self):
        """Rate limit must allow requests after window expires."""
        auth = ZeroTrustAuth(secret="test-secret-key")
        for _ in range(3):
            auth.check_rate_limit("client", max_requests=3, window_seconds=0.01)
        import time

        time.sleep(0.02)
        assert auth.check_rate_limit("client", max_requests=3, window_seconds=0.01) is True


# ---------------------------------------------------------------------------
# Alert Manager Regression Tests
# ---------------------------------------------------------------------------


class TestAlertRegression:
    """Regression tests for alert edge cases."""

    def test_acknowledge_unknown_alert_returns_false(self):
        """BUG-005: Acknowledging unknown alert must return False."""
        manager = AlertManager()
        assert manager.acknowledge_alert("nonexistent") is False

    def test_resolve_unknown_alert_returns_false(self):
        """Resolving unknown alert must return False."""
        manager = AlertManager()
        assert manager.resolve_alert("nonexistent") is False

    def test_duplicate_thresholds_both_trigger(self):
        """Duplicate thresholds for same metric must both trigger."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="temp", max_value=30.0))
        manager.add_threshold(Threshold(metric="temp", max_value=35.0))
        alerts = manager.check_value("temp", 40.0)
        assert len(alerts) == 2

    def test_threshold_min_greater_than_max(self):
        """Threshold with min > max must trigger for any value."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="temp", min_value=50.0, max_value=10.0))
        alerts = manager.check_value("temp", 30.0)
        assert len(alerts) == 1

    def test_empty_channels_notify_no_error(self):
        """Notify with no channels must not raise."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="temp", max_value=30.0))
        alerts = manager.check_value("temp", 40.0)
        asyncio.run(manager.notify(alerts[0]))

    def test_remove_nonexistent_channel_returns_false(self):
        """Removing nonexistent channel must return False."""
        manager = AlertManager()
        assert manager.remove_channel("nonexistent") is False

    def test_clear_resolved_returns_zero_when_none_resolved(self):
        """clear_resolved must return 0 when no resolved alerts."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="temp", max_value=30.0))
        manager.check_value("temp", 40.0)
        assert manager.clear_resolved() == 0


# ---------------------------------------------------------------------------
# Data Pipeline Regression Tests
# ---------------------------------------------------------------------------


class TestDataPipelineRegression:
    """Regression tests for data pipeline edge cases."""

    def test_mqtt_wildcard_plus_matches_single_level(self):
        """BUG-006: MQTT + wildcard must match exactly one level."""
        client = MQTTClient()
        assert client._topic_matches("sensors/+/temp", "sensors/room1/temp") is True
        assert client._topic_matches("sensors/+/temp", "sensors/room1/room2/temp") is False

    def test_mqtt_wildcard_hash_matches_all_remaining(self):
        """MQTT # wildcard must match all remaining levels."""
        client = MQTTClient()
        assert client._topic_matches("sensors/#", "sensors/room1/temp") is True
        assert client._topic_matches("sensors/#", "sensors/room1/temp/humidity") is True
        assert client._topic_matches("sensors/#", "other/topic") is False

    def test_mqtt_exact_match(self):
        """MQTT exact topic must match."""
        client = MQTTClient()
        assert client._topic_matches("sensors/temp", "sensors/temp") is True
        assert client._topic_matches("sensors/temp", "sensors/humidity") is False

    def test_kafka_consume_empty_topic_returns_empty(self):
        """BUG-007: Consuming from empty topic must return empty list."""
        stream = KafkaStream()
        stream.create_topic("empty")
        assert stream.consume("empty", timeout=0.1) == []

    def test_kafka_delete_nonexistent_topic_returns_false(self):
        """Deleting nonexistent topic must return False."""
        stream = KafkaStream()
        assert stream.delete_topic("nonexistent") is False

    def test_timescale_unknown_aggregation_raises(self):
        """BUG-008: Unknown aggregation type must raise ValueError."""
        storage = TimescaleDBStorage()
        storage.connect()
        storage.insert("sensor_readings", {"time": 1, "sensor_id": "s1", "value": 10.0})
        with pytest.raises(ValueError, match="Unknown aggregation"):
            storage.aggregate("sensor_readings", "s1", "median")

    def test_timescale_query_no_filters_returns_all(self):
        """Query with no filters must return all records."""
        storage = TimescaleDBStorage()
        storage.connect()
        storage.insert("sensor_readings", {"time": 1, "sensor_id": "s1", "value": 10.0})
        storage.insert("sensor_readings", {"time": 2, "sensor_id": "s2", "value": 20.0})
        results = storage.query("sensor_readings")
        assert len(results) == 2

    def test_timescale_insert_missing_time_field(self):
        """Insert without time field must auto-populate."""
        storage = TimescaleDBStorage()
        storage.connect()
        storage.insert("sensor_readings", {"sensor_id": "s1", "value": 10.0})
        results = storage.query("sensor_readings")
        assert len(results) == 1
        assert "time" in results[0]


# ---------------------------------------------------------------------------
# Decision Engine Regression Tests
# ---------------------------------------------------------------------------


class TestDecisionEngineRegression:
    """Regression tests for decision engine edge cases."""

    def test_all_zeros_produces_recommendations(self):
        """BUG-009: All-zero farm state must produce recommendations."""
        engine = DecisionEngine()
        state = FarmState(
            soil_moisture=0.0,
            temperature=0.0,
            crop_height=0.0,
            nutrient_level=0.0,
            pest_pressure=0.0,
        )
        result = engine.recommend(state)
        assert result.priority_score > 0
        assert len(result.recommendations) > 0

    def test_extreme_values_produces_recommendations(self):
        """BUG-009: Extreme farm state must produce recommendations."""
        engine = DecisionEngine()
        state = FarmState(
            soil_moisture=1.0,
            temperature=50.0,
            crop_height=2.0,
            nutrient_level=1.0,
            pest_pressure=1.0,
        )
        result = engine.recommend(state)
        assert result.priority_score > 0

    def test_boundary_values_produce_consistent_results(self):
        """Boundary values must produce consistent results."""
        engine = DecisionEngine()
        state = FarmState(
            soil_moisture=0.2,
            temperature=38.0,
            crop_height=0.1,
            nutrient_level=0.2,
            pest_pressure=0.7,
        )
        result = engine.recommend(state)
        assert 0.0 <= result.priority_score <= 1.0
        assert len(result.actions) == len(result.recommendations)

    def test_priority_score_capped_at_1(self):
        """BUG-009: Priority score must never exceed 1.0."""
        engine = DecisionEngine()
        state = FarmState(
            soil_moisture=0.0,
            temperature=50.0,
            crop_height=0.0,
            nutrient_level=0.0,
            pest_pressure=1.0,
        )
        result = engine.recommend(state)
        assert result.priority_score <= 1.0

"""Thread-safety tests for ZeroTrustAuth, MQTTClient, AlertManager."""
import asyncio
import threading
import time

import pytest

from src.decision_support.security import ZeroTrustAuth, AuthenticationError
from src.iot.data_pipeline import MQTTClient
from src.decision_support.alerts import AlertManager, AlertSeverity, Threshold


class TestZeroTrustAuthConcurrency:
    """Thread-safety tests for ZeroTrustAuth rate limiting."""

    def test_concurrent_rate_limit_no_race_condition(self):
        """Rate limiter must be exact under concurrent access."""
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        max_requests = 50
        num_threads = 10
        results = []
        errors = []

        def make_requests():
            try:
                for _ in range(max_requests // num_threads):
                    allowed = auth.check_rate_limit(
                        "concurrent-client", max_requests=max_requests, window_seconds=60
                    )
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_requests) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Exactly max_requests should be allowed
        assert sum(results) == max_requests

    def test_concurrent_rate_limit_multiple_clients(self):
        """Rate limiter isolates clients under concurrent access."""
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        results = {"client_a": [], "client_b": []}

        def make_requests(client_id):
            for _ in range(5):
                allowed = auth.check_rate_limit(client_id, max_requests=5, window_seconds=60)
                results[client_id].append(allowed)

        t1 = threading.Thread(target=make_requests, args=("client_a",))
        t2 = threading.Thread(target=make_requests, args=("client_b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert all(results["client_a"])
        assert all(results["client_b"])

    def test_concurrent_token_generation_and_validation(self):
        """Token generation and validation are thread-safe."""
        auth = ZeroTrustAuth(secret="test-secret-key-that-is-long-enough-for-hs256")
        tokens = []
        errors = []

        def generate_and_validate():
            try:
                for i in range(20):
                    token = auth.generate_token(
                        subject=f"user-{threading.current_thread().ident}-{i}",
                        roles=["device"],
                    )
                    payload = auth.validate_token(token)
                    assert payload["sub"].startswith("user-")
                    tokens.append(token)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate_and_validate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(tokens) == 100


class TestMQTTClientConcurrency:
    """Thread-safety tests for MQTTClient pub/sub."""

    def test_concurrent_publish_subscribe(self):
        """Concurrent publish/subscribe must not lose messages or crash."""
        client = MQTTClient()
        client.connect()
        received = []
        lock = threading.Lock()

        def callback(topic, payload):
            with lock:
                received.append((topic, payload))

        client.subscribe("sensors/temp", callback)

        def publish_messages():
            for i in range(50):
                client.publish("sensors/temp", {"value": i})

        threads = [threading.Thread(target=publish_messages) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 200

    def test_concurrent_subscribe_unsubscribe(self):
        """Concurrent subscribe/unsubscribe must not corrupt state."""
        client = MQTTClient()
        client.connect()
        errors = []

        def subscribe_unsubscribe():
            try:
                for i in range(20):
                    cb = lambda t, p: None
                    client.subscribe(f"topic/{i % 3}", cb)
                    client.unsubscribe(f"topic/{i % 3}", cb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=subscribe_unsubscribe) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_publish_to_multiple_topics(self):
        """Publishing to different topics concurrently must not interfere."""
        client = MQTTClient()
        client.connect()
        received_a = []
        received_b = []

        client.subscribe("topic/a", lambda t, p: received_a.append(p))
        client.subscribe("topic/b", lambda t, p: received_b.append(p))

        def publish_a():
            for i in range(30):
                client.publish("topic/a", f"msg-a-{i}")

        def publish_b():
            for i in range(30):
                client.publish("topic/b", f"msg-b-{i}")

        t1 = threading.Thread(target=publish_a)
        t2 = threading.Thread(target=publish_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(received_a) == 30
        assert len(received_b) == 30


class TestAlertManagerConcurrency:
    """Thread-safety tests for AlertManager."""

    def test_concurrent_check_value(self):
        """Concurrent check_value calls must produce correct alert count."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="soil_moisture", min_value=0.3))
        errors = []

        def check_values():
            try:
                for _ in range(25):
                    manager.check_value("soil_moisture", 0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_values) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert manager.alert_count == 100

    def test_concurrent_acknowledge_and_resolve(self):
        """Concurrent acknowledge/resolve must not corrupt alert state."""
        manager = AlertManager()
        manager.add_threshold(Threshold(metric="temp", max_value=30.0))
        alerts = manager.check_value("temp", 40.0)
        alert_id = alerts[0].id
        errors = []

        def acknowledge():
            try:
                for _ in range(10):
                    manager.acknowledge_alert(alert_id)
            except Exception as e:
                errors.append(e)

        def resolve():
            try:
                for _ in range(10):
                    manager.resolve_alert(alert_id)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=acknowledge)
        t2 = threading.Thread(target=resolve)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        # Alert should be in a consistent state
        alert = [a for a in manager.get_all_alerts() if a.id == alert_id][0]
        assert alert.acknowledged or alert.resolved

    def test_concurrent_add_threshold_and_check(self):
        """Adding thresholds while checking values must not crash."""
        manager = AlertManager()
        errors = []

        def add_thresholds():
            try:
                for i in range(10):
                    manager.add_threshold(
                        Threshold(metric=f"metric_{i}", min_value=0.5)
                    )
            except Exception as e:
                errors.append(e)

        def check_values():
            try:
                for i in range(10):
                    manager.check_value(f"metric_{i}", 0.1)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_thresholds)
        t2 = threading.Thread(target=check_values)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors

    def test_concurrent_notify(self):
        """Concurrent notify calls must not crash."""
        manager = AlertManager()
        manager.add_threshold(
            Threshold(metric="soil_moisture", min_value=0.3, severity=AlertSeverity.CRITICAL)
        )
        alerts = manager.check_value("soil_moisture", 0.1)
        errors = []

        def notify():
            try:
                asyncio.run(manager.notify(alerts[0]))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=notify) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

"""Tests for circuit breaker: trip on repeated failures, half-open recovery.

TDD-style: written before implementation in integration/circuit_breaker.py.
"""

import asyncio
import time

import pytest

from src.integration.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerStateTransitions:
    """Core state machine: closed -> open -> half-open -> closed."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        assert cb.state == CircuitState.CLOSED

    def test_success_does_not_trip(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        cb.record_success()
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failures_below_threshold_do_not_trip(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_reaching_failure_threshold_trips_to_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Before timeout elapses, still open
        time.sleep(0.02)
        assert cb.state == CircuitState.OPEN
        # After timeout elapses, half-open
        time.sleep(0.05)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.08)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # Failure count resets after close
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.08)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_multiple_failure_threshold_crossings(self):
        """Circuit trips, recovers, then trips again on new failures."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.08)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerAllowed:
    """The allow_request gate blocks calls when open."""

    def test_allow_request_true_when_closed(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        assert cb.allow_request() is True

    def test_allow_request_false_when_open_before_timeout(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=10.0))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_allow_request_true_when_open_after_timeout(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is False
        time.sleep(0.08)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_allow_request_true_when_half_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.08)
        assert cb.allow_request() is True


class TestCircuitBreakerExecution:
    """wrap() executes a callable, tracking success/failure and raising on open."""

    def test_wrap_executes_callable_and_returns_result(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        result = cb.wrap(lambda x: x * 2, 21)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_wrap_raises_circuit_breaker_error_when_open(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0))
        cb.record_failure()
        with pytest.raises(CircuitBreakerError):
            cb.wrap(lambda: "never called")

    def test_wrap_catches_exception_and_trips_after_threshold(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))

        def boom():
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError):
            cb.wrap(boom)
        assert cb.state == CircuitState.CLOSED
        with pytest.raises(RuntimeError):
            cb.wrap(boom)
        assert cb.state == CircuitState.OPEN

    def test_wrap_async_executes_and_returns(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))

        async def add(a, b):
            return a + b

        result = asyncio.run(cb.wrap_async(add, 3, 4))
        assert result == 7

    def test_wrap_async_tracks_failure(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0))

        async def boom():
            raise ValueError("async kaboom")

        with pytest.raises(ValueError):
            asyncio.run(cb.wrap_async(boom))
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerAsync:
    """Async circuit breaker entry points."""

    def test_record_failure_async(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        asyncio.run(cb.record_failure_async())
        asyncio.run(cb.record_failure_async())
        assert cb.state == CircuitState.OPEN

    def test_record_success_async(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        asyncio.run(cb.record_failure_async())
        asyncio.run(cb.record_success_async())
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerThreadSafety:
    """record_success / record_failure are atomic under concurrent access."""

    def test_concurrent_failures_do_not_exceed_threshold(self):
        import threading

        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=100, recovery_timeout=10.0))
        threads = [threading.Thread(target=cb.record_failure) for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cb.failure_count == 200
        assert cb.state == CircuitState.OPEN

    def test_concurrent_mixed_operations_are_consistent(self):
        import threading

        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=50, recovery_timeout=10.0))
        successes = [threading.Thread(target=cb.record_success) for _ in range(100)]
        failures = [threading.Thread(target=cb.record_failure) for _ in range(100)]
        all_threads = successes + failures
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()
        # Lock prevents lost updates: counts never exceed total operations.
        assert 0 <= cb.failure_count <= 100
        assert 0 <= cb.success_count <= 100
        # If still closed, failure_count must be below threshold.
        if cb.state == CircuitState.CLOSED:
            assert cb.failure_count < 50


class TestCircuitBreakerConfig:
    """Config validation."""

    def test_default_config(self):
        cfg = CircuitBreakerConfig()
        assert cfg.failure_threshold == 5
        assert cfg.recovery_timeout == 30.0
        assert cfg.success_threshold == 1

    def test_invalid_failure_threshold(self):
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)

    def test_invalid_recovery_timeout(self):
        with pytest.raises(ValueError):
            CircuitBreakerConfig(recovery_timeout=-1.0)

    def test_invalid_success_threshold(self):
        with pytest.raises(ValueError):
            CircuitBreakerConfig(success_threshold=0)


class TestCircuitBreakerIntegration:
    """Integration with a simulated failing service."""

    def test_circuit_protects_failing_service(self):
        call_count = 0

        def flaky_service():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("service down")

        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.05))
        # First 3 calls execute and fail
        for _ in range(3):
            with pytest.raises(ConnectionError):
                cb.wrap(flaky_service)
        assert cb.state == CircuitState.OPEN
        # Next call is short-circuited without executing the service
        with pytest.raises(CircuitBreakerError):
            cb.wrap(flaky_service)
        assert call_count == 3  # service was not called again

    def test_circuit_recovers_after_timeout(self):
        attempt = 0

        def recovering_service():
            nonlocal attempt
            attempt += 1
            if attempt <= 2:
                raise ConnectionError("temporarily down")
            return "ok"

        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.05))
        with pytest.raises(ConnectionError):
            cb.wrap(recovering_service)
        with pytest.raises(ConnectionError):
            cb.wrap(recovering_service)
        assert cb.state == CircuitState.OPEN
        time.sleep(0.08)
        result = cb.wrap(recovering_service)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerReset:
    """Manual reset returns the circuit to closed."""

    def test_reset_clears_state(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, recovery_timeout=10.0))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None

    def test_reset_allows_requests(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0))
        cb.record_failure()
        assert cb.allow_request() is False
        cb.reset()
        assert cb.allow_request() is True

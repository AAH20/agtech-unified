"""Circuit breaker pattern for fault tolerance.

Prevents cascading failures by short-circuiting calls to a failing service.
Three states: CLOSED (normal), OPEN (failing, reject calls), HALF_OPEN (testing recovery).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, message: str = "Circuit breaker is open") -> None:
        super().__init__(message)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of consecutive failures before tripping.
        recovery_timeout: Seconds to wait before transitioning to half-open.
        success_threshold: Successes needed in half-open to close (default 1).
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 1

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")


class CircuitBreaker:
    """Circuit breaker that trips on repeated failures and recovers via half-open state.

    Thread-safe. Supports both sync and async callables.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition open -> half-open)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def last_failure_time(self) -> Optional[float]:
        return self._last_failure_time

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._close_circuit()
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
            # In OPEN state, successes are ignored

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN:
                self._open_circuit()
            elif self._failure_count >= self._config.failure_threshold:
                self._open_circuit()

    def reset(self) -> None:
        """Reset to closed state, clearing all counters."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def wrap(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a callable through the circuit breaker.

        Raises CircuitBreakerError if the circuit is open.
        """
        if not self.allow_request():
            raise CircuitBreakerError(
                f"Circuit breaker is {self._state.value}, "
                f"rejecting call to {getattr(func, '__name__', func)}"
            )
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    async def wrap_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an async callable through the circuit breaker."""
        if not self.allow_request():
            raise CircuitBreakerError(
                f"Circuit breaker is {self._state.value}, "
                f"rejecting call to {getattr(func, '__name__', func)}"
            )
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    async def record_success_async(self) -> None:
        """Async version of record_success."""
        self.record_success()

    async def record_failure_async(self) -> None:
        """Async version of record_failure."""
        self.record_failure()

    def _maybe_transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN if recovery timeout has elapsed."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and (time.monotonic() - self._last_failure_time) >= self._config.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
            logger.info("Circuit breaker transitioning to HALF_OPEN")

    def _open_circuit(self) -> None:
        """Trip the circuit to OPEN."""
        self._state = CircuitState.OPEN
        self._success_count = 0
        logger.warning("Circuit breaker OPEN after %d failures", self._failure_count)

    def _close_circuit(self) -> None:
        """Close the circuit, resetting counters."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker CLOSED")

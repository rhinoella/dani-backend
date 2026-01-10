"""
Circuit Breaker implementation for resilient external service calls.

Implements the Circuit Breaker pattern to:
- Prevent cascading failures when services are down
- Provide graceful degradation
- Allow services to recover without overwhelming them

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, requests fail fast
- HALF_OPEN: Testing if service has recovered
"""

from __future__ import annotations

import logging
import time
import asyncio
from enum import Enum
from typing import Optional, Callable, Any, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Number of failures before opening circuit
    failure_threshold: int = 5
    # Time in seconds before attempting recovery
    recovery_timeout: float = 30.0
    # Number of successes in half-open to close circuit
    success_threshold: int = 2
    # Exceptions that trigger the circuit breaker
    expected_exceptions: tuple = (Exception,)
    # Name for logging
    name: str = "default"


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected when open
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerOpen(Exception):
    """Raised when circuit is open and requests are being rejected."""
    
    def __init__(self, name: str, recovery_time: float):
        self.name = name
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Service unavailable. Will retry in {recovery_time:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit breaker for protecting external service calls.
    
    Usage:
        breaker = CircuitBreaker(CircuitBreakerConfig(name="ollama"))
        
        @breaker
        async def call_ollama():
            ...
        
        # Or manually:
        async with breaker:
            result = await risky_operation()
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._state != CircuitState.OPEN:
            return False
        
        elapsed = time.time() - self._last_state_change
        return elapsed >= self.config.recovery_timeout
    
    def _time_until_reset(self) -> float:
        """Get remaining time until recovery attempt."""
        if self._state != CircuitState.OPEN:
            return 0.0
        
        elapsed = time.time() - self._last_state_change
        return max(0.0, self.config.recovery_timeout - elapsed)
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        
        logger.info(
            f"Circuit breaker '{self.config.name}' transitioned: "
            f"{old_state.value} -> {new_state.value}"
        )
        
        if new_state == CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.config.name}' OPENED after "
                f"{self._stats.consecutive_failures} consecutive failures"
            )
        elif new_state == CircuitState.CLOSED:
            logger.info(
                f"Circuit breaker '{self.config.name}' CLOSED - service recovered"
            )
    
    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()
            
            # In half-open state, check if we can close
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()
            
            # Check if we should open the circuit
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
            
            # In half-open state, a single failure opens the circuit
            elif self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
    
    async def _record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self._stats.rejected_calls += 1
    
    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry - check if call is allowed."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to(CircuitState.HALF_OPEN)
                else:
                    await self._record_rejection()
                    raise CircuitBreakerOpen(
                        self.config.name,
                        self._time_until_reset()
                    )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit - record result."""
        if exc_type is None:
            await self._record_success()
        elif isinstance(exc_val, self.config.expected_exceptions):
            await self._record_failure()
        
        # Don't suppress exceptions
        return False
    
    def __enter__(self) -> "CircuitBreaker":
        """Sync context manager entry - check if call is allowed."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = time.time()
                logger.info(
                    f"Circuit breaker '{self.config.name}' transitioned: "
                    f"open -> half_open"
                )
            else:
                self._stats.rejected_calls += 1
                raise CircuitBreakerOpen(
                    self.config.name,
                    self._time_until_reset()
                )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Sync context manager exit - record result."""
        if exc_type is None:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()
            
            # In half-open state, check if we can close
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._last_state_change = time.time()
                    logger.info(
                        f"Circuit breaker '{self.config.name}' CLOSED - service recovered"
                    )
        elif isinstance(exc_val, self.config.expected_exceptions):
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.time()
                    logger.warning(
                        f"Circuit breaker '{self.config.name}' OPENED after "
                        f"{self._stats.consecutive_failures} consecutive failures"
                    )
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
        
        # Don't suppress exceptions
        return False
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for async functions."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper
    
    def get_status(self) -> dict:
        """Get current status for monitoring."""
        return {
            "name": self.config.name,
            "state": self._state.value,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "rejected_calls": self._stats.rejected_calls,
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
            },
            "time_until_reset": self._time_until_reset() if self.is_open else None,
        }
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
        logger.info(f"Circuit breaker '{self.config.name}' manually reset")
    
    async def force_open(self) -> None:
        """Manually open the circuit (for testing/maintenance)."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)
        logger.warning(f"Circuit breaker '{self.config.name}' manually opened")


# Pre-configured circuit breakers for common services
ollama_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="ollama",
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        expected_exceptions=(
            ConnectionError,
            TimeoutError,
            Exception,  # Catch-all for httpx errors
        ),
    )
)

qdrant_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="qdrant",
        failure_threshold=5,
        recovery_timeout=15.0,
        success_threshold=2,
        expected_exceptions=(
            ConnectionError,
            TimeoutError,
            Exception,
        ),
    )
)

embeddings_breaker = CircuitBreaker(
    CircuitBreakerConfig(
        name="embeddings",
        failure_threshold=3,
        recovery_timeout=20.0,
        success_threshold=2,
        expected_exceptions=(
            ConnectionError,
            TimeoutError,
            Exception,
        ),
    )
)


def get_all_breaker_status() -> dict:
    """Get status of all circuit breakers."""
    return {
        "ollama": ollama_breaker.get_status(),
        "qdrant": qdrant_breaker.get_status(),
        "embeddings": embeddings_breaker.get_status(),
    }

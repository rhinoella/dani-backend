"""
Tests for Circuit Breaker.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    CircuitStats,
)


# ============== Fixtures ==============

@pytest.fixture
def config():
    """Create default circuit breaker config."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=1.0,  # Short timeout for tests
        success_threshold=2,
        name="test",
    )


@pytest.fixture
def breaker(config):
    """Create circuit breaker for testing."""
    return CircuitBreaker(config)


# ============== Tests ==============

class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.name == "default"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=60.0,
            success_threshold=3,
            name="custom",
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.name == "custom"


class TestCircuitStats:
    """Tests for CircuitStats."""
    
    def test_default_stats(self):
        """Test default stats values."""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.rejected_calls == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0


class TestCircuitBreakerOpen:
    """Tests for CircuitBreakerOpen exception."""
    
    def test_exception_message(self):
        """Test exception message includes name and recovery time."""
        exc = CircuitBreakerOpen("test_service", 30.0)
        assert "test_service" in str(exc)
        assert "30.0" in str(exc)
        assert exc.name == "test_service"
        assert exc.recovery_time == 30.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        breaker = CircuitBreaker()
        assert breaker.config.name == "default"
        assert breaker.state == CircuitState.CLOSED
    
    def test_init_custom_config(self, config):
        """Test initialization with custom config."""
        breaker = CircuitBreaker(config)
        assert breaker.config.name == "test"
        assert breaker.state == CircuitState.CLOSED
    
    def test_state_properties(self, breaker):
        """Test state property methods."""
        assert breaker.is_closed is True
        assert breaker.is_open is False
        assert breaker.is_half_open is False
    
    @pytest.mark.asyncio
    async def test_successful_call_updates_stats(self, breaker):
        """Test that successful calls update statistics."""
        await breaker._record_success()
        
        assert breaker.stats.total_calls == 1
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.consecutive_successes == 1
    
    @pytest.mark.asyncio
    async def test_failed_call_updates_stats(self, breaker):
        """Test that failed calls update statistics."""
        await breaker._record_failure()
        
        assert breaker.stats.total_calls == 1
        assert breaker.stats.failed_calls == 1
        assert breaker.stats.consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, breaker):
        """Test circuit opens after failure threshold."""
        # Record failures up to threshold
        for _ in range(breaker.config.failure_threshold):
            await breaker._record_failure()
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True
    
    @pytest.mark.asyncio
    async def test_circuit_stays_closed_below_threshold(self, breaker):
        """Test circuit stays closed below threshold."""
        # Record failures below threshold
        for _ in range(breaker.config.failure_threshold - 1):
            await breaker._record_failure()
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, breaker):
        """Test success resets consecutive failures."""
        await breaker._record_failure()
        await breaker._record_failure()
        await breaker._record_success()
        
        assert breaker.stats.consecutive_failures == 0
        assert breaker.stats.consecutive_successes == 1
    
    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, breaker):
        """Test circuit rejects calls when open."""
        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            await breaker._record_failure()
        
        assert breaker.state == CircuitState.OPEN
        
        # Try to enter - should raise
        with pytest.raises(CircuitBreakerOpen):
            async with breaker:
                pass
    
    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, breaker):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            await breaker._record_failure()
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)
        
        # Next call should transition to half-open
        try:
            async with breaker:
                pass
        except:
            pass
        
        assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self, config):
        """Test half-open transitions to closed after successes."""
        breaker = CircuitBreaker(config)
        
        # Open the circuit
        for _ in range(config.failure_threshold):
            await breaker._record_failure()
        
        # Wait for timeout
        await asyncio.sleep(config.recovery_timeout + 0.1)
        
        # Manually transition to half-open
        await breaker._transition_to(CircuitState.HALF_OPEN)
        
        # Record successes
        for _ in range(config.success_threshold):
            await breaker._record_success()
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, config):
        """Test half-open transitions to open on failure."""
        breaker = CircuitBreaker(config)
        
        # Manually set to half-open
        await breaker._transition_to(CircuitState.HALF_OPEN)
        
        # Single failure should open circuit
        await breaker._record_failure()
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_time_until_reset(self, breaker):
        """Test time_until_reset calculation."""
        # Should be 0 when closed
        assert breaker._time_until_reset() == 0.0
        
        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            await breaker._record_failure()
        
        # Should be close to recovery_timeout
        time_remaining = breaker._time_until_reset()
        assert 0 < time_remaining <= breaker.config.recovery_timeout
    
    @pytest.mark.asyncio
    async def test_should_attempt_reset(self, breaker):
        """Test should_attempt_reset logic."""
        # Should be False when closed
        assert breaker._should_attempt_reset() is False
        
        # Open the circuit
        for _ in range(breaker.config.failure_threshold):
            await breaker._record_failure()
        
        # Should be False immediately after opening
        assert breaker._should_attempt_reset() is False
        
        # Wait for timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)
        
        # Should be True now
        assert breaker._should_attempt_reset() is True
    
    @pytest.mark.asyncio
    async def test_context_manager_success(self, breaker):
        """Test context manager records success."""
        async with breaker:
            pass  # Success
        
        assert breaker.stats.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_context_manager_failure(self, breaker):
        """Test context manager records failure on exception."""
        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")
        
        assert breaker.stats.failed_calls == 1


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker as decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_success(self, config):
        """Test decorator records successful calls."""
        breaker = CircuitBreaker(config)
        
        @breaker
        async def successful_func():
            return "success"
        
        result = await successful_func()
        
        assert result == "success"
        assert breaker.stats.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_decorator_failure(self, config):
        """Test decorator records failed calls."""
        breaker = CircuitBreaker(config)
        
        @breaker
        async def failing_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            await failing_func()
        
        assert breaker.stats.failed_calls == 1
    
    @pytest.mark.asyncio
    async def test_decorator_opens_circuit(self, config):
        """Test decorator opens circuit after failures."""
        breaker = CircuitBreaker(config)
        
        @breaker
        async def failing_func():
            raise ValueError("test error")
        
        # Trigger enough failures
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await failing_func()
        
        assert breaker.state == CircuitState.OPEN
        
        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpen):
            await failing_func()


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""
    
    @pytest.mark.asyncio
    async def test_stats_accumulate(self, breaker):
        """Test stats accumulate correctly."""
        await breaker._record_success()
        await breaker._record_success()
        await breaker._record_failure()
        
        assert breaker.stats.total_calls == 3
        assert breaker.stats.successful_calls == 2
        assert breaker.stats.failed_calls == 1
    
    @pytest.mark.asyncio
    async def test_rejected_calls_tracked(self, breaker):
        """Test rejected calls are tracked."""
        # Open circuit
        for _ in range(breaker.config.failure_threshold):
            await breaker._record_failure()
        
        # Try calls that get rejected
        for _ in range(5):
            try:
                async with breaker:
                    pass
            except CircuitBreakerOpen:
                pass
        
        assert breaker.stats.rejected_calls == 5
    
    @pytest.mark.asyncio
    async def test_timestamps_recorded(self, breaker):
        """Test success/failure timestamps are recorded."""
        await breaker._record_success()
        assert breaker.stats.last_success_time is not None
        
        await breaker._record_failure()
        assert breaker.stats.last_failure_time is not None

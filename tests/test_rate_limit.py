"""
Tests for rate limiting middleware.
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch

from app.middleware.rate_limit import (
    SlidingWindowRateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    configure_rate_limits,
    api_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_size == 10
        assert config.endpoint_limits == {}
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            endpoint_limits={"/api/v1/chat": {"minute": 20}},
        )
        
        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert "/api/v1/chat" in config.endpoint_limits


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""
    
    @pytest.fixture
    def limiter(self):
        """Create a rate limiter for testing."""
        config = RateLimitConfig(
            requests_per_minute=5,
            requests_per_hour=100,
            requests_per_day=1000,
        )
        return SlidingWindowRateLimiter(config)
    
    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = MagicMock()
        request.client.host = "192.168.1.1"
        request.headers = {}
        request.url.path = "/api/v1/test"
        return request
    
    def test_client_key_from_ip(self, limiter, mock_request):
        """Test client key generation from IP."""
        key = limiter._get_client_key(mock_request)
        assert key == "ip:192.168.1.1"
    
    def test_client_key_from_forwarded(self, limiter, mock_request):
        """Test client key generation from X-Forwarded-For."""
        mock_request.headers = {"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}
        key = limiter._get_client_key(mock_request)
        assert key == "ip:10.0.0.1"
    
    def test_client_key_from_user_id(self, limiter, mock_request):
        """Test client key generation from user ID."""
        key = limiter._get_client_key(mock_request, user_id="user-123")
        assert key == "user:user-123"
    
    @pytest.mark.asyncio
    async def test_allows_within_limit(self, limiter, mock_request):
        """Test that requests within limit are allowed."""
        for i in range(5):
            allowed, info = await limiter.check_rate_limit(mock_request)
            assert allowed, f"Request {i+1} should be allowed"
            # After request i, remaining should be limit - (i+1)
            # But info is calculated before increment, so it shows limit - i
            # After increment, actual remaining is limit - (i+1)
            # The info shows remaining BEFORE this request was counted
            expected_remaining = 5 - i  # Shows remaining before this request
            assert info["minute"]["remaining"] == expected_remaining, f"Request {i+1}: expected {expected_remaining}, got {info['minute']['remaining']}"
    
    @pytest.mark.asyncio
    async def test_blocks_over_limit(self, limiter, mock_request):
        """Test that requests over limit are blocked."""
        # Use up the limit
        for _ in range(5):
            await limiter.check_rate_limit(mock_request)
        
        # Next request should be blocked
        allowed, info = await limiter.check_rate_limit(mock_request)
        assert not allowed
        assert info.get("exceeded") == "minute"
        assert info["minute"]["remaining"] == 0
    
    @pytest.mark.asyncio
    async def test_different_clients_independent(self, limiter):
        """Test that different clients have independent limits."""
        request1 = MagicMock()
        request1.client.host = "192.168.1.1"
        request1.headers = {}
        request1.url.path = "/api/v1/test"
        
        request2 = MagicMock()
        request2.client.host = "192.168.1.2"
        request2.headers = {}
        request2.url.path = "/api/v1/test"
        
        # Use up client 1's limit
        for _ in range(5):
            await limiter.check_rate_limit(request1)
        
        # Client 2 should still be allowed
        allowed, _ = await limiter.check_rate_limit(request2)
        assert allowed
    
    @pytest.mark.asyncio
    async def test_limit_resets_after_window(self, limiter, mock_request):
        """Test that limit resets after time window."""
        # Use up limit
        for _ in range(5):
            await limiter.check_rate_limit(mock_request)
        
        # Should be blocked
        allowed, _ = await limiter.check_rate_limit(mock_request)
        assert not allowed
        
        # Manually reset the window
        client_key = limiter._get_client_key(mock_request)
        limiter._entries[client_key].minute_reset = time.time() - 1
        limiter._entries[client_key].minute_count = 0
        
        # Should be allowed again
        allowed, _ = await limiter.check_rate_limit(mock_request)
        assert allowed
    
    @pytest.mark.asyncio
    async def test_get_remaining(self, limiter, mock_request):
        """Test get_remaining returns correct info."""
        # Make a few requests
        for _ in range(3):
            await limiter.check_rate_limit(mock_request)
        
        remaining = await limiter.get_remaining(mock_request)
        
        assert remaining["minute"]["limit"] == 5
        assert remaining["minute"]["remaining"] == 2
    
    def test_clear_specific_client(self, limiter, mock_request):
        """Test clearing specific client."""
        asyncio.run(limiter.check_rate_limit(mock_request))
        
        client_key = limiter._get_client_key(mock_request)
        assert client_key in limiter._entries
        
        limiter.clear(client_key)
        assert client_key not in limiter._entries
    
    def test_clear_all(self, limiter, mock_request):
        """Test clearing all entries."""
        asyncio.run(limiter.check_rate_limit(mock_request))
        assert len(limiter._entries) > 0
        
        limiter.clear()
        assert len(limiter._entries) == 0


class TestConfigureRateLimits:
    """Tests for configure_rate_limits function."""
    
    def test_default_limits(self):
        """Test default rate limits."""
        config = configure_rate_limits()
        
        assert "/api/v1/chat" in config.endpoint_limits
        assert "/api/v1/retrieval" in config.endpoint_limits
        assert "/api/v1/ingest" in config.endpoint_limits
        assert "/api/v1/auth" in config.endpoint_limits
    
    def test_custom_limits(self):
        """Test custom rate limits."""
        config = configure_rate_limits(
            chat_per_minute=10,
            retrieval_per_minute=100,
        )
        
        assert config.endpoint_limits["/api/v1/chat"]["minute"] == 10
        assert config.endpoint_limits["/api/v1/retrieval"]["minute"] == 100


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app."""
        return MagicMock()
    
    @pytest.fixture
    def middleware(self, mock_app):
        """Create middleware for testing."""
        limiter = SlidingWindowRateLimiter(
            RateLimitConfig(requests_per_minute=5)
        )
        return RateLimitMiddleware(mock_app, rate_limiter=limiter)
    
    def test_excluded_paths(self, middleware):
        """Test that excluded paths are correct."""
        assert "/health" in middleware.EXCLUDED_PATHS
        assert "/api/v1/health" in middleware.EXCLUDED_PATHS
        assert "/docs" in middleware.EXCLUDED_PATHS
    
    @pytest.mark.asyncio
    async def test_excludes_health_endpoint(self, middleware):
        """Test that health endpoint is not rate limited."""
        request = MagicMock()
        request.url.path = "/health"
        
        call_next = AsyncMock(return_value=MagicMock())
        
        with patch.object(middleware.rate_limiter, 'check_rate_limit') as mock_check:
            await middleware.dispatch(request, call_next)
            mock_check.assert_not_called()


class AsyncMock(MagicMock):
    """Mock for async functions."""
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

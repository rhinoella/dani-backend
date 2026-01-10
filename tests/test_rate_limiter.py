"""
Tests for Rate Limiter module.
"""

import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from app.cache.rate_limiter import RateLimiter, RateLimitResult, rate_limiter


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""
    
    def test_create_result_allowed(self):
        """Test creating an allowed result."""
        result = RateLimitResult(
            allowed=True,
            remaining=10,
            reset_at=time.time() + 60,
            limit=20
        )
        assert result.allowed is True
        assert result.remaining == 10
        assert result.limit == 20
        assert result.retry_after is None
    
    def test_create_result_denied(self):
        """Test creating a denied result."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=time.time() + 30,
            limit=20,
            retry_after=30
        )
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    @pytest.fixture
    def limiter(self):
        """Create a RateLimiter instance."""
        return RateLimiter(
            prefix="test_ratelimit",
            per_minute=10,
            per_day=100
        )
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        # Default mock for pipeline
        pipe = AsyncMock()
        pipe.execute = AsyncMock(return_value=[1, True, 1, True])
        client.pipeline.return_value = pipe
        return client
    
    def test_minute_key_format(self, limiter):
        """Test minute key generation."""
        key = limiter._minute_key("user123")
        assert key.startswith("test_ratelimit:user123:min:")
        # Key should contain current minute
        minute = int(time.time() // 60)
        assert str(minute) in key
    
    def test_day_key_format(self, limiter):
        """Test day key generation."""
        key = limiter._day_key("user123")
        assert key.startswith("test_ratelimit:user123:day:")
        # Key should contain current day
        day = int(time.time() // 86400)
        assert str(day) in key
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_disabled(self, limiter):
        """Test rate limit check when disabled."""
        with patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = False
            
            result = await limiter.check_rate_limit("user123")
            
            assert result.allowed is True
            assert result.remaining == limiter.per_minute
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, limiter, mock_client):
        """Test rate limit check when within limits."""
        mock_client.get.side_effect = ["5", "50"]  # minute_count, day_count
        mock_client.ttl.return_value = 30
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client), \
             patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            
            result = await limiter.check_rate_limit("user123")
            
            assert result.allowed is True
            assert result.remaining == 5  # min(10-5, 100-50)
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_minute_exceeded(self, limiter, mock_client):
        """Test rate limit check when minute limit exceeded."""
        mock_client.get.side_effect = ["10", None]  # minute_count at limit
        mock_client.ttl.return_value = 30
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client), \
             patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            
            result = await limiter.check_rate_limit("user123")
            
            assert result.allowed is False
            assert result.remaining == 0
            assert result.retry_after == 30
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_day_exceeded(self, limiter, mock_client):
        """Test rate limit check when day limit exceeded."""
        mock_client.get.side_effect = ["5", "100"]  # day_count at limit
        mock_client.ttl.return_value = 3600
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client), \
             patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            
            result = await limiter.check_rate_limit("user123")
            
            assert result.allowed is False
            assert result.limit == limiter.per_day
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_error(self, limiter, mock_client):
        """Test rate limit check fails open on Redis error."""
        mock_client.get.side_effect = Exception("Redis unavailable")
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client), \
             patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            
            result = await limiter.check_rate_limit("user123")
            
            # Should fail open - allow the request
            assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_increment_disabled(self, limiter):
        """Test increment when rate limiting disabled."""
        with patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = False
            
            result = await limiter.increment("user123")
            
            assert result == (0, 0)
    
    @pytest.mark.asyncio
    async def test_increment_success(self, limiter):
        """Test successful increment."""
        mock_client = MagicMock()  # client is sync, only execute is async
        pipe = MagicMock()
        pipe.incr = MagicMock()
        pipe.expire = MagicMock()
        pipe.execute = AsyncMock(return_value=[5, True, 25, True])
        mock_client.pipeline.return_value = pipe
        
        async def mock_get_client():
            return mock_client
        
        with patch("app.cache.rate_limiter.get_redis_client", new=mock_get_client), \
             patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            
            result = await limiter.increment("user123")
            
            assert result == (5, 25)  # (minute_count, day_count)
            pipe.incr.assert_called()
            pipe.expire.assert_called()
    
    @pytest.mark.asyncio
    async def test_increment_error(self, limiter, mock_client):
        """Test increment handles errors gracefully."""
        mock_client.pipeline.side_effect = Exception("Redis error")
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client), \
             patch("app.cache.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_ENABLED = True
            
            result = await limiter.increment("user123")
            
            assert result == (0, 0)
    
    @pytest.mark.asyncio
    async def test_get_usage_success(self, limiter):
        """Test getting usage statistics."""
        mock_client = MagicMock()  # client is sync, only execute is async
        pipe = MagicMock()
        pipe.get = MagicMock()
        pipe.ttl = MagicMock()
        pipe.execute = AsyncMock(return_value=["5", 30, "50", 3600])
        mock_client.pipeline.return_value = pipe
        
        async def mock_get_client():
            return mock_client
        
        with patch("app.cache.rate_limiter.get_redis_client", new=mock_get_client):
            result = await limiter.get_usage("user123")
            
            assert result["minute"]["used"] == 5
            assert result["minute"]["limit"] == 10
            assert result["minute"]["reset_in"] == 30
            assert result["day"]["used"] == 50
            assert result["day"]["limit"] == 100
            assert result["day"]["reset_in"] == 3600
    
    @pytest.mark.asyncio
    async def test_get_usage_no_data(self, limiter):
        """Test getting usage when no data exists."""
        mock_client = MagicMock()  # client is sync, only execute is async
        pipe = MagicMock()
        pipe.get = MagicMock()
        pipe.ttl = MagicMock()
        pipe.execute = AsyncMock(return_value=[None, -2, None, -2])
        mock_client.pipeline.return_value = pipe
        
        async def mock_get_client():
            return mock_client
        
        with patch("app.cache.rate_limiter.get_redis_client", new=mock_get_client):
            result = await limiter.get_usage("new_user")
            
            assert result["minute"]["used"] == 0
            assert result["day"]["used"] == 0
    
    @pytest.mark.asyncio
    async def test_get_usage_error(self, limiter, mock_client):
        """Test get_usage handles errors gracefully."""
        mock_client.pipeline.side_effect = Exception("Redis error")
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client):
            result = await limiter.get_usage("user123")
            
            # Should return defaults
            assert result["minute"]["used"] == 0
            assert result["day"]["used"] == 0
    
    @pytest.mark.asyncio
    async def test_reset_success(self, limiter, mock_client):
        """Test resetting rate limits for a user."""
        mock_client.scan.side_effect = [
            (1, ["test_ratelimit:user123:min:123"]),
            (0, ["test_ratelimit:user123:day:456"]),
        ]
        mock_client.delete = AsyncMock()
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client):
            result = await limiter.reset("user123")
            
            assert result is True
            assert mock_client.delete.call_count == 2
    
    @pytest.mark.asyncio
    async def test_reset_no_keys(self, limiter, mock_client):
        """Test reset when no keys found."""
        mock_client.scan.return_value = (0, [])
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client):
            result = await limiter.reset("new_user")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_reset_error(self, limiter, mock_client):
        """Test reset handles errors gracefully."""
        mock_client.scan.side_effect = Exception("Redis error")
        
        with patch("app.cache.rate_limiter.get_redis_client", return_value=mock_client):
            result = await limiter.reset("user123")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_status_alias(self, limiter):
        """Test get_status is alias for get_usage."""
        mock_client = MagicMock()  # client is sync, only execute is async
        pipe = MagicMock()
        pipe.get = MagicMock()
        pipe.ttl = MagicMock()
        pipe.execute = AsyncMock(return_value=["3", 45, "30", 7200])
        mock_client.pipeline.return_value = pipe
        
        async def mock_get_client():
            return mock_client
        
        with patch("app.cache.rate_limiter.get_redis_client", new=mock_get_client):
            result = await limiter.get_status("user123")
            
            assert result["minute"]["used"] == 3
            assert result["day"]["used"] == 30


class TestGlobalRateLimiter:
    """Tests for global rate_limiter instance."""
    
    def test_global_instance_exists(self):
        """Test that global rate_limiter instance exists."""
        assert rate_limiter is not None
        assert isinstance(rate_limiter, RateLimiter)
    
    def test_global_instance_configuration(self):
        """Test global rate_limiter has expected configuration."""
        assert rate_limiter.prefix == "ratelimit"

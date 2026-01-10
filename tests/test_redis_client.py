"""
Tests for Redis client module.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.cache.redis_client import (
    get_redis_pool,
    get_redis_client,
    init_redis,
    close_redis,
    check_health,
    RedisCache,
    RedisList,
)


class TestRedisConnection:
    """Tests for Redis connection functions."""
    
    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global variables before each test."""
        import app.cache.redis_client as redis_module
        redis_module._pool = None
        redis_module._client = None
        yield
        redis_module._pool = None
        redis_module._client = None
    
    @pytest.mark.asyncio
    async def test_get_redis_pool_creates_pool(self):
        """Test that get_redis_pool creates a connection pool."""
        with patch("app.cache.redis_client.ConnectionPool") as MockPool:
            mock_pool = MagicMock()
            MockPool.from_url.return_value = mock_pool
            
            pool = await get_redis_pool()
            
            assert pool == mock_pool
            MockPool.from_url.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_redis_pool_reuses_pool(self):
        """Test that subsequent calls reuse the same pool."""
        with patch("app.cache.redis_client.ConnectionPool") as MockPool:
            mock_pool = MagicMock()
            MockPool.from_url.return_value = mock_pool
            
            pool1 = await get_redis_pool()
            pool2 = await get_redis_pool()
            
            assert pool1 is pool2
            # Should only be called once
            assert MockPool.from_url.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_redis_client_creates_client(self):
        """Test that get_redis_client creates a Redis client."""
        with patch("app.cache.redis_client.get_redis_pool") as mock_get_pool, \
             patch("app.cache.redis_client.redis.Redis") as MockRedis:
            mock_pool = MagicMock()
            mock_get_pool.return_value = mock_pool
            mock_client = AsyncMock()
            MockRedis.return_value = mock_client
            
            client = await get_redis_client()
            
            assert client == mock_client
            MockRedis.assert_called_once_with(connection_pool=mock_pool)
    
    @pytest.mark.asyncio
    async def test_init_redis_success(self):
        """Test successful Redis initialization."""
        with patch("app.cache.redis_client.get_redis_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_get_client.return_value = mock_client
            
            await init_redis()
            
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_init_redis_failure(self):
        """Test Redis initialization failure raises exception."""
        with patch("app.cache.redis_client.get_redis_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_get_client.return_value = mock_client
            
            with pytest.raises(Exception, match="Connection refused"):
                await init_redis()
    
    @pytest.mark.asyncio
    async def test_close_redis(self):
        """Test closing Redis connections."""
        import app.cache.redis_client as redis_module
        
        mock_client = AsyncMock()
        mock_pool = AsyncMock()
        redis_module._client = mock_client
        redis_module._pool = mock_pool
        
        await close_redis()
        
        mock_client.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert redis_module._client is None
        assert redis_module._pool is None
    
    @pytest.mark.asyncio
    async def test_close_redis_when_none(self):
        """Test closing Redis when not initialized."""
        import app.cache.redis_client as redis_module
        redis_module._client = None
        redis_module._pool = None
        
        # Should not raise
        await close_redis()
    
    @pytest.mark.asyncio
    async def test_check_health_healthy(self):
        """Test health check when Redis is healthy."""
        with patch("app.cache.redis_client.get_redis_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.info.return_value = {"redis_version": "7.0.0"}
            mock_get_client.return_value = mock_client
            
            result = await check_health()
            
            assert result["status"] == "healthy"
            assert result["database"] == "redis"
            assert result["version"] == "7.0.0"
    
    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self):
        """Test health check when Redis is unhealthy."""
        with patch("app.cache.redis_client.get_redis_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Connection timeout")
            
            result = await check_health()
            
            assert result["status"] == "unhealthy"
            assert result["database"] == "redis"
            assert "Connection timeout" in result["error"]


class TestRedisCache:
    """Tests for RedisCache class."""
    
    @pytest.fixture
    def cache(self):
        """Create a RedisCache instance."""
        return RedisCache(prefix="test", default_ttl=300)
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Redis client."""
        return AsyncMock()
    
    def test_key_with_prefix(self, cache):
        """Test key generation with prefix."""
        assert cache._key("mykey") == "test:mykey"
    
    def test_key_without_prefix(self):
        """Test key generation without prefix."""
        cache = RedisCache(prefix="", default_ttl=300)
        assert cache._key("mykey") == "mykey"
    
    @pytest.mark.asyncio
    async def test_get_success(self, cache, mock_client):
        """Test successful get operation."""
        mock_client.get.return_value = json.dumps({"key": "value"})
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get("mykey")
        
        assert result == {"key": "value"}
        mock_client.get.assert_called_once_with("test:mykey")
    
    @pytest.mark.asyncio
    async def test_get_not_found(self, cache, mock_client):
        """Test get when key not found."""
        mock_client.get.return_value = None
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get("missing")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_error(self, cache, mock_client):
        """Test get handles errors gracefully."""
        mock_client.get.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.get("mykey")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_success(self, cache, mock_client):
        """Test successful set operation."""
        mock_client.set = AsyncMock()
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.set("mykey", {"data": 123})
        
        assert result is True
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "test:mykey"
        assert json.loads(call_args[0][1]) == {"data": 123}
        assert call_args[1]["ex"] == 300  # default TTL
    
    @pytest.mark.asyncio
    async def test_set_custom_ttl(self, cache, mock_client):
        """Test set with custom TTL."""
        mock_client.set = AsyncMock()
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.set("mykey", "value", ttl=600)
        
        assert result is True
        call_args = mock_client.set.call_args
        assert call_args[1]["ex"] == 600
    
    @pytest.mark.asyncio
    async def test_set_error(self, cache, mock_client):
        """Test set handles errors gracefully."""
        mock_client.set.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.set("mykey", "value")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_success(self, cache, mock_client):
        """Test successful delete operation."""
        mock_client.delete.return_value = 1
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.delete("mykey")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_not_found(self, cache, mock_client):
        """Test delete when key not found."""
        mock_client.delete.return_value = 0
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.delete("missing")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_error(self, cache, mock_client):
        """Test delete handles errors gracefully."""
        mock_client.delete.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.delete("mykey")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_true(self, cache, mock_client):
        """Test exists returns true when key exists."""
        mock_client.exists.return_value = 1
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.exists("mykey")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self, cache, mock_client):
        """Test exists returns false when key doesn't exist."""
        mock_client.exists.return_value = 0
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.exists("missing")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_exists_error(self, cache, mock_client):
        """Test exists handles errors gracefully."""
        mock_client.exists.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.exists("mykey")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_expire_success(self, cache, mock_client):
        """Test setting expiration on a key."""
        mock_client.expire.return_value = True
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.expire("mykey", 600)
        
        assert result is True
        mock_client.expire.assert_called_once_with("test:mykey", 600)
    
    @pytest.mark.asyncio
    async def test_expire_error(self, cache, mock_client):
        """Test expire handles errors gracefully."""
        mock_client.expire.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.expire("mykey", 600)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_ttl_success(self, cache, mock_client):
        """Test getting TTL for a key."""
        mock_client.ttl.return_value = 120
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.ttl("mykey")
        
        assert result == 120
    
    @pytest.mark.asyncio
    async def test_ttl_no_expiry(self, cache, mock_client):
        """Test TTL when key has no expiration."""
        mock_client.ttl.return_value = -1
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.ttl("mykey")
        
        assert result == -1
    
    @pytest.mark.asyncio
    async def test_ttl_error(self, cache, mock_client):
        """Test TTL handles errors gracefully."""
        mock_client.ttl.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.ttl("mykey")
        
        assert result == -2
    
    @pytest.mark.asyncio
    async def test_clear_prefix_success(self, cache, mock_client):
        """Test clearing all keys with prefix."""
        mock_client.scan.side_effect = [
            (1, ["test:key1", "test:key2"]),
            (0, ["test:key3"]),
        ]
        mock_client.delete.side_effect = [2, 1]
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.clear_prefix()
        
        assert result == 3  # Total deleted
    
    @pytest.mark.asyncio
    async def test_clear_prefix_empty(self, cache, mock_client):
        """Test clearing when no keys found."""
        mock_client.scan.return_value = (0, [])
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.clear_prefix()
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_clear_prefix_error(self, cache, mock_client):
        """Test clear_prefix handles errors gracefully."""
        mock_client.scan.side_effect = Exception("Redis error")
        
        with patch.object(cache, "_get_client", return_value=mock_client):
            result = await cache.clear_prefix()
        
        assert result == 0


class TestRedisList:
    """Tests for RedisList class."""
    
    @pytest.fixture
    def redis_list(self):
        """Create a RedisList instance."""
        return RedisList(prefix="list", default_ttl=300)
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Redis client."""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_lpush_success(self, redis_list, mock_client):
        """Test pushing values to front of list."""
        mock_client.lpush.return_value = 3
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.lpush("mylist", {"a": 1}, {"b": 2})
        
        assert result == 3
        mock_client.lpush.assert_called_once()
        call_args = mock_client.lpush.call_args
        assert call_args[0][0] == "list:mylist"
    
    @pytest.mark.asyncio
    async def test_lpush_error(self, redis_list, mock_client):
        """Test lpush handles errors gracefully."""
        mock_client.lpush.side_effect = Exception("Redis error")
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.lpush("mylist", "value")
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_rpush_success(self, redis_list, mock_client):
        """Test pushing values to end of list."""
        mock_client.rpush.return_value = 2
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.rpush("mylist", "val1", "val2")
        
        assert result == 2
    
    @pytest.mark.asyncio
    async def test_rpush_error(self, redis_list, mock_client):
        """Test rpush handles errors gracefully."""
        mock_client.rpush.side_effect = Exception("Redis error")
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.rpush("mylist", "value")
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_lrange_success(self, redis_list, mock_client):
        """Test getting range of values from list."""
        mock_client.lrange.return_value = [
            json.dumps({"a": 1}),
            json.dumps({"b": 2}),
        ]
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.lrange("mylist", 0, -1)
        
        assert result == [{"a": 1}, {"b": 2}]
    
    @pytest.mark.asyncio
    async def test_lrange_empty(self, redis_list, mock_client):
        """Test lrange on empty list."""
        mock_client.lrange.return_value = []
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.lrange("emptylist")
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_lrange_error(self, redis_list, mock_client):
        """Test lrange handles errors gracefully."""
        mock_client.lrange.side_effect = Exception("Redis error")
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.lrange("mylist")
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_llen_success(self, redis_list, mock_client):
        """Test getting list length."""
        mock_client.llen.return_value = 5
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.llen("mylist")
        
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_llen_empty(self, redis_list, mock_client):
        """Test llen on non-existent list."""
        mock_client.llen.return_value = 0
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.llen("missing")
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_llen_error(self, redis_list, mock_client):
        """Test llen handles errors gracefully."""
        mock_client.llen.side_effect = Exception("Redis error")
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.llen("mylist")
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_ltrim_success(self, redis_list, mock_client):
        """Test trimming list."""
        mock_client.ltrim = AsyncMock()
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.ltrim("mylist", 0, 9)
        
        assert result is True
        mock_client.ltrim.assert_called_once_with("list:mylist", 0, 9)
    
    @pytest.mark.asyncio
    async def test_ltrim_error(self, redis_list, mock_client):
        """Test ltrim handles errors gracefully."""
        mock_client.ltrim.side_effect = Exception("Redis error")
        
        with patch.object(redis_list, "_get_client", return_value=mock_client):
            result = await redis_list.ltrim("mylist", 0, 9)
        
        assert result is False

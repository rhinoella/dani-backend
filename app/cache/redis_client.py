"""
Redis client for DANI Engine.

Provides async Redis connection management and utilities.
"""

from __future__ import annotations

import logging
import json
from typing import Any, Optional, List
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[ConnectionPool] = None
_client: Optional[redis.Redis] = None


async def get_redis_pool() -> ConnectionPool:
    """Get or create the Redis connection pool."""
    global _pool
    
    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            decode_responses=True,  # Return strings instead of bytes
        )
    
    return _pool


async def get_redis_client() -> redis.Redis:
    """Get the Redis client instance."""
    global _client
    
    if _client is None:
        pool = await get_redis_pool()
        _client = redis.Redis(connection_pool=pool)
    
    return _client


async def init_redis() -> None:
    """
    Initialize Redis connection and verify connectivity.
    Called on application startup.
    """
    logger.info("Initializing Redis connection...")
    
    try:
        client = await get_redis_client()
        await client.ping()
        logger.info("✅ Redis connection established successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise


async def close_redis() -> None:
    """
    Close Redis connections gracefully.
    Called on application shutdown.
    """
    global _client, _pool
    
    logger.info("Closing Redis connections...")
    
    if _client is not None:
        await _client.close()
        _client = None
    
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
    
    logger.info("✅ Redis connections closed")


async def check_health() -> dict:
    """Check Redis health for health endpoint."""
    try:
        client = await get_redis_client()
        info = await client.info("server")
        return {
            "status": "healthy",
            "database": "redis",
            "version": info.get("redis_version"),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "redis",
            "error": str(e),
        }


class RedisCache:
    """
    Base class for Redis caching operations.
    
    Provides common methods for get, set, delete with JSON serialization.
    """
    
    def __init__(self, prefix: str = "", default_ttl: int = 3600):
        """
        Initialize cache with key prefix and default TTL.
        
        Args:
            prefix: Prefix for all keys (e.g., "conv", "user")
            default_ttl: Default time-to-live in seconds
        """
        self.prefix = prefix
        self.default_ttl = default_ttl
    
    def _key(self, key: str) -> str:
        """Build full key with prefix."""
        if self.prefix:
            return f"{self.prefix}:{key}"
        return key
    
    async def _get_client(self) -> redis.Redis:
        """Get Redis client."""
        return await get_redis_client()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            client = await self._get_client()
            value = await client.get(self._key(key))
            
            if value is not None:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.warning(f"Redis GET error for {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses default if not specified)
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            serialized = json.dumps(value, default=str)
            
            await client.set(
                self._key(key),
                serialized,
                ex=ttl or self.default_ttl,
            )
            return True
            
        except Exception as e:
            logger.warning(f"Redis SET error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        try:
            client = await self._get_client()
            result = await client.delete(self._key(key))
            return result > 0
            
        except Exception as e:
            logger.warning(f"Redis DELETE error for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            client = await self._get_client()
            return await client.exists(self._key(key)) > 0
        except Exception as e:
            logger.warning(f"Redis EXISTS error for {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on existing key."""
        try:
            client = await self._get_client()
            return await client.expire(self._key(key), ttl)
        except Exception as e:
            logger.warning(f"Redis EXPIRE error for {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key (-1 if no expiry, -2 if not exists)."""
        try:
            client = await self._get_client()
            return await client.ttl(self._key(key))
        except Exception as e:
            logger.warning(f"Redis TTL error for {key}: {e}")
            return -2
    
    async def clear_prefix(self) -> int:
        """
        Delete all keys with this cache's prefix.
        Use with caution!
        
        Returns:
            Number of keys deleted
        """
        try:
            client = await self._get_client()
            pattern = f"{self.prefix}:*" if self.prefix else "*"
            
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break
            
            return deleted
            
        except Exception as e:
            logger.warning(f"Redis CLEAR error for prefix {self.prefix}: {e}")
            return 0


class RedisList(RedisCache):
    """
    Redis List operations for storing ordered items.
    """
    
    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to the left (front) of list."""
        try:
            client = await self._get_client()
            serialized = [json.dumps(v, default=str) for v in values]
            return await client.lpush(self._key(key), *serialized)
        except Exception as e:
            logger.warning(f"Redis LPUSH error for {key}: {e}")
            return 0
    
    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to the right (end) of list."""
        try:
            client = await self._get_client()
            serialized = [json.dumps(v, default=str) for v in values]
            return await client.rpush(self._key(key), *serialized)
        except Exception as e:
            logger.warning(f"Redis RPUSH error for {key}: {e}")
            return 0
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of values from list."""
        try:
            client = await self._get_client()
            values = await client.lrange(self._key(key), start, end)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.warning(f"Redis LRANGE error for {key}: {e}")
            return []
    
    async def llen(self, key: str) -> int:
        """Get length of list."""
        try:
            client = await self._get_client()
            return await client.llen(self._key(key))
        except Exception as e:
            logger.warning(f"Redis LLEN error for {key}: {e}")
            return 0
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        try:
            client = await self._get_client()
            await client.ltrim(self._key(key), start, end)
            return True
        except Exception as e:
            logger.warning(f"Redis LTRIM error for {key}: {e}")
            return False

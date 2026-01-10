"""
Rate limiter using Redis.

Implements sliding window rate limiting for API endpoints.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple
from dataclasses import dataclass

from app.cache.redis_client import get_redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    limit: int
    retry_after: Optional[int] = None


class RateLimiter:
    """
    Sliding window rate limiter using Redis.
    
    Supports multiple rate limits (per-minute, per-day).
    """
    
    def __init__(
        self,
        prefix: str = "ratelimit",
        per_minute: int = 20,
        per_day: int = 500,
    ):
        """
        Initialize rate limiter.
        
        Args:
            prefix: Redis key prefix
            per_minute: Maximum requests per minute
            per_day: Maximum requests per day
        """
        self.prefix = prefix
        self.per_minute = per_minute
        self.per_day = per_day
    
    def _minute_key(self, user_id: str) -> str:
        """Generate key for per-minute rate limit."""
        minute = int(time.time() // 60)
        return f"{self.prefix}:{user_id}:min:{minute}"
    
    def _day_key(self, user_id: str) -> str:
        """Generate key for per-day rate limit."""
        day = int(time.time() // 86400)
        return f"{self.prefix}:{user_id}:day:{day}"
    
    async def check_rate_limit(
        self,
        user_id: str,
    ) -> RateLimitResult:
        """
        Check if user is within rate limits.
        
        Does not increment counters - use increment() for that.
        
        Args:
            user_id: User identifier
            
        Returns:
            RateLimitResult with allowed status and limits
        """
        if not settings.RATE_LIMIT_ENABLED:
            return RateLimitResult(
                allowed=True,
                remaining=self.per_minute,
                reset_at=time.time() + 60,
                limit=self.per_minute,
            )
        
        try:
            client = await get_redis_client()
            
            # Check per-minute limit
            minute_key = self._minute_key(user_id)
            minute_count = await client.get(minute_key)
            minute_count = int(minute_count) if minute_count else 0
            
            if minute_count >= self.per_minute:
                ttl = await client.ttl(minute_key)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=time.time() + max(ttl, 1),
                    limit=self.per_minute,
                    retry_after=max(ttl, 1),
                )
            
            # Check per-day limit
            day_key = self._day_key(user_id)
            day_count = await client.get(day_key)
            day_count = int(day_count) if day_count else 0
            
            if day_count >= self.per_day:
                ttl = await client.ttl(day_key)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=time.time() + max(ttl, 1),
                    limit=self.per_day,
                    retry_after=max(ttl, 1),
                )
            
            # Calculate remaining (use the more restrictive)
            minute_remaining = self.per_minute - minute_count
            day_remaining = self.per_day - day_count
            
            return RateLimitResult(
                allowed=True,
                remaining=min(minute_remaining, day_remaining),
                reset_at=time.time() + 60,
                limit=self.per_minute,
            )
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open - allow request if Redis is unavailable
            return RateLimitResult(
                allowed=True,
                remaining=self.per_minute,
                reset_at=time.time() + 60,
                limit=self.per_minute,
            )
    
    async def increment(
        self,
        user_id: str,
    ) -> Tuple[int, int]:
        """
        Increment rate limit counters.
        
        Call this after successfully processing a request.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (minute_count, day_count)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return (0, 0)
        
        try:
            client = await get_redis_client()
            
            # Increment per-minute counter
            minute_key = self._minute_key(user_id)
            pipe = client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            
            # Increment per-day counter
            day_key = self._day_key(user_id)
            pipe.incr(day_key)
            pipe.expire(day_key, 86400)
            
            results = await pipe.execute()
            
            minute_count = results[0]
            day_count = results[2]
            
            return (minute_count, day_count)
            
        except Exception as e:
            logger.error(f"Rate limit increment error: {e}")
            return (0, 0)
    
    async def get_usage(
        self,
        user_id: str,
    ) -> dict:
        """
        Get current usage for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with minute and day usage
        """
        try:
            client = await get_redis_client()
            
            minute_key = self._minute_key(user_id)
            day_key = self._day_key(user_id)
            
            pipe = client.pipeline()
            pipe.get(minute_key)
            pipe.ttl(minute_key)
            pipe.get(day_key)
            pipe.ttl(day_key)
            
            results = await pipe.execute()
            
            return {
                "minute": {
                    "used": int(results[0]) if results[0] else 0,
                    "limit": self.per_minute,
                    "reset_in": max(results[1], 0) if results[1] and results[1] > 0 else 60,
                },
                "day": {
                    "used": int(results[2]) if results[2] else 0,
                    "limit": self.per_day,
                    "reset_in": max(results[3], 0) if results[3] and results[3] > 0 else 86400,
                },
            }
            
        except Exception as e:
            logger.error(f"Rate limit usage error: {e}")
            return {
                "minute": {"used": 0, "limit": self.per_minute, "reset_in": 60},
                "day": {"used": 0, "limit": self.per_day, "reset_in": 86400},
            }
    
    async def reset(self, user_id: str) -> bool:
        """
        Reset rate limits for a user.
        
        For admin use only.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if reset successful
        """
        try:
            client = await get_redis_client()
            
            # Delete all rate limit keys for this user
            pattern = f"{self.prefix}:{user_id}:*"
            cursor = 0
            
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit reset error: {e}")
            return False

    async def get_status(self, user_id: str) -> dict:
        """Alias for get_usage for backward compatibility."""
        return await self.get_usage(user_id)


# Global rate limiter instance
rate_limiter = RateLimiter(
    prefix="ratelimit",
    per_minute=settings.RATE_LIMIT_PER_MINUTE,
    per_day=settings.RATE_LIMIT_PER_DAY,
)

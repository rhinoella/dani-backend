"""
API Rate Limiting Middleware for DANI Engine.

Provides:
- IP-based rate limiting
- User-based rate limiting (when authenticated)
- Sliding window algorithm
- Configurable limits per endpoint
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Callable, Any
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

from app.core.config import settings
from app.core.exceptions import RateLimitError, ErrorResponse, ErrorDetail

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10  # Allow burst of requests
    
    # Per-endpoint overrides (path -> config)
    endpoint_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class RateLimitEntry:
    """Tracks rate limit state for a client."""
    minute_count: int = 0
    minute_reset: float = 0.0
    hour_count: int = 0
    hour_reset: float = 0.0
    day_count: int = 0
    day_reset: float = 0.0
    last_request: float = 0.0


class SlidingWindowRateLimiter:
    """
    In-memory sliding window rate limiter.
    
    For production, this should be backed by Redis for distributed rate limiting.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig(
            requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
            requests_per_day=settings.RATE_LIMIT_PER_DAY,
        )
        self._entries: Dict[str, RateLimitEntry] = defaultdict(RateLimitEntry)
        self._lock = asyncio.Lock()
    
    def _get_client_key(self, request: Request, user_id: Optional[str] = None) -> str:
        """Generate rate limit key from request."""
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def _get_endpoint_limits(self, path: str) -> Dict[str, int]:
        """Get rate limits for specific endpoint."""
        # Check for exact match
        if path in self.config.endpoint_limits:
            return self.config.endpoint_limits[path]
        
        # Check for prefix match
        for endpoint, limits in self.config.endpoint_limits.items():
            if path.startswith(endpoint):
                return limits
        
        # Default limits
        return {
            "minute": self.config.requests_per_minute,
            "hour": self.config.requests_per_hour,
            "day": self.config.requests_per_day,
        }
    
    async def check_rate_limit(
        self,
        request: Request,
        user_id: Optional[str] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, info_dict)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, {}
        
        client_key = self._get_client_key(request, user_id)
        limits = self._get_endpoint_limits(request.url.path)
        now = time.time()
        
        async with self._lock:
            entry = self._entries[client_key]
            
            # Reset windows if expired
            if now > entry.minute_reset:
                entry.minute_count = 0
                entry.minute_reset = now + 60
            
            if now > entry.hour_reset:
                entry.hour_count = 0
                entry.hour_reset = now + 3600
            
            if now > entry.day_reset:
                entry.day_count = 0
                entry.day_reset = now + 86400
            
            # Check limits
            minute_limit = limits.get("minute", self.config.requests_per_minute)
            hour_limit = limits.get("hour", self.config.requests_per_hour)
            day_limit = limits.get("day", self.config.requests_per_day)
            
            info = {
                "minute": {
                    "limit": minute_limit,
                    "remaining": max(0, minute_limit - entry.minute_count),
                    "reset": int(entry.minute_reset),
                },
                "hour": {
                    "limit": hour_limit,
                    "remaining": max(0, hour_limit - entry.hour_count),
                    "reset": int(entry.hour_reset),
                },
                "day": {
                    "limit": day_limit,
                    "remaining": max(0, day_limit - entry.day_count),
                    "reset": int(entry.day_reset),
                },
            }
            
            # Check if any limit exceeded
            if entry.minute_count >= minute_limit:
                return False, {"exceeded": "minute", **info}
            
            if entry.hour_count >= hour_limit:
                return False, {"exceeded": "hour", **info}
            
            if entry.day_count >= day_limit:
                return False, {"exceeded": "day", **info}
            
            # Increment counters
            entry.minute_count += 1
            entry.hour_count += 1
            entry.day_count += 1
            entry.last_request = now
            
            return True, info
    
    async def get_remaining(
        self,
        request: Request,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get remaining rate limit info without consuming."""
        client_key = self._get_client_key(request, user_id)
        limits = self._get_endpoint_limits(request.url.path)
        now = time.time()
        
        async with self._lock:
            entry = self._entries[client_key]
            
            minute_limit = limits.get("minute", self.config.requests_per_minute)
            hour_limit = limits.get("hour", self.config.requests_per_hour)
            day_limit = limits.get("day", self.config.requests_per_day)
            
            return {
                "minute": {
                    "limit": minute_limit,
                    "remaining": max(0, minute_limit - entry.minute_count) if now < entry.minute_reset else minute_limit,
                    "reset": int(entry.minute_reset) if now < entry.minute_reset else int(now + 60),
                },
                "hour": {
                    "limit": hour_limit,
                    "remaining": max(0, hour_limit - entry.hour_count) if now < entry.hour_reset else hour_limit,
                    "reset": int(entry.hour_reset) if now < entry.hour_reset else int(now + 3600),
                },
                "day": {
                    "limit": day_limit,
                    "remaining": max(0, day_limit - entry.day_count) if now < entry.day_reset else day_limit,
                    "reset": int(entry.day_reset) if now < entry.day_reset else int(now + 86400),
                },
            }
    
    def clear(self, client_key: Optional[str] = None) -> None:
        """Clear rate limit entries."""
        if client_key:
            self._entries.pop(client_key, None)
        else:
            self._entries.clear()


# Global rate limiter instance
api_rate_limiter = SlidingWindowRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Applies rate limits to all incoming requests based on IP or user ID.
    """
    
    # Paths to exclude from rate limiting
    EXCLUDED_PATHS = {
        "/health",
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
    
    def __init__(self, app, rate_limiter: Optional[SlidingWindowRateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or api_rate_limiter
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # Skip if rate limiting disabled
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Extract user ID from request state if available (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        
        # Check rate limit
        allowed, info = await self.rate_limiter.check_rate_limit(request, user_id)
        
        if not allowed:
            exceeded_type = info.get("exceeded", "minute")
            limit_info = info.get(exceeded_type, {})
            
            logger.warning(
                f"Rate limit exceeded for {self.rate_limiter._get_client_key(request, user_id)}: "
                f"{exceeded_type} limit"
            )
            
            # Return 429 with rate limit info
            response = ErrorResponse(
                error=ErrorDetail(
                    code="RATE_LIMIT_EXCEEDED",
                    message=f"Rate limit exceeded. Please wait before making more requests.",
                    details={
                        "limit_type": exceeded_type,
                        "limit": limit_info.get("limit", 0),
                        "reset_at": limit_info.get("reset", 0),
                    },
                )
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=response.model_dump(),
                headers={
                    "X-RateLimit-Limit": str(limit_info.get("limit", 0)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(limit_info.get("reset", 0)),
                    "Retry-After": str(max(1, limit_info.get("reset", 0) - int(time.time()))),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        minute_info = info.get("minute", {})
        response.headers["X-RateLimit-Limit"] = str(minute_info.get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(minute_info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(minute_info.get("reset", 0))
        
        return response


def configure_rate_limits(
    chat_per_minute: int = 20,
    retrieval_per_minute: int = 60,
    ingestion_per_minute: int = 10,
    auth_per_minute: int = 30,
) -> RateLimitConfig:
    """
    Create rate limit configuration with per-endpoint limits.
    
    Args:
        chat_per_minute: Rate limit for chat endpoints
        retrieval_per_minute: Rate limit for retrieval endpoints
        ingestion_per_minute: Rate limit for ingestion endpoints
        auth_per_minute: Rate limit for auth endpoints
    
    Returns:
        RateLimitConfig with endpoint-specific limits
    """
    return RateLimitConfig(
        requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
        requests_per_day=settings.RATE_LIMIT_PER_DAY,
        endpoint_limits={
            "/api/v1/chat": {"minute": chat_per_minute, "hour": chat_per_minute * 30, "day": chat_per_minute * 300},
            "/api/v1/retrieval": {"minute": retrieval_per_minute, "hour": retrieval_per_minute * 30, "day": retrieval_per_minute * 300},
            "/api/v1/ingest": {"minute": ingestion_per_minute, "hour": ingestion_per_minute * 30, "day": ingestion_per_minute * 300},
            "/api/v1/auth": {"minute": auth_per_minute, "hour": auth_per_minute * 30, "day": auth_per_minute * 300},
        },
    )

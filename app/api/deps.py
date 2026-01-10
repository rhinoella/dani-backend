"""
FastAPI dependencies for authentication and database access.

Provides dependency injection for routes.
"""

from __future__ import annotations

import logging
from typing import Optional, AsyncGenerator
from uuid import UUID
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_async_session
from app.database.models import User
from app.core.auth import GoogleAuthVerifier, GoogleAuthError, GoogleUser
from app.core.tokens import verify_access_token, TokenError
from app.core.config import settings
from app.cache.rate_limiter import rate_limiter, RateLimiter, RateLimitResult
from app.cache.conversation_cache import ConversationCache, UserConversationsCache
from app.cache.redis_client import get_redis_client
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)

# Bearer token security scheme
security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for database session.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async for session in get_async_session():
        yield session


async def get_rate_limiter() -> RateLimiter:
    """Dependency for rate limiter."""
    return rate_limiter


async def get_conversation_cache() -> Optional[ConversationCache]:
    """Dependency for conversation cache."""
    from app.cache.conversation_cache import conversation_cache
    try:
        client = await get_redis_client()
        if client:
            return conversation_cache
    except Exception as e:
        logger.debug(f"Conversation cache unavailable: {e}")
    return None


async def get_user_conversations_cache() -> Optional[UserConversationsCache]:
    """Dependency for user conversations cache."""
    from app.cache.conversation_cache import user_conversations_cache
    try:
        client = await get_redis_client()
        if client:
            return user_conversations_cache
    except Exception as e:
        logger.debug(f"User conversations cache unavailable: {e}")
    return None


async def get_google_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> GoogleUser:
    """
    Dependency to verify Google token and get user info.
    
    NOTE: This is mainly used for initial authentication.
    For regular API calls, use get_current_user which accepts JWTs.
    
    Raises:
        HTTPException 401 if token is invalid or missing
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Skip auth in development if not configured
    if settings.GOOGLE_CLIENT_ID == "__MISSING__":
        if settings.ENV == "development":
            # Return mock user for development
            logger.warning("Google auth not configured - using mock user")
            return GoogleUser(
                google_id="dev-user-123",
                email="dev@example.com",
                name="Development User",
                picture_url=None,
                email_verified=True,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication not configured",
            )
    
    try:
        verifier = GoogleAuthVerifier()
        return verifier.verify_token(credentials.credentials)
    except GoogleAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Accepts application-issued JWT access tokens (preferred) or
    Google ID tokens (for backward compatibility).
    
    Usage:
        @app.get("/me")
        async def get_me(user: User = Depends(get_current_user)):
            return user.to_dict()
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    user_repo = UserRepository(db)
    
    # Development mode fallback
    if settings.ENV == "development" and settings.GOOGLE_CLIENT_ID == "__MISSING__":
        logger.warning("Auth not configured - using dev user")
        user = await user_repo.get_by_google_id("dev-user-123")
        if user:
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Dev user not found. Run seed_users.py first.",
        )
    
    # Try JWT verification first (our issued tokens)
    try:
        payload = verify_access_token(token)
        user = await user_repo.get_by_id(payload.sub)
        if user:
            return user
        # User ID in token but not in DB
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    except TokenError:
        # Not a valid JWT, try Google token for backward compatibility
        pass
    
    # Fallback: Try Google token verification (backward compatibility)
    try:
        verifier = GoogleAuthVerifier()
        google_user = verifier.verify_token(token)
        
        # Look up user by Google ID or email
        user = await user_repo.get_by_google_id(google_user.google_id)
        if not user:
            user = await user_repo.get_by_email(google_user.email)
            if user:
                # Link Google ID to existing user
                user.google_id = google_user.google_id
                user.last_login_at = datetime.now(timezone.utc)
                if google_user.name:
                    user.name = google_user.name
                if google_user.picture_url:
                    user.picture_url = google_user.picture_url
                await db.commit()
                await db.refresh(user)
                logger.info(f"Linked Google account to existing user: {google_user.email}")
                return user
        
        if not user:
            logger.warning(f"Access denied for unregistered user: {google_user.email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Your account is not registered in the system.",
            )
        
        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        return user
        
    except GoogleAuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Dependency for optional authentication.
    
    Returns user if authenticated, None otherwise.
    Does not raise exceptions for missing/invalid tokens.
    
    Usage:
        @app.get("/items")
        async def get_items(user: Optional[User] = Depends(get_optional_user)):
            if user:
                # Authenticated request
            else:
                # Anonymous request
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    user_repo = UserRepository(db)
    
    # Development mode fallback
    if settings.ENV == "development" and settings.GOOGLE_CLIENT_ID == "__MISSING__":
        user = await user_repo.get_by_google_id("dev-user-123")
        return user
    
    # Try JWT verification first (our issued tokens)
    try:
        payload = verify_access_token(token)
        user = await user_repo.get_by_id(payload.sub)
        if user:
            return user
    except TokenError:
        # Not a valid JWT, try Google token
        pass
    
    # Fallback: Try Google token verification
    try:
        verifier = GoogleAuthVerifier()
        google_user = verifier.verify_token_lenient(token)
        if google_user:
            return await user_repo.get_by_google_id(google_user.google_id)
    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
    
    return None


async def check_rate_limit(
    user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to check rate limits.
    
    Raises HTTPException 429 if rate limit exceeded.
    
    Usage:
        @app.post("/chat")
        async def chat(user: User = Depends(check_rate_limit)):
            ...
    """
    result = await rate_limiter.check_rate_limit(str(user.id))
    
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {result.retry_after} seconds.",
            headers={
                "Retry-After": str(result.retry_after),
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(result.reset_at)),
            },
        )
    
    # Increment the rate limit counter
    await rate_limiter.increment(str(user.id))
    
    return user


async def increment_rate_limit(user_id: str) -> None:
    """
    Increment rate limit counters after successful request.
    
    Call this in route handlers after processing the request.
    """
    await rate_limiter.increment(user_id)


class RateLimitHeaders:
    """Helper to add rate limit headers to responses."""
    
    @staticmethod
    async def get_headers(user_id: str) -> dict:
        """Get rate limit headers for response."""
        result = await rate_limiter.check_rate_limit(user_id)
        return {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at)),
        }

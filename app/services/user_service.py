"""
User service for user management operations.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.repositories.user_repository import UserRepository
from app.database.models import User
from app.core.auth import GoogleUser
from app.cache.rate_limiter import RateLimiter
from app.core.config import settings

logger = logging.getLogger(__name__)


class UserService:
    """Service for user management."""
    
    def __init__(self, session: AsyncSession, rate_limiter: Optional[RateLimiter] = None):
        self.session = session
        self.user_repo = UserRepository(session)
        self.rate_limiter = rate_limiter
    
    async def get_or_create_from_google(self, google_user: GoogleUser) -> User:
        """
        Get existing user or create new from Google auth.
        Updates last login timestamp.
        """
        user = await self.user_repo.create_or_update_from_google(
            google_id=google_user.google_id,
            email=google_user.email,
            name=google_user.name,
            picture_url=google_user.picture_url
        )
        
        await self.session.commit()
        logger.info(f"User authenticated: {user.email}")
        
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return await self.user_repo.get_by_id(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return await self.user_repo.get_by_email(email)
    
    async def get_user_profile(self, user_id: str) -> Optional[dict]:
        """Get user profile with statistics."""
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return None
        
        stats = await self.user_repo.get_user_stats(user_id)
        
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "picture_url": user.picture_url,
            "created_at": user.created_at,
            "last_login_at": user.last_login_at,
            "conversation_count": stats["conversation_count"],
            "message_count": stats["message_count"]
        }
    
    async def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        picture_url: Optional[str] = None
    ) -> Optional[User]:
        """Update user profile."""
        user = await self.user_repo.update(
            user_id,
            name=name,
            picture_url=picture_url
        )
        
        if user:
            await self.session.commit()
            logger.info(f"User updated: {user_id}")
        
        return user
    
    async def delete_user(self, user_id: str) -> bool:
        """Soft delete a user."""
        success = await self.user_repo.soft_delete(user_id)
        
        if success:
            await self.session.commit()
            logger.info(f"User deleted: {user_id}")
        
        return success
    
    async def get_rate_limit_status(self, user_id: str) -> dict:
        """Get user's current rate limit status."""
        if not self.rate_limiter:
            return {
                "minute": {"used": 0, "limit": settings.RATE_LIMIT_PER_MINUTE, "reset_in": 60},
                "day": {"used": 0, "limit": settings.RATE_LIMIT_PER_DAY, "reset_in": 86400}
            }
        
        return await self.rate_limiter.get_status(user_id)
    
    async def check_rate_limit(self, user_id: str) -> tuple[bool, Optional[dict]]:
        """
        Check if user is within rate limits.
        Returns (allowed, limit_info).
        """
        if not self.rate_limiter:
            return True, None
        
        result = await self.rate_limiter.check_rate_limit(user_id)
        
        if not result.allowed:
            status = await self.rate_limiter.get_status(user_id)
            return False, {
                "retry_after": result.retry_after,
                "status": status
            }
        
        return True, None
    
    async def increment_rate_limit(self, user_id: str) -> None:
        """Increment rate limit counters after a request."""
        if self.rate_limiter:
            await self.rate_limiter.increment(user_id)

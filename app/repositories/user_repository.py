"""
User repository for user-related database operations.
"""

from typing import Optional, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from app.repositories.base import BaseRepository
from app.database.models import User, Conversation, Message


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class UserRepository(BaseRepository[User]):
    """Repository for user operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(User, session)
    
    async def get_by_google_id(self, google_id: str) -> Optional[User]:
        """Get user by Google ID."""
        query = select(User).where(
            User.google_id == google_id,
            User.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        query = select(User).where(
            User.email == email,
            User.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def create_or_update_from_google(
        self,
        google_id: str,
        email: str,
        name: Optional[str] = None,
        picture_url: Optional[str] = None
    ) -> User:
        """Create a new user or update existing from Google auth."""
        user = await self.get_by_google_id(google_id)
        
        if user:
            # Update existing user
            user.last_login_at = utc_now()
            if name:
                user.name = name
            if picture_url:
                user.picture_url = picture_url
            await self.session.flush()
            await self.session.refresh(user)
            return user
        
        # Create new user
        return await self.create(
            google_id=google_id,
            email=email,
            name=name,
            picture_url=picture_url,
            last_login_at=utc_now()
        )
    
    async def update_last_login(self, user_id: str) -> Optional[User]:
        """Update user's last login timestamp."""
        return await self.update(user_id, last_login_at=utc_now())
    
    async def get_user_stats(self, user_id: str) -> dict:
        """Get user statistics."""
        # Count conversations
        conv_query = select(func.count()).select_from(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        conv_result = await self.session.execute(conv_query)
        conversation_count = conv_result.scalar() or 0
        
        # Count messages
        msg_query = select(func.count()).select_from(Message).join(
            Conversation,
            Message.conversation_id == Conversation.id
        ).where(
            Conversation.user_id == user_id,
            Message.deleted_at.is_(None)
        )
        msg_result = await self.session.execute(msg_query)
        message_count = msg_result.scalar() or 0
        
        return {
            "conversation_count": conversation_count,
            "message_count": message_count
        }
    
    async def search_users(
        self,
        query: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[User]:
        """Search users by email or name."""
        search_query = select(User).where(
            User.deleted_at.is_(None),
            (User.email.ilike(f"%{query}%") | User.name.ilike(f"%{query}%"))
        ).offset(skip).limit(limit)
        
        result = await self.session.execute(search_query)
        return list(result.scalars().all())
    
    async def get_active_users_count(self, days: int = 30) -> int:
        """Get count of users active in the last N days."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = select(func.count()).select_from(User).where(
            User.deleted_at.is_(None),
            User.last_login_at >= cutoff
        )
        result = await self.session.execute(query)
        return result.scalar() or 0

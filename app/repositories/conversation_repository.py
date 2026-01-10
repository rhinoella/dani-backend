"""
Conversation repository for conversation-related database operations.
"""

from typing import Optional, List, Tuple
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from datetime import datetime

from app.repositories.base import BaseRepository
from app.database.models import Conversation, Message


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for conversation operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Conversation, session)
    
    async def get_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
        include_deleted: bool = False
    ) -> List[Conversation]:
        """Get conversations for a user."""
        query = select(Conversation).where(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc())
        
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))
        
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_user_with_preview(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[dict]:
        """
        Get conversations with last message preview.
        Optimized to avoid N+1 queries by using eager loading.
        """
        # Single query with eager loading of messages
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        ).options(
            selectinload(Conversation.messages)
        ).order_by(Conversation.updated_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        conversations = list(result.scalars().all())
        
        output = []
        for conv in conversations:
            # Get last non-deleted message from already loaded messages
            active_messages = [
                m for m in conv.messages 
                if m.deleted_at is None
            ]
            last_message = max(active_messages, key=lambda m: m.created_at) if active_messages else None
            
            preview = None
            if last_message:
                preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content
            
            output.append({
                "conversation": conv,
                "last_message_preview": preview
            })
        
        return output
    
    async def count_by_user(self, user_id: str, include_deleted: bool = False) -> int:
        """Count conversations for a user."""
        query = select(func.count()).select_from(Conversation).where(
            Conversation.user_id == user_id
        )
        
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_with_messages(
        self,
        conversation_id: str,
        message_limit: int = 50
    ) -> Optional[Conversation]:
        """Get conversation with messages."""
        query = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.deleted_at.is_(None)
        ).options(selectinload(Conversation.messages))
        
        result = await self.session.execute(query)
        conversation = result.scalar_one_or_none()
        
        if conversation:
            # Filter and limit messages
            conversation.messages = [
                m for m in conversation.messages 
                if m.deleted_at is None
            ][-message_limit:]
        
        return conversation
    
    async def search_user_conversations(
        self,
        user_id: str,
        query: str,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[Conversation], int]:
        """Search conversations by title or content."""
        # Search in conversation titles
        base_query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
            or_(
                Conversation.title.ilike(f"%{query}%"),
                Conversation.summary.ilike(f"%{query}%")
            )
        ).order_by(Conversation.updated_at.desc())
        
        # Count total
        count_query = select(func.count()).select_from(base_query.subquery())
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0
        
        # Get page
        result = await self.session.execute(
            base_query.offset(skip).limit(limit)
        )
        conversations = list(result.scalars().all())
        
        return conversations, total
    
    async def update_message_count(self, conversation_id: str) -> Optional[Conversation]:
        """Update the message count for a conversation."""
        count_query = select(func.count()).select_from(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        )
        result = await self.session.execute(count_query)
        count = result.scalar() or 0
        
        return await self.update(conversation_id, message_count=count)
    
    async def update_summary(
        self,
        conversation_id: str,
        summary: str
    ) -> Optional[Conversation]:
        """Update conversation summary."""
        return await self.update(conversation_id, summary=summary)
    
    async def auto_generate_title(
        self,
        conversation_id: str,
        first_user_message: str
    ) -> Optional[Conversation]:
        """Generate a meaningful title from first user message."""
        import re
        
        # Clean up the message
        message = first_user_message.strip()
        
        # Remove common filler words from start
        filler_patterns = [
            r'^(hey|hi|hello|please|can you|could you|i want to|i need to|tell me about|what is|what are|what was|what were|how do|how did|how can|who is|who are|who was|when did|when was|where is|where was|why did|why is)\s+',
        ]
        
        cleaned = message
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # If cleaning removed too much, use original
        if len(cleaned) < 10:
            cleaned = message
        
        # Remove question marks and extra punctuation
        cleaned = re.sub(r'[?!]+$', '', cleaned)
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        # Truncate to reasonable length (40 chars for better display)
        if len(cleaned) > 40:
            # Try to cut at a word boundary
            title = cleaned[:40]
            last_space = title.rfind(' ')
            if last_space > 20:  # Only cut at space if we have enough content
                title = title[:last_space]
            title += "..."
        else:
            title = cleaned
        
        # Fallback if title is empty
        if not title or title.isspace():
            title = "New conversation"
        
        return await self.update(conversation_id, title=title)
    
    async def get_recent_conversations(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Conversation]:
        """Get most recent conversations."""
        return await self.get_by_user(user_id, skip=0, limit=limit)
    
    async def bulk_delete_by_user(self, user_id: str) -> int:
        """Soft delete all conversations for a user."""
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        conversations = list(result.scalars().all())
        
        count = 0
        for conv in conversations:
            conv.deleted_at = datetime.utcnow()
            count += 1
        
        await self.session.flush()
        return count
    
    async def verify_ownership(self, conversation_id: str, user_id: str) -> bool:
        """Verify that a conversation belongs to a user."""
        query = select(func.count()).select_from(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        return (result.scalar() or 0) > 0

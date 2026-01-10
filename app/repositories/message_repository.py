"""
Message repository for message-related database operations.
"""

from typing import Optional, List, Tuple
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.repositories.base import BaseRepository
from app.database.models import Message, Conversation


class MessageRepository(BaseRepository[Message]):
    """Repository for message operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Message, session)
    
    async def get_by_conversation(
        self,
        conversation_id: str,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
        ascending: bool = True
    ) -> List[Message]:
        """Get messages for a conversation."""
        query = select(Message).where(
            Message.conversation_id == conversation_id
        )
        
        if not include_deleted:
            query = query.where(Message.deleted_at.is_(None))
        
        if ascending:
            query = query.order_by(Message.created_at.asc())
        else:
            query = query.order_by(Message.created_at.desc())
        
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[Message]:
        """Get most recent messages (in chronological order)."""
        # Get recent messages in descending order, then reverse
        query = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        ).order_by(Message.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        messages = list(result.scalars().all())
        return list(reversed(messages))
    
    async def count_by_conversation(
        self,
        conversation_id: str,
        include_deleted: bool = False
    ) -> int:
        """Count messages in a conversation."""
        query = select(func.count()).select_from(Message).where(
            Message.conversation_id == conversation_id
        )
        
        if not include_deleted:
            query = query.where(Message.deleted_at.is_(None))
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[dict]] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> Message:
        """Add a message to a conversation."""
        return await self.create(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
            confidence_score=confidence_score,
            metadata_=metadata
        )
    
    async def add_user_message(
        self,
        conversation_id: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> Message:
        """Add a user message."""
        return await self.add_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
            metadata=metadata
        )
    
    async def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        sources: Optional[List[dict]] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> Message:
        """Add an assistant message."""
        return await self.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            sources=sources,
            confidence_score=confidence_score,
            metadata=metadata
        )
    
    async def get_last_message(
        self,
        conversation_id: str,
        role: Optional[str] = None
    ) -> Optional[Message]:
        """Get the last message in a conversation."""
        query = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        )
        
        if role:
            query = query.where(Message.role == role)
        
        query = query.order_by(Message.created_at.desc()).limit(1)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_context_messages(
        self,
        conversation_id: str,
        limit: int = 15,
        max_tokens: int = 2000
    ) -> Tuple[List[Message], bool]:
        """
        Get messages for context, respecting token limit.
        Returns messages and whether truncation occurred.
        """
        messages = await self.get_recent_messages(conversation_id, limit * 2)
        
        # Simple token estimation (4 chars per token)
        selected_messages = []
        total_tokens = 0
        truncated = False
        
        for msg in reversed(messages):
            msg_tokens = len(msg.content) // 4
            if total_tokens + msg_tokens > max_tokens:
                truncated = True
                break
            selected_messages.insert(0, msg)
            total_tokens += msg_tokens
            
            if len(selected_messages) >= limit:
                truncated = len(messages) > limit
                break
        
        return selected_messages, truncated
    
    async def search_in_conversation(
        self,
        conversation_id: str,
        query: str,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[Message], int]:
        """Search messages within a conversation."""
        base_query = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
            Message.content.ilike(f"%{query}%")
        ).order_by(Message.created_at.desc())
        
        # Count total
        count_query = select(func.count()).select_from(base_query.subquery())
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0
        
        # Get page
        result = await self.session.execute(
            base_query.offset(skip).limit(limit)
        )
        messages = list(result.scalars().all())
        
        return messages, total
    
    async def search_user_messages(
        self,
        user_id: str,
        query: str,
        skip: int = 0,
        limit: int = 20,
        role: Optional[str] = None
    ) -> Tuple[List[dict], int]:
        """Search messages across all user conversations."""
        base_query = select(Message, Conversation).join(
            Conversation,
            Message.conversation_id == Conversation.id
        ).where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
            Message.deleted_at.is_(None),
            Message.content.ilike(f"%{query}%")
        )
        
        if role:
            base_query = base_query.where(Message.role == role)
        
        base_query = base_query.order_by(Message.created_at.desc())
        
        # Count total
        count_query = select(func.count()).select_from(base_query.subquery())
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0
        
        # Get page
        result = await self.session.execute(
            base_query.offset(skip).limit(limit)
        )
        rows = result.all()
        
        results = [
            {
                "message": row[0],
                "conversation": row[1]
            }
            for row in rows
        ]
        
        return results, total
    
    async def get_messages_by_role(
        self,
        conversation_id: str,
        role: str,
        limit: int = 50
    ) -> List[Message]:
        """Get messages filtered by role."""
        query = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
            Message.role == role
        ).order_by(Message.created_at.asc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def bulk_delete_by_conversation(self, conversation_id: str) -> int:
        """Soft delete all messages in a conversation."""
        query = select(Message).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None)
        )
        result = await self.session.execute(query)
        messages = list(result.scalars().all())
        
        count = 0
        for msg in messages:
            msg.deleted_at = datetime.utcnow()
            count += 1
        
        await self.session.flush()
        return count
    
    async def get_conversation_export(self, conversation_id: str) -> List[dict]:
        """Get all messages for export."""
        messages = await self.get_by_conversation(
            conversation_id,
            limit=10000,
            ascending=True
        )
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "sources": msg.sources,
                "confidence_score": msg.confidence_score
            }
            for msg in messages
        ]

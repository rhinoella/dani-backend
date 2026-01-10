"""
Conversation service for conversation management.
"""

from typing import Optional, List, Tuple, Union
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.database.models import Conversation, Message
from app.cache.conversation_cache import ConversationCache, UserConversationsCache
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationSummary,
    ConversationListResponse,
)
from app.schemas.message import MessageCreate, SourceReference
from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for conversation management."""
    
    def __init__(
        self,
        session: AsyncSession,
        conversation_cache: Optional[ConversationCache] = None,
        user_conversations_cache: Optional[UserConversationsCache] = None
    ):
        self.session = session
        self.conv_repo = ConversationRepository(session)
        self.msg_repo = MessageRepository(session)
        self.conv_cache = conversation_cache
        self.user_conv_cache = user_conversations_cache
    
    async def create_conversation(
        self,
        user_id: Union[str, UUID],
        title: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = await self.conv_repo.create(
            user_id=user_id,
            title=title,
            metadata=metadata
        )
        
        await self.session.commit()
        
        # Invalidate user conversations cache
        if self.user_conv_cache:
            await self.user_conv_cache.invalidate(user_id)
        
        logger.info(f"Conversation created: {conversation.id} for user {user_id}")
        return conversation
    
    async def get_conversation(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID]
    ) -> Optional[Conversation]:
        """Get a conversation if user owns it."""
        conversation = await self.conv_repo.get_by_id(conversation_id)
        
        if conversation and conversation.user_id == user_id:
            return conversation
        
        return None
    
    async def get_conversation_with_messages(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID],
        message_limit: int = 50
    ) -> Optional[dict]:
        """Get conversation with messages."""
        conversation = await self.conv_repo.get_with_messages(
            conversation_id,
            message_limit
        )
        
        if not conversation or conversation.user_id != user_id:
            return None
        
        return {
            "conversation": conversation,
            "messages": conversation.messages
        }
    
    async def list_conversations(
        self,
        user_id: Union[str, UUID],
        page: int = 1,
        page_size: int = 20
    ) -> ConversationListResponse:
        """List user's conversations with pagination."""
        skip = (page - 1) * page_size
        
        # Try cache first
        if self.user_conv_cache:
            cached = await self.user_conv_cache.get_conversations(user_id)
            if cached and skip == 0 and page_size <= len(cached):
                # Return from cache if it covers the requested page
                total = await self.conv_repo.count_by_user(user_id)
                conversations = cached[:page_size]
                return ConversationListResponse(
                    conversations=[
                        ConversationSummary(
                            id=c["id"],
                            title=c.get("title"),
                            message_count=c.get("message_count", 0),
                            created_at=c["created_at"],
                            updated_at=c.get("updated_at"),
                            last_message_preview=c.get("last_message_preview")
                        )
                        for c in conversations
                    ],
                    total=total,
                    page=page,
                    page_size=page_size,
                    has_more=total > page * page_size
                )
        
        # Get from database
        conversations_with_preview = await self.conv_repo.get_by_user_with_preview(
            user_id, skip, page_size
        )
        total = await self.conv_repo.count_by_user(user_id)
        
        # Update cache
        if self.user_conv_cache and skip == 0:
            cache_data = [
                {
                    "id": item["conversation"].id,
                    "title": item["conversation"].title,
                    "message_count": item["conversation"].message_count,
                    "created_at": item["conversation"].created_at.isoformat(),
                    "updated_at": item["conversation"].updated_at.isoformat() if item["conversation"].updated_at else None,
                    "last_message_preview": item["last_message_preview"]
                }
                for item in conversations_with_preview
            ]
            await self.user_conv_cache.set_conversations(user_id, cache_data)
        
        return ConversationListResponse(
            conversations=[
                ConversationSummary(
                    id=item["conversation"].id,
                    title=item["conversation"].title,
                    message_count=item["conversation"].message_count,
                    created_at=item["conversation"].created_at,
                    updated_at=item["conversation"].updated_at,
                    last_message_preview=item["last_message_preview"]
                )
                for item in conversations_with_preview
            ],
            total=total,
            page=page,
            page_size=page_size,
            has_more=total > page * page_size
        )
    
    async def update_conversation(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID],
        update: ConversationUpdate
    ) -> Optional[Conversation]:
        """Update conversation details."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        updated = await self.conv_repo.update(
            conversation_id,
            title=update.title,
            metadata=update.metadata
        )
        
        await self.session.commit()
        
        # Invalidate caches
        if self.user_conv_cache:
            await self.user_conv_cache.invalidate(user_id)
        
        logger.info(f"Conversation updated: {conversation_id}")
        return updated
    
    async def delete_conversation(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID]
    ) -> bool:
        """Soft delete a conversation."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return False
        
        # Soft delete messages first
        await self.msg_repo.bulk_delete_by_conversation(conversation_id)
        
        # Then delete conversation
        success = await self.conv_repo.soft_delete(conversation_id)
        
        if success:
            await self.session.commit()
            
            # Invalidate caches
            if self.conv_cache:
                await self.conv_cache.invalidate(conversation_id)
            if self.user_conv_cache:
                await self.user_conv_cache.invalidate(user_id)
            
            logger.info(f"Conversation deleted: {conversation_id}")
        
        return success
    
    async def search_conversations(
        self,
        user_id: Union[str, UUID],
        query: str,
        page: int = 1,
        page_size: int = 20
    ) -> ConversationListResponse:
        """Search user's conversations."""
        skip = (page - 1) * page_size
        
        conversations, total = await self.conv_repo.search_user_conversations(
            user_id, query, skip, page_size
        )
        
        return ConversationListResponse(
            conversations=[
                ConversationSummary(
                    id=conv.id,
                    title=conv.title,
                    message_count=conv.message_count,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at
                )
                for conv in conversations
            ],
            total=total,
            page=page,
            page_size=page_size,
            has_more=total > page * page_size
        )
    
    async def add_message(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID],
        role: str,
        content: str,
        sources: Optional[Union[List[SourceReference], List[dict]]] = None,
        confidence_score: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> Optional[Message]:
        """Add a message to a conversation."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        # Convert SourceReference to dict if needed
        sources_dict = None
        if sources:
            if isinstance(sources[0], SourceReference):
                sources_dict = [s.model_dump() for s in sources]
            elif isinstance(sources[0], dict):
                sources_dict = sources
        
        message = await self.msg_repo.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources_dict,
            confidence_score=confidence_score,
            metadata=metadata
        )
        
        # Update message count
        await self.conv_repo.update_message_count(conversation_id)
        
        # Auto-generate title from first user message (if no title exists)
        if role == "user" and (not conversation.title or conversation.title in ["", "Untitled", "New conversation"]):
            await self.conv_repo.auto_generate_title(conversation_id, content)
        
        await self.session.commit()
        
        # Update cache
        if self.conv_cache:
            await self.conv_cache.add_message(
                conversation_id,
                {
                    "id": message.id,
                    "role": role,
                    "content": content,
                    "created_at": message.created_at.isoformat()
                }
            )
        
        # Invalidate user conversations cache (for updated_at and preview)
        if self.user_conv_cache:
            await self.user_conv_cache.invalidate(user_id)
        
        return message
    
    async def get_messages(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID],
        page: int = 1,
        page_size: int = 50
    ) -> Optional[Tuple[List[Message], int, bool]]:
        """Get messages for a conversation."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        skip = (page - 1) * page_size
        
        messages = await self.msg_repo.get_by_conversation(
            conversation_id, skip, page_size
        )
        total = await self.msg_repo.count_by_conversation(conversation_id)
        has_more = total > page * page_size
        
        return messages, total, has_more
    
    async def export_conversation(
        self,
        conversation_id: Union[str, UUID],
        user_id: Union[str, UUID]
    ) -> Optional[dict]:
        """Export conversation for download."""
        conversation = await self.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        messages = await self.msg_repo.get_conversation_export(conversation_id)
        
        return {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "messages": messages,
            "metadata": conversation.metadata
        }

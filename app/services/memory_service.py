"""
Memory service for building conversation context for LLM.
"""

from typing import Optional, List, Tuple, Union
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.database.models import Conversation, Message
from app.cache.conversation_cache import ConversationCache
from app.schemas.message import ConversationContext, MessageBase, MessageRole
from app.core.config import settings

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Service for managing conversation memory and context building.
    Handles context window management, summarization, and token budgeting.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        conversation_cache: Optional[ConversationCache] = None
    ):
        self.session = session
        self.conv_repo = ConversationRepository(session)
        self.msg_repo = MessageRepository(session)
        self.conv_cache = conversation_cache
        
        # Configuration
        self.min_context_messages = settings.MIN_HISTORY_MESSAGES
        self.max_context_messages = settings.MAX_HISTORY_MESSAGES
        self.context_token_budget = settings.CONTEXT_TOKEN_BUDGET
        self.summarize_threshold = settings.SUMMARIZE_THRESHOLD
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4
    
    async def get_context(
        self,
        conversation_id: Union[str, UUID],
        include_system_context: bool = True
    ) -> ConversationContext:
        """
        Build conversation context for LLM.
        Uses dynamic context window based on message length.
        """
        # Try cache first for recent messages
        cached_messages = None
        if self.conv_cache:
            cached_messages = await self.conv_cache.get_messages(conversation_id)
        
        if cached_messages:
            messages = [
                MessageBase(
                    role=MessageRole(m["role"]),
                    content=m["content"]
                )
                for m in cached_messages
            ]
            total_messages = len(messages)
        else:
            # Get from database
            db_messages, truncated = await self.msg_repo.get_context_messages(
                conversation_id,
                limit=self.max_context_messages,
                max_tokens=self.context_token_budget
            )
            
            messages = [
                MessageBase(
                    role=MessageRole(m.role),
                    content=m.content
                )
                for m in db_messages
            ]
            total_messages = await self.msg_repo.count_by_conversation(conversation_id)
            
            # Update cache
            if self.conv_cache and db_messages:
                cache_data = [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "created_at": m.created_at.isoformat()
                    }
                    for m in db_messages
                ]
                await self.conv_cache.set_messages(conversation_id, cache_data)
        
        # Calculate token count
        context_token_count = sum(
            self._estimate_tokens(m.content) for m in messages
        )
        
        # Get summary if available and conversation is long
        summary = None
        if total_messages > self.summarize_threshold:
            conversation = await self.conv_repo.get_by_id(conversation_id)
            if conversation and conversation.summary:
                summary = conversation.summary
        
        return ConversationContext(
            messages=messages,
            summary=summary,
            total_messages=total_messages,
            context_token_count=context_token_count,
            truncated=total_messages > len(messages)
        )
    
    async def get_context_for_chat(
        self,
        conversation_id: Union[str, UUID]
    ) -> List[dict]:
        """
        Get context formatted for chat API.
        Returns list of {"role": str, "content": str}.
        """
        context = await self.get_context(conversation_id)
        
        messages = []
        
        # Add summary as system message if available
        if context.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {context.summary}"
            })
        
        # Add recent messages
        for msg in context.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return messages
    
    async def should_summarize(self, conversation_id: str) -> bool:
        """Check if conversation should be summarized."""
        count = await self.msg_repo.count_by_conversation(conversation_id)
        return count >= self.summarize_threshold
    
    async def generate_summary(
        self,
        conversation_id: str,
        llm_summarize_func
    ) -> Optional[str]:
        """
        Generate and store conversation summary.
        Requires an LLM summarization function.
        """
        if not await self.should_summarize(conversation_id):
            return None
        
        # Get messages to summarize (older messages not in context window)
        all_messages = await self.msg_repo.get_by_conversation(
            conversation_id,
            limit=self.summarize_threshold
        )
        
        if len(all_messages) < self.summarize_threshold:
            return None
        
        # Build content for summarization
        content_to_summarize = "\n".join([
            f"{m.role}: {m.content}"
            for m in all_messages[:self.summarize_threshold - self.min_context_messages]
        ])
        
        try:
            summary = await llm_summarize_func(content_to_summarize)
            
            # Store summary
            await self.conv_repo.update_summary(conversation_id, summary)
            await self.session.commit()
            
            logger.info(f"Generated summary for conversation {conversation_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
    
    async def get_adaptive_context(
        self,
        conversation_id: str,
        query_tokens: int = 0
    ) -> ConversationContext:
        """
        Get context with adaptive message count based on available token budget.
        """
        # Calculate available tokens for context
        available_tokens = self.context_token_budget - query_tokens
        
        # Get messages with token awareness
        messages = []
        total_tokens = 0
        truncated = False
        
        # First, try cache
        cached = None
        if self.conv_cache:
            cached = await self.conv_cache.get_messages(conversation_id)
        
        if cached:
            # Use cached messages within token budget
            for m in reversed(cached):
                msg_tokens = self._estimate_tokens(m["content"])
                if total_tokens + msg_tokens > available_tokens:
                    truncated = True
                    break
                messages.insert(0, MessageBase(
                    role=MessageRole(m["role"]),
                    content=m["content"]
                ))
                total_tokens += msg_tokens
                
                if len(messages) >= self.max_context_messages:
                    break
        else:
            # Get from database with dynamic limit
            db_messages, truncated = await self.msg_repo.get_context_messages(
                conversation_id,
                limit=self.max_context_messages,
                max_tokens=available_tokens
            )
            messages = [
                MessageBase(role=MessageRole(m.role), content=m.content)
                for m in db_messages
            ]
            total_tokens = sum(self._estimate_tokens(m.content) for m in messages)
        
        # Ensure minimum context
        if len(messages) < self.min_context_messages:
            # Try to get at least min messages even if over budget
            additional_needed = self.min_context_messages - len(messages)
            additional = await self.msg_repo.get_recent_messages(
                conversation_id,
                limit=self.min_context_messages
            )
            messages = [
                MessageBase(role=MessageRole(m.role), content=m.content)
                for m in additional
            ]
            total_tokens = sum(self._estimate_tokens(m.content) for m in messages)
        
        total_messages = await self.msg_repo.count_by_conversation(conversation_id)
        
        # Get summary if truncated
        summary = None
        if truncated:
            conversation = await self.conv_repo.get_by_id(conversation_id)
            if conversation:
                summary = conversation.summary
        
        return ConversationContext(
            messages=messages,
            summary=summary,
            total_messages=total_messages,
            context_token_count=total_tokens,
            truncated=truncated
        )
    
    async def clear_context_cache(self, conversation_id: str) -> None:
        """Clear cached context for a conversation."""
        if self.conv_cache:
            await self.conv_cache.invalidate(conversation_id)

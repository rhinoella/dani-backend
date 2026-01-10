"""
Conversation cache using Redis.

Caches recent messages for fast context window loading.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.cache.redis_client import RedisCache, RedisList, get_redis_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationCache:
    """
    Cache for conversation messages.
    
    Stores the most recent messages for each conversation
    to enable fast context window building.
    """
    
    def __init__(
        self,
        max_messages: int = 20,
        ttl_seconds: int = 1800,  # 30 minutes
    ):
        """
        Initialize conversation cache.
        
        Args:
            max_messages: Maximum messages to cache per conversation
            ttl_seconds: Time-to-live for cached messages
        """
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._cache = RedisList(prefix="conv", default_ttl=ttl_seconds)
    
    def _messages_key(self, conversation_id: str) -> str:
        """Get key for conversation messages."""
        return f"{conversation_id}:messages"
    
    def _meta_key(self, conversation_id: str) -> str:
        """Get key for conversation metadata."""
        return f"{conversation_id}:meta"
    
    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get cached messages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum messages to return (default: all cached)
            
        Returns:
            List of message dictionaries (newest first)
        """
        key = self._messages_key(conversation_id)
        messages = await self._cache.lrange(key, 0, -1)
        
        # Refresh TTL on access
        await self._cache.expire(key, self.ttl_seconds)
        
        if limit and len(messages) > limit:
            return messages[:limit]
        
        return messages
    
    async def add_message(
        self,
        conversation_id: str,
        message: Dict[str, Any],
    ) -> None:
        """
        Add a message to the cache.
        
        Maintains max_messages limit by trimming old messages.
        
        Args:
            conversation_id: Conversation ID
            message: Message dictionary
        """
        key = self._messages_key(conversation_id)
        
        # Add to front of list (newest first)
        await self._cache.lpush(key, message)
        
        # Trim to max messages
        await self._cache.ltrim(key, 0, self.max_messages - 1)
        
        # Set/refresh TTL
        await self._cache.expire(key, self.ttl_seconds)
    
    async def set_messages(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """
        Replace all cached messages for a conversation.
        
        Used when loading from database.
        
        Args:
            conversation_id: Conversation ID
            messages: List of messages (newest first)
        """
        key = self._messages_key(conversation_id)
        
        try:
            client = await get_redis_client()
            
            # Delete existing and add new in a pipeline
            pipe = client.pipeline()
            pipe.delete(self._cache._key(key))
            
            if messages:
                import json
                serialized = [json.dumps(m, default=str) for m in messages[:self.max_messages]]
                pipe.rpush(self._cache._key(key), *serialized)
                pipe.expire(self._cache._key(key), self.ttl_seconds)
            
            await pipe.execute()
            
        except Exception as e:
            logger.warning(f"Failed to set messages cache: {e}")
    
    async def invalidate(self, conversation_id: str) -> None:
        """
        Invalidate cache for a conversation.
        
        Call this when messages are deleted or conversation is modified.
        
        Args:
            conversation_id: Conversation ID
        """
        await self._cache.delete(self._messages_key(conversation_id))
        await self._cache.delete(self._meta_key(conversation_id))
    
    async def get_metadata(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached conversation metadata."""
        meta_cache = RedisCache(prefix="conv", default_ttl=self.ttl_seconds)
        return await meta_cache.get(self._meta_key(conversation_id))
    
    async def set_metadata(
        self,
        conversation_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Set cached conversation metadata."""
        meta_cache = RedisCache(prefix="conv", default_ttl=self.ttl_seconds)
        await meta_cache.set(self._meta_key(conversation_id), metadata)


class UserConversationsCache:
    """
    Cache for user's conversation list.
    
    Stores recent conversations for quick list display.
    """
    
    def __init__(
        self,
        max_conversations: int = 50,
        ttl_seconds: int = 300,  # 5 minutes
    ):
        """
        Initialize user conversations cache.
        
        Args:
            max_conversations: Maximum conversations to cache per user
            ttl_seconds: Time-to-live
        """
        self.max_conversations = max_conversations
        self.ttl_seconds = ttl_seconds
        self._cache = RedisCache(prefix="user", default_ttl=ttl_seconds)
    
    def _key(self, user_id: str) -> str:
        """Get key for user's conversations."""
        return f"{user_id}:conversations"
    
    async def get_conversations(
        self,
        user_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached conversations for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of conversation summaries or None if not cached
        """
        return await self._cache.get(self._key(user_id))
    
    async def set_conversations(
        self,
        user_id: str,
        conversations: List[Dict[str, Any]],
    ) -> None:
        """
        Cache conversations for a user.
        
        Args:
            user_id: User ID
            conversations: List of conversation summaries
        """
        # Only cache up to max
        to_cache = conversations[:self.max_conversations]
        await self._cache.set(self._key(user_id), to_cache)
    
    async def invalidate(self, user_id: str) -> None:
        """
        Invalidate cache for a user.
        
        Call when conversations are created/deleted/archived.
        
        Args:
            user_id: User ID
        """
        await self._cache.delete(self._key(user_id))


# Global cache instances
conversation_cache = ConversationCache(
    max_messages=20,
    ttl_seconds=1800,  # 30 minutes
)

user_conversations_cache = UserConversationsCache(
    max_conversations=50,
    ttl_seconds=300,  # 5 minutes
)

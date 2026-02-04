"""
Enhanced Memory Service with Semantic Memory Search.

Key Improvements:
1. Semantic search over past conversation messages
2. Entity extraction and tracking across conversations
3. Topic continuity detection
4. Smart context compression with important facts preserved
"""

from __future__ import annotations

import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.cache.conversation_cache import ConversationCache
from app.embeddings.factory import get_embedding_client
from app.llm.ollama import OllamaClient
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationMemory:
    """Enhanced conversation memory with semantic understanding."""
    messages: List[Dict[str, str]]
    relevant_history: List[Dict[str, str]]  # Semantically relevant past messages
    entities: Dict[str, List[str]]  # Extracted entities (people, topics, dates)
    topic_summary: Optional[str]  # Summary of current topic thread
    total_messages: int
    context_token_count: int


class SemanticMemorySearch:
    """
    Searches past conversation messages semantically.
    
    When user says "what did I ask about budgets last time?",
    this finds relevant messages even if they're not recent.
    """
    
    def __init__(self, embedder=None):
        self.embedder = embedder or get_embedding_client()
    
    async def search_messages(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        top_k: int = 5,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search past messages semantically.
        
        Args:
            query: What to search for
            messages: List of message dicts with 'content' and 'role'
            top_k: Number of results
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of relevant messages with similarity scores
        """
        if not messages:
            return []
        
        # Get query embedding
        query_vec = await self.embedder.embed_query(query)
        
        # Get message embeddings (in parallel for speed)
        import asyncio
        
        async def embed_message(msg: Dict[str, Any], idx: int) -> Tuple[int, List[float]]:
            content = msg.get("content", "")[:500]  # Truncate long messages
            vec = await self.embedder.embed_document(content)
            return idx, vec
        
        embeddings = await asyncio.gather(*[
            embed_message(msg, i) for i, msg in enumerate(messages)
        ])
        
        # Calculate similarities
        from numpy import dot
        from numpy.linalg import norm
        
        def cosine_sim(a, b):
            return dot(a, b) / (norm(a) * norm(b))
        
        scored = []
        for idx, vec in embeddings:
            sim = cosine_sim(query_vec, vec)
            if sim >= min_similarity:
                msg_copy = messages[idx].copy()
                msg_copy["similarity"] = sim
                msg_copy["index"] = idx
                scored.append(msg_copy)
        
        # Sort by similarity
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        
        return scored[:top_k]


class EntityExtractor:
    """
    Extracts and tracks entities from conversations.
    
    Entities include:
    - People mentioned (names, roles)
    - Topics discussed
    - Dates and deadlines
    - Projects and initiatives
    - Numbers and metrics
    """
    
    EXTRACTION_PROMPT = """Extract key entities from this conversation snippet.

Text: {text}

Output JSON with these categories (empty arrays if none found):
{{
    "people": ["names and roles mentioned"],
    "topics": ["main topics or subjects"],
    "dates": ["dates, deadlines, timeframes"],
    "projects": ["projects, initiatives, plans"],
    "metrics": ["numbers, percentages, amounts"]
}}

Output ONLY valid JSON:"""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm or OllamaClient()
    
    async def extract(self, messages: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Extract entities from recent messages."""
        if not messages:
            return {"people": [], "topics": [], "dates": [], "projects": [], "metrics": []}
        
        # Combine recent messages
        text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')[:300]}"
            for m in messages[-6:]  # Last 6 messages
        ])
        
        try:
            prompt = self.EXTRACTION_PROMPT.format(text=text)
            response = await self.llm.generate(prompt, options={"num_predict": 300})
            
            # Parse JSON
            import json
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            entities = json.loads(response)
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {"people": [], "topics": [], "dates": [], "projects": [], "metrics": []}


class TopicSummarizer:
    """
    Summarizes the current topic thread for context.
    
    Helps the LLM understand what's being discussed without
    needing full message history.
    """
    
    SUMMARY_PROMPT = """Summarize the current topic being discussed in this conversation.
Focus on: what's being discussed, key points made, and any pending questions.

Conversation:
{conversation}

Write a brief 2-3 sentence summary:"""

    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm or OllamaClient()
    
    async def summarize(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Generate a topic summary."""
        if len(messages) < 3:
            return None
        
        try:
            conversation = "\n".join([
                f"{m.get('role', 'user').title()}: {m.get('content', '')[:400]}"
                for m in messages[-8:]
            ])
            
            prompt = self.SUMMARY_PROMPT.format(conversation=conversation)
            summary = await self.llm.generate(prompt, options={"num_predict": 150})
            
            return summary.strip()
        except Exception as e:
            logger.warning(f"Topic summarization failed: {e}")
            return None


class EnhancedMemoryService:
    """
    Enhanced memory service with semantic search and entity tracking.
    
    Key capabilities:
    1. Find semantically relevant past messages (not just recent)
    2. Track entities across conversation
    3. Provide topic context summaries
    4. Smart token budgeting
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
        
        # Enhanced components
        self.semantic_search = SemanticMemorySearch()
        self.entity_extractor = EntityExtractor()
        self.topic_summarizer = TopicSummarizer()
        
        # Configuration
        self.max_context_messages = settings.MAX_HISTORY_MESSAGES
        self.context_token_budget = settings.CONTEXT_TOKEN_BUDGET
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4
    
    async def get_enhanced_context(
        self,
        conversation_id: str,
        current_query: str,
        include_semantic_search: bool = True,
        include_entities: bool = True,
        include_topic_summary: bool = True,
    ) -> ConversationMemory:
        """
        Get enhanced conversation context.
        
        Args:
            conversation_id: The conversation to get context for
            current_query: The current user query (for semantic search)
            include_semantic_search: Search for semantically relevant past messages
            include_entities: Extract and include entities
            include_topic_summary: Include topic summary
        
        Returns:
            ConversationMemory with enhanced context
        """
        # Get recent messages
        messages = await self._get_recent_messages(conversation_id)
        
        # Format for output
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        
        # Semantic search for relevant history
        relevant_history = []
        if include_semantic_search and len(messages) > 6:
            # Search older messages for relevance to current query
            all_messages = [{"role": m.role, "content": m.content} for m in messages]
            relevant_history = await self.semantic_search.search_messages(
                query=current_query,
                messages=all_messages[:-4],  # Exclude very recent (already in context)
                top_k=3,
                min_similarity=0.55
            )
        
        # Extract entities
        entities = {"people": [], "topics": [], "dates": [], "projects": [], "metrics": []}
        if include_entities:
            entities = await self.entity_extractor.extract(formatted_messages)
        
        # Topic summary
        topic_summary = None
        if include_topic_summary and len(formatted_messages) >= 4:
            topic_summary = await self.topic_summarizer.summarize(formatted_messages)
        
        # Calculate token count
        context_token_count = sum(
            self._estimate_tokens(m["content"]) for m in formatted_messages
        )
        
        return ConversationMemory(
            messages=formatted_messages,
            relevant_history=relevant_history,
            entities=entities,
            topic_summary=topic_summary,
            total_messages=len(messages),
            context_token_count=context_token_count,
        )
    
    async def _get_recent_messages(self, conversation_id: str) -> List[Any]:
        """Get recent messages from cache or database."""
        # Try cache first
        if self.conv_cache:
            cached = await self.conv_cache.get_messages(conversation_id)
            if cached:
                # Convert to Message-like objects
                from types import SimpleNamespace
                return [
                    SimpleNamespace(role=m["role"], content=m["content"])
                    for m in cached
                ]
        
        # Fall back to database
        messages = await self.msg_repo.get_recent_messages(
            conversation_id,
            limit=self.max_context_messages
        )
        return messages
    
    async def get_context_for_chat(
        self,
        conversation_id: str,
        current_query: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Get context formatted for chat API with enhancements.
        
        Returns list of {"role": str, "content": str} including:
        - System message with topic summary and entities
        - Relevant historical messages (if found)
        - Recent conversation messages
        """
        if current_query:
            memory = await self.get_enhanced_context(
                conversation_id,
                current_query,
                include_semantic_search=True,
                include_entities=True,
                include_topic_summary=True,
            )
        else:
            # Basic context without enhancements
            messages = await self._get_recent_messages(conversation_id)
            return [{"role": m.role, "content": m.content} for m in messages]
        
        result = []
        
        # Add topic summary and entities as system context
        if memory.topic_summary or memory.entities:
            system_parts = []
            
            if memory.topic_summary:
                system_parts.append(f"Current topic: {memory.topic_summary}")
            
            # Add entities if present
            if memory.entities.get("people"):
                system_parts.append(f"People mentioned: {', '.join(memory.entities['people'][:5])}")
            if memory.entities.get("topics"):
                system_parts.append(f"Topics: {', '.join(memory.entities['topics'][:5])}")
            
            if system_parts:
                result.append({
                    "role": "system",
                    "content": "Conversation context:\n" + "\n".join(system_parts)
                })
        
        # Add relevant historical messages (if any)
        if memory.relevant_history:
            for msg in memory.relevant_history:
                # Mark as historical context
                result.append({
                    "role": msg["role"],
                    "content": f"[Earlier in conversation] {msg['content']}"
                })
        
        # Add recent messages
        result.extend(memory.messages)
        
        return result


# Factory function
def get_enhanced_memory_service(
    session: AsyncSession,
    conversation_cache: Optional[ConversationCache] = None
) -> EnhancedMemoryService:
    """Get an enhanced memory service instance."""
    return EnhancedMemoryService(session, conversation_cache)

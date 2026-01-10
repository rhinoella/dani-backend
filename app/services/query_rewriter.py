"""
Query Rewriter Service.

Rewrites follow-up queries using conversation history to improve retrieval accuracy.
Handles coreference resolution and context expansion for multi-turn conversations.
"""

from __future__ import annotations

import re
import logging
from typing import Optional, List, Dict, Any

from app.llm.ollama import OllamaClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrites ambiguous follow-up queries using conversation context.
    
    Examples:
        History: "What's our mobile strategy?" → "Launch Q1, React Native..."
        Query: "What about the budget?"
        Rewritten: "What is the budget for the mobile strategy?"
        
        History: "Tell me about the Q1 roadmap"
        Query: "Who's leading it?"
        Rewritten: "Who is leading the Q1 roadmap?"
    """
    
    # Patterns that indicate a follow-up query needing context
    FOLLOWUP_PATTERNS = [
        r"^what about\b",           # "What about the timeline?"
        r"^how about\b",            # "How about costs?"
        r"^and\b",                  # "And the budget?"
        r"^also\b",                 # "Also, who's responsible?"
        r"^but\b",                  # "But what about risks?"
        r"^can you\b.*\b(it|that|this|them)\b",  # "Can you elaborate on that?"
        r"^tell me more\b",         # "Tell me more"
        r"^more details?\b",        # "More details"
        r"^expand on\b",            # "Expand on that"
        r"^elaborate\b",            # "Elaborate"
    ]
    
    # Pronouns and references that need resolution
    PRONOUN_PATTERNS = [
        r"\b(it|its|it's)\b",       # "What's the timeline for it?"
        r"\b(this|that|these|those)\b",  # "Tell me more about that"
        r"\b(they|them|their)\b",   # "When are they launching?"
        r"\b(he|she|his|her)\b",    # "What did she say?"
        r"\bthe same\b",            # "The same for next quarter?"
        r"\b(above|previous|earlier|mentioned)\b",  # "The mentioned project"
    ]
    
    # Short query threshold (likely needs context)
    SHORT_QUERY_WORDS = 5
    
    REWRITE_PROMPT_TEMPLATE = """You are a query rewriting assistant. Your job is to rewrite ambiguous follow-up questions to be self-contained by incorporating context from the conversation history.

CONVERSATION HISTORY:
{history}

CURRENT QUERY: {query}

INSTRUCTIONS:
1. If the query contains pronouns (it, that, this, they, etc.) or references to previous topics, replace them with the actual subjects from history.
2. If the query is short and lacks context, expand it using relevant information from history.
3. If the query is already clear and self-contained, return it unchanged.
4. Keep the rewritten query concise and natural-sounding.
5. Do NOT answer the question - only rewrite it.
6. Return ONLY the rewritten query, nothing else.

REWRITTEN QUERY:"""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        """
        Initialize QueryRewriter.
        
        Args:
            llm_client: Optional LLM client (creates one if not provided)
        """
        self.llm = llm_client or OllamaClient()
        
        # Compile patterns for efficiency
        self._followup_regex = [re.compile(p, re.IGNORECASE) for p in self.FOLLOWUP_PATTERNS]
        self._pronoun_regex = [re.compile(p, re.IGNORECASE) for p in self.PRONOUN_PATTERNS]
    
    def needs_rewrite(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        Determine if a query needs rewriting based on heuristics.
        
        Args:
            query: The user's current query
            conversation_history: Previous messages in conversation
            
        Returns:
            True if query should be rewritten for better retrieval
        """
        # No history = no context to add
        if not conversation_history or len(conversation_history) == 0:
            return False
        
        query_lower = query.lower().strip()
        
        # Check for follow-up patterns
        for pattern in self._followup_regex:
            if pattern.search(query_lower):
                logger.debug(f"Query matches follow-up pattern: {query}")
                return True
        
        # Check for pronouns/references
        for pattern in self._pronoun_regex:
            if pattern.search(query_lower):
                logger.debug(f"Query contains pronouns/references: {query}")
                return True
        
        # Check if query is very short (likely incomplete)
        word_count = len(query_lower.split())
        if word_count <= self.SHORT_QUERY_WORDS:
            logger.debug(f"Query is short ({word_count} words): {query}")
            return True
        
        return False
    
    def _format_history(self, conversation_history: List[Dict[str, str]], max_messages: int = 6) -> str:
        """Format conversation history for the prompt."""
        # Take recent messages
        recent = conversation_history[-max_messages:]
        
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:500]  # Truncate long messages
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    async def rewrite(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        Rewrite a query using conversation history.
        
        Args:
            query: The user's current query
            conversation_history: Previous messages in conversation
            
        Returns:
            Rewritten query (or original if rewriting fails/unnecessary)
        """
        if not self.needs_rewrite(query, conversation_history):
            logger.debug(f"Query doesn't need rewriting: {query}")
            return query
        
        # Format history for prompt
        history_text = self._format_history(conversation_history)
        
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            history=history_text,
            query=query,
        )
        
        try:
            logger.info(f"Rewriting query: '{query}'")
            
            # Use LLM to rewrite
            rewritten = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a query rewriting assistant. Output only the rewritten query.",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200,   # Queries should be short
            )
            
            # Clean up response
            rewritten = rewritten.strip()
            
            # Remove any quotes the LLM might add
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            
            # Sanity check: don't use if it's too different or looks like an answer
            if len(rewritten) > len(query) * 3:
                logger.warning(f"Rewritten query too long, using original")
                return query
            
            if rewritten.lower().startswith(("i ", "the answer", "based on", "according to")):
                logger.warning(f"LLM answered instead of rewriting, using original")
                return query
            
            logger.info(f"Rewritten query: '{query}' → '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query  # Fallback to original
    
    async def rewrite_if_needed(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Conditionally rewrite query and return metadata.
        
        Args:
            query: The user's current query
            conversation_history: Previous messages in conversation
            
        Returns:
            Dict with:
                - original_query: The input query
                - rewritten_query: The rewritten query (or original if not rewritten)
                - was_rewritten: Boolean indicating if rewriting occurred
        """
        if not conversation_history:
            return {
                "original_query": query,
                "rewritten_query": query,
                "was_rewritten": False,
            }
        
        if not self.needs_rewrite(query, conversation_history):
            return {
                "original_query": query,
                "rewritten_query": query,
                "was_rewritten": False,
            }
        
        rewritten = await self.rewrite(query, conversation_history)
        
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "was_rewritten": rewritten != query,
        }


# Singleton instance
_query_rewriter: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    """Get or create the query rewriter singleton."""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter

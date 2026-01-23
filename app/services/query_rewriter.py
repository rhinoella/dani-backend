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
        r"^what about\b",           
        r"^how about\b",           
        r"^and\b",                
        r"^also\b",                
        r"^but\b",                 
        r"^can you\b.*\b(it|that|this|them)\b",  
        r"^tell me more\b",         
        r"^talk more\b",            
        r"^more details?\b",       
        r"^expand on\b",           
        r"^elaborate\b",            
        r"^continue\b",             
        r"^go on\b",                
        r"^explain more\b",         
        r"^what else\b",            
        r"^anything else\b",        
    ]
    
    # Pronouns and references that need resolution - ONLY when they refer to previous context
    # Note: "who", "what", "where" etc. are NOT pronouns that need resolution
    PRONOUN_PATTERNS = [
        r"\bit\b",                          # "it" - always a reference to something
        r"\bits\b",                         # "its" - possessive reference
        r"\bit's\b",                        # "it's" - contraction
        r"\bthis\b(?!\s+(is|was|are|were)\b)",  # "this" but not "this is/was" at start
        r"\bthat\b(?!\s+(is|was|are|were)\b)",  # "that" but not "that is/was"
        r"\b(these|those)\b",   
        r"\b(they|them|their)\b",   
        r"\b(he|she|his|her)\b",    
        r"\bthe same\b",            
        r"\b(above|previous|earlier|mentioned)\b",  
        r"\bthe (meeting|discussion|call|session|plan|document|report|presentation|project|proposal)\b",
    ]
    
    # Patterns that indicate a STANDALONE query (should NOT be rewritten)
    STANDALONE_PATTERNS = [
        r"^who (is|are|was|were)\b",         # "Who is X", "Who are Y" - direct question about entity
        r"^what (is|are|was|were)\b",        # "What is X", "What are Y" - direct definition question
        r"^where (is|are|was|were)\b",       # "Where is X" - direct location question
        r"^when (is|are|was|were|did)\b",    # "When is X" - direct time question
        r"^why (is|are|was|were|did|do)\b",  # "Why is X" - direct reason question
        r"^how (is|are|was|were|do|does|did|can|could|would|should)\b",  # "How is X" - direct method question
        r"^tell me about\b",                 # "Tell me about X" - direct topic request
        r"^explain\b(?!\s+more)",            # "Explain X" but not "explain more"
        r"^describe\b",                      # "Describe X" - direct description request
        r"^summarize\b",                     # "Summarize X" - direct summary request
        r"^list\b",                          # "List X" - direct list request
        r"^give me\b",                       # "Give me X" - direct request
        r"^show me\b",                       # "Show me X" - direct request
        r"^find\b",                          # "Find X" - direct search request
        r"^search\b",                        # "Search for X" - direct search request
    ]
    
    # Short query threshold (likely needs context) - but only if it contains pronouns
    SHORT_QUERY_WORDS = 3  # Reduced from 5 - only very short queries like "continue" or "go on"
    
    REWRITE_PROMPT_TEMPLATE = """You are a query rewriting assistant. Your job is to rewrite ambiguous follow-up questions to be self-contained by incorporating context from the conversation history.

CONVERSATION HISTORY:
{history}

CURRENT QUERY: {query}

INSTRUCTIONS:
1. CRITICAL: First determine if this is a NEW TOPIC or a FOLLOW-UP question.
2. If the query asks about a SPECIFIC NEW entity (person, company, concept) that was NOT discussed before, return the query UNCHANGED.
3. ONLY rewrite if the query contains pronouns (it, that, this, they) or vague references that clearly refer to the previous conversation.
4. If the query is short like "continue", "go on", "talk more", rewrite it to explicitly ask for more information about the PREVIOUS TOPIC.
5. Keep the rewritten query concise and natural-sounding.
6. Do NOT answer the question - only rewrite it.
7. Return ONLY the rewritten query, nothing else.

EXAMPLES:
- History about "Advancly", Query: "Who are splashers" → "Who are splashers" (NEW TOPIC - unchanged)
- History about "mobile app", Query: "who is leading it?" → "Who is leading the mobile app?" (FOLLOW-UP - has "it")
- History about "Q1 roadmap", Query: "continue" → "Continue explaining the Q1 roadmap" (VAGUE - needs context)
- History about "budget", Query: "What is Greenhouse?" → "What is Greenhouse?" (NEW TOPIC - unchanged)

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
        self._standalone_regex = [re.compile(p, re.IGNORECASE) for p in self.STANDALONE_PATTERNS]
    
    def _is_standalone_query(self, query: str) -> bool:
        """
        Check if query is a standalone question that doesn't need context.
        
        Args:
            query: The user's query
            
        Returns:
            True if query appears to be standalone (new topic)
        """
        query_lower = query.lower().strip()
        
        # Check for standalone patterns
        for pattern in self._standalone_regex:
            if pattern.search(query_lower):
                logger.debug(f"Query matches standalone pattern: {query}")
                return True
        
        # Check if query contains a proper noun or specific entity (capitalized word not at start)
        words = query.split()
        if len(words) >= 2:
            # Check words after the first for capitalization (potential proper nouns)
            for word in words[1:]:
                # Skip common words that might be capitalized
                if word[0].isupper() and word.lower() not in ['i', 'the', 'a', 'an', 'is', 'are', 'was', 'were']:
                    logger.debug(f"Query contains potential proper noun '{word}': {query}")
                    return True
        
        return False
    
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
        
        # FIRST: Check for pronouns/references that clearly need resolution
        # This takes priority because "Who is leading it?" should be rewritten
        # even though it looks like a standalone question
        for pattern in self._pronoun_regex:
            if pattern.search(query_lower):
                logger.debug(f"Query contains pronouns/references: {query}")
                return True
        
        # SECOND: Check for follow-up patterns (e.g., "what about", "continue", "go on")
        for pattern in self._followup_regex:
            if pattern.search(query_lower):
                logger.debug(f"Query matches follow-up pattern: {query}")
                return True
        
        # THIRD: Check if it's a standalone query (new topic) - these should NOT be rewritten
        if self._is_standalone_query(query):
            logger.debug(f"Query is standalone, skipping rewrite: {query}")
            return False
        
        # Only rewrite VERY short queries (1-3 words) that are vague commands
        # Examples: "continue", "go on", "more", "explain", "elaborate"
        word_count = len(query_lower.split())
        if word_count <= self.SHORT_QUERY_WORDS:
            # Skip single question words like "What?" "How?" "Why?"
            single_question_words = {"what", "who", "where", "when", "why", "how", "which"}
            clean_query = query_lower.rstrip("?!.")
            if clean_query in single_question_words:
                logger.debug(f"Query is just a question word, skipping rewrite: {query}")
                return False
            
            # But only if it doesn't contain a specific entity (proper noun)
            # "Who are splashers" (3 words) should NOT be rewritten
            # "go on" (2 words) SHOULD be rewritten
            has_entity = any(word[0].isupper() for word in query.split()[1:] if len(word) > 1)
            if not has_entity:
                logger.debug(f"Query is short ({word_count} words) without entity: {query}")
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
                system="You are a query rewriting assistant. Output only the rewritten query.",
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "num_predict": 200,  # Queries should be short
                }
            )
            
            # Clean up response
            rewritten = rewritten.strip()
            
            # Remove any quotes the LLM might add
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            
            # Sanity check: don't use if it's too different or looks like an answer
            if len(rewritten) > min(len(query) * 5, 500):  # Allow up to 5x original length or 500 chars
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

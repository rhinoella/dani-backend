"""
Query processing utilities for RAG pipeline optimization.

Provides:
- Query compression/expansion
- Query intent detection
- Confidence scoring
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Detected query intent and metadata."""
    intent_type: str  # "factual", "summary", "comparison", "action_items", "person_search"
    entities: List[str]
    time_references: List[str]
    confidence: float
    processed_query: str


class QueryProcessor:
    """
    Processes and optimizes queries for better retrieval.
    """
    
    # Intent patterns
    INTENT_PATTERNS = {
        "summary": [
            r"summarize|summary|overview|recap|highlights",
            r"what happened|what was discussed",
            r"give me a|tell me about",
        ],
        "action_items": [
            r"action items?|tasks?|to.?do|assignments?",
            r"who (is|was) (supposed to|going to|assigned)",
            r"what needs to be done|next steps",
        ],
        "decisions": [
            r"decisions?|decided|agreed|concluded|resolved",
            r"what did we decide|final decision",
        ],
        "person_search": [
            r"(what did|did)\s+\w+\s+(say|mention|discuss|talk about)",
            r"\w+'s (thoughts|opinion|view|perspective)",
            r"(from|by|according to)\s+\w+",
        ],
        "comparison": [
            r"compare|comparison|versus|vs\.?|difference",
            r"how does .+ relate to",
        ],
        "timeline": [
            r"when|timeline|schedule|deadline",
            r"(last|next)\s+(week|month|meeting)",
            r"on\s+(monday|tuesday|wednesday|thursday|friday)",
        ],
    }
    
    # Time reference patterns
    TIME_PATTERNS = [
        r"today|yesterday|tomorrow",
        r"last\s+(week|month|quarter|year)",
        r"this\s+(week|month|quarter|year)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"q[1-4]\s*\d{2,4}",
    ]
    
    # Stop words for compression
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "please", "thanks", "thank", "hi", "hello", "hey",
        "can you", "could you", "would you", "i want", "i need", "i'd like",
        "tell me", "show me", "give me", "find me", "help me",
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._intent_compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        self._time_compiled = [re.compile(p, re.IGNORECASE) for p in self.TIME_PATTERNS]
    
    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect the intent and extract entities from query.
        
        Args:
            query: User query
        
        Returns:
            QueryIntent with detected type and metadata
        """
        query_lower = query.lower()
        
        # Detect intent type
        intent_scores: Dict[str, float] = {}
        for intent_type, patterns in self._intent_compiled.items():
            matches = sum(1 for p in patterns if p.search(query_lower))
            if matches > 0:
                intent_scores[intent_type] = matches / len(patterns)
        
        if intent_scores:
            intent_type = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[intent_type] + 0.5, 1.0)
        else:
            intent_type = "factual"
            confidence = 0.5
        
        # Extract entities (capitalized words that aren't at sentence start)
        words = query.split()
        entities = []
        for i, word in enumerate(words):
            # Skip first word (might be capitalized due to sentence start)
            if i == 0:
                continue
            # Check if word is capitalized and not all caps
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and not clean_word.isupper():
                entities.append(clean_word)
        
        # Extract time references
        time_refs = []
        for pattern in self._time_compiled:
            matches = pattern.findall(query_lower)
            time_refs.extend(matches)
        
        # Create processed query
        processed = self.compress_query(query)
        
        return QueryIntent(
            intent_type=intent_type,
            entities=entities,
            time_references=time_refs,
            confidence=confidence,
            processed_query=processed,
        )
    
    def compress_query(self, query: str) -> str:
        """
        Compress query by removing filler words while keeping semantic meaning.
        
        Useful for generating more focused embeddings.
        
        Args:
            query: Original query
        
        Returns:
            Compressed query string
        """
        # Lowercase for stop word matching
        words = query.split()
        
        # Keep words that are:
        # 1. Not stop words
        # 2. Capitalized (proper nouns)
        # 3. Contain numbers
        compressed = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            
            # Keep if capitalized (likely proper noun)
            if word[0].isupper() and len(word) > 1:
                compressed.append(word)
                continue
            
            # Keep if contains numbers
            if any(c.isdigit() for c in word):
                compressed.append(word)
                continue
            
            # Skip stop words
            if word_lower in self.STOP_WORDS:
                continue
            
            # Keep significant words
            if len(word_lower) > 2:
                compressed.append(word_lower)
        
        return " ".join(compressed)
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple variations for better recall.
        
        Args:
            query: Original query
        
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add compressed version
        compressed = self.compress_query(query)
        if compressed != query.lower():
            variations.append(compressed)
        
        # Add question rephrasing
        query_lower = query.lower()
        
        # Convert "what did X say about Y" to "X Y"
        match = re.search(r"what did (\w+) (say|mention|discuss) about (.+)", query_lower)
        if match:
            variations.append(f"{match.group(1)} {match.group(3)}")
        
        # Convert "tell me about X" to just "X"
        match = re.search(r"(?:tell|show|give) me (?:about )?(.+)", query_lower)
        if match:
            variations.append(match.group(1))
        
        return list(set(variations))


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG responses.
    
    Thresholds calibrated for raw cosine similarity on conversational content,
    which typically ranges from 0.05-0.30 for good matches.
    """
    
    def __init__(
        self,
        min_chunks_for_high_confidence: int = 3,
        min_avg_similarity: float = 0.08,   # Lowered from 0.75 - raw cosine is low
        min_top_similarity: float = 0.12,   # Lowered from 0.80
    ):
        self.min_chunks = min_chunks_for_high_confidence
        self.min_avg_similarity = min_avg_similarity
        self.min_top_similarity = min_top_similarity
    
    def score(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """
        Calculate confidence score based on retrieval quality.
        
        Args:
            chunks: Retrieved chunks with scores
            query: Original query
        
        Returns:
            Confidence metrics
        """
        if not chunks:
            return {
                "score": 0.0,
                "level": "none",
                "reason": "no_chunks_retrieved",
                "metrics": {},
            }
        
        # Extract scores
        scores = [c.get("score", 0) for c in chunks]
        
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # Calculate confidence components
        top_score_factor = min(top_score / self.min_top_similarity, 1.0)
        avg_score_factor = min(avg_score / self.min_avg_similarity, 1.0)
        chunk_count_factor = min(len(chunks) / self.min_chunks, 1.0)
        consistency_factor = 1.0 - min(score_variance, 0.3)  # Lower variance = higher confidence
        
        # Weighted combination
        confidence = (
            top_score_factor * 0.35 +
            avg_score_factor * 0.30 +
            chunk_count_factor * 0.15 +
            consistency_factor * 0.20
        )
        
        # Determine confidence level
        if confidence >= 0.85:
            level = "high"
        elif confidence >= 0.65:
            level = "medium"
        elif confidence >= 0.45:
            level = "low"
        else:
            level = "very_low"
        
        # Determine reason
        reasons = []
        if top_score < self.min_top_similarity:
            reasons.append("low_relevance")
        if len(chunks) < self.min_chunks:
            reasons.append("few_sources")
        if score_variance > 0.2:
            reasons.append("inconsistent_relevance")
        
        return {
            "score": round(confidence, 3),
            "level": level,
            "reason": ", ".join(reasons) if reasons else "good_retrieval",
            "metrics": {
                "top_similarity": round(top_score, 3),
                "avg_similarity": round(avg_score, 3),
                "chunk_count": len(chunks),
                "score_variance": round(score_variance, 4),
            },
        }
    
    def should_fallback(self, confidence: Dict[str, Any]) -> bool:
        """
        Determine if response should include fallback/disclaimer.
        
        Args:
            confidence: Confidence metrics from score()
        
        Returns:
            True if fallback should be shown
        """
        return confidence["level"] in ["low", "very_low"]
    
    def get_disclaimer(self, confidence: Dict[str, Any]) -> Optional[str]:
        """
        Get appropriate disclaimer based on confidence level.
        
        Args:
            confidence: Confidence metrics
        
        Returns:
            Disclaimer text or None
        """
        if confidence["level"] == "very_low":
            return (
                "⚠️ I have limited information about this topic. "
                "The response may be incomplete or inaccurate."
            )
        elif confidence["level"] == "low":
            return (
                "ℹ️ I found some relevant information, but it may not fully "
                "answer your question. Please verify with additional sources."
            )
        return None

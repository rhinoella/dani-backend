"""
Email Style Service.

Manages indexing and retrieval of past email examples to learn
Bunmi's writing style patterns for ghostwriting.

Features:
- Index email examples with metadata
- Extract stylistic patterns (greetings, sign-offs, phrases)
- Retrieve similar emails for style injection
- Analyze writing characteristics
"""

from __future__ import annotations

import logging
import re
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from app.embeddings.factory import get_embedding_client
from app.vectorstore.qdrant import QdrantStore
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailTone(str, Enum):
    """Email tone classifications."""
    FORMAL = "formal"
    INFORMAL = "informal"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    DIRECT = "direct"


class EmailStyleService:
    """
    Service for learning and applying Bunmi's email writing style.
    
    Indexes past email examples and extracts stylistic patterns
    that can be injected into ghostwritten drafts.
    """

    def __init__(self):
        self.embedder = get_embedding_client()
        self.store = QdrantStore()
        self.collection = settings.QDRANT_COLLECTION_EMAIL_STYLES
        self._initialized = False

    async def ensure_collection(self) -> None:
        """Ensure the email styles collection exists."""
        if self._initialized:
            return
        
        try:
            # Get embedding dimension
            test_embedding = await self.embedder.embed_document("test")
            vector_size = len(test_embedding)
            
            self.store.ensure_collection(self.collection, vector_size)
            self._initialized = True
            logger.info(f"[EMAIL_STYLE] Collection '{self.collection}' ready")
        except Exception as e:
            logger.error(f"[EMAIL_STYLE] Failed to initialize collection: {e}")
            raise

    async def index_email(
        self,
        email_content: str,
        subject: Optional[str] = None,
        recipient_type: Optional[str] = None,  # e.g., "team", "client", "board", "vendor"
        tone: Optional[EmailTone] = None,
        date: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Index an email example for style learning.
        
        Args:
            email_content: Full email text
            subject: Email subject line
            recipient_type: Type of recipient (team, client, board, etc.)
            tone: Tone classification
            date: Date the email was sent
            tags: Additional tags for categorization
            
        Returns:
            Dict with indexing result and extracted patterns
        """
        await self.ensure_collection()
        
        # Generate unique ID based on content hash
        content_hash = hashlib.md5(email_content.encode()).hexdigest()[:12]
        email_id = f"email_{content_hash}"
        
        # Extract stylistic patterns
        patterns = self._extract_patterns(email_content)
        
        # Detect tone if not provided
        if not tone:
            tone = self._detect_tone(email_content)
        
        # Create embedding
        embedding = await self.embedder.embed_document(email_content)
        
        # Build metadata
        metadata = {
            "email_id": email_id,
            "subject": subject or "",
            "recipient_type": recipient_type or "general",
            "tone": tone.value if tone else "direct",
            "date": date or datetime.now().isoformat()[:10],
            "tags": tags or [],
            "word_count": len(email_content.split()),
            "greeting": patterns.get("greeting", ""),
            "sign_off": patterns.get("sign_off", ""),
            "has_cta": patterns.get("has_cta", False),
            "paragraph_count": patterns.get("paragraph_count", 1),
            "avg_sentence_length": patterns.get("avg_sentence_length", 0),
            "indexed_at": datetime.now().isoformat(),
        }
        
        # Build point for Qdrant
        from qdrant_client.http import models as qm
        
        point = qm.PointStruct(
            id=content_hash,
            vector=embedding,
            payload={
                "text": email_content,
                **metadata,
            }
        )
        
        # Upsert to collection
        self.store.upsert(self.collection, [point])
        
        logger.info(f"[EMAIL_STYLE] Indexed email: {email_id}, tone={tone}, recipient={recipient_type}")
        
        return {
            "email_id": email_id,
            "patterns": patterns,
            "metadata": metadata,
            "status": "indexed",
        }

    async def index_emails_batch(
        self,
        emails: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Index multiple emails in batch.
        
        Args:
            emails: List of email dicts with 'content' and optional metadata
            
        Returns:
            Summary of batch indexing
        """
        await self.ensure_collection()
        
        results = []
        for email in emails:
            try:
                result = await self.index_email(
                    email_content=email["content"],
                    subject=email.get("subject"),
                    recipient_type=email.get("recipient_type"),
                    tone=EmailTone(email["tone"]) if email.get("tone") else None,
                    date=email.get("date"),
                    tags=email.get("tags"),
                )
                results.append({"status": "success", **result})
            except Exception as e:
                results.append({"status": "error", "error": str(e)})
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "total": len(emails),
            "success": success_count,
            "failed": len(emails) - success_count,
            "results": results,
        }

    async def get_similar_emails(
        self,
        query: str,
        recipient_type: Optional[str] = None,
        tone: Optional[EmailTone] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past emails for style reference.
        
        Args:
            query: The topic/content to match
            recipient_type: Filter by recipient type
            tone: Filter by tone
            limit: Maximum emails to retrieve
            
        Returns:
            List of similar email examples with metadata
        """
        await self.ensure_collection()
        
        # Create query embedding
        query_embedding = await self.embedder.embed_query(query)
        
        # Build filter
        from qdrant_client.http import models as qm
        
        conditions = []
        if recipient_type:
            conditions.append(
                qm.FieldCondition(
                    key="recipient_type",
                    match=qm.MatchValue(value=recipient_type),
                )
            )
        if tone:
            conditions.append(
                qm.FieldCondition(
                    key="tone",
                    match=qm.MatchValue(value=tone.value),
                )
            )
        
        qdrant_filter = qm.Filter(must=conditions) if conditions else None
        
        # Search
        results = await self.store.search(
            collection=self.collection,
            query_vector=query_embedding,
            limit=limit,
            filter_=qdrant_filter,
        )
        
        if not results:
            return []
        
        output = []
        for hit in results:
            payload = hit.payload or {}
            output.append({
                "text": payload.get("text", ""),
                "subject": payload.get("subject", ""),
                "recipient_type": payload.get("recipient_type"),
                "tone": payload.get("tone"),
                "greeting": payload.get("greeting"),
                "sign_off": payload.get("sign_off"),
                "score": hit.score,
            })
        return output

    async def get_style_patterns(
        self,
        recipient_type: Optional[str] = None,
        tone: Optional[EmailTone] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate stylistic patterns from indexed emails.
        
        Returns common greetings, sign-offs, phrases, and writing characteristics.
        """
        await self.ensure_collection()
        
        # Get all emails (or filtered subset)
        # For efficiency, we'll sample from the collection
        from qdrant_client.http import models as qm
        
        conditions = []
        if recipient_type:
            conditions.append(
                qm.FieldCondition(
                    key="recipient_type",
                    match=qm.MatchValue(value=recipient_type),
                )
            )
        if tone:
            conditions.append(
                qm.FieldCondition(
                    key="tone",
                    match=qm.MatchValue(value=tone.value),
                )
            )
        
        qdrant_filter = qm.Filter(must=conditions) if conditions else None
        
        # Scroll through collection to aggregate patterns
        try:
            points, _ = self.store.client.scroll(
                collection_name=self.collection,
                scroll_filter=qdrant_filter,
                limit=100,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"[EMAIL_STYLE] Failed to scroll collection: {e}")
            return self._get_default_patterns()
        
        if not points:
            return self._get_default_patterns()
        
        # Aggregate patterns
        greetings = {}
        sign_offs = {}
        avg_word_count = 0
        avg_paragraphs = 0
        
        for point in points:
            payload = point.payload
            if payload is None:
                continue
            
            greeting = payload.get("greeting", "")
            if greeting:
                greetings[greeting] = greetings.get(greeting, 0) + 1
            
            sign_off = payload.get("sign_off", "")
            if sign_off:
                sign_offs[sign_off] = sign_offs.get(sign_off, 0) + 1
            
            avg_word_count += payload.get("word_count", 0)
            avg_paragraphs += payload.get("paragraph_count", 0)
        
        n = len(points)
        
        return {
            "common_greetings": sorted(greetings.items(), key=lambda x: -x[1])[:5],
            "common_sign_offs": sorted(sign_offs.items(), key=lambda x: -x[1])[:5],
            "avg_word_count": round(avg_word_count / n) if n > 0 else 150,
            "avg_paragraphs": round(avg_paragraphs / n) if n > 0 else 3,
            "sample_count": n,
            "filter_applied": {
                "recipient_type": recipient_type,
                "tone": tone.value if tone else None,
            },
        }

    def _extract_patterns(self, email_content: str) -> Dict[str, Any]:
        """Extract stylistic patterns from email content."""
        lines = email_content.strip().split("\n")
        paragraphs = [p.strip() for p in email_content.split("\n\n") if p.strip()]
        
        # Extract greeting (first non-empty line)
        greeting = ""
        for line in lines:
            line = line.strip()
            if line:
                # Check if it looks like a greeting
                if any(g in line.lower() for g in ["hi", "hello", "dear", "hey", "good morning", "good afternoon"]):
                    greeting = line
                break
        
        # Extract sign-off (look for common patterns near end)
        sign_off = ""
        sign_off_patterns = [
            r"(?i)(best|regards|thanks|cheers|sincerely|warm regards|best regards|thank you|kind regards)",
            r"(?i)(talk soon|looking forward|let me know)",
        ]
        
        for line in reversed(lines[-5:]):
            line = line.strip()
            for pattern in sign_off_patterns:
                if re.search(pattern, line):
                    sign_off = line
                    break
            if sign_off:
                break
        
        # Detect call-to-action
        cta_patterns = [
            r"(?i)(let me know|please.*respond|can you|would you|action required|by.*(?:monday|tuesday|wednesday|thursday|friday|eod|end of day|tomorrow))",
        ]
        has_cta = any(re.search(p, email_content) for p in cta_patterns)
        
        # Calculate sentence length
        sentences = re.split(r'[.!?]+', email_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences)
            if sentences else 0
        )
        
        return {
            "greeting": greeting,
            "sign_off": sign_off,
            "has_cta": has_cta,
            "paragraph_count": len(paragraphs),
            "avg_sentence_length": round(avg_sentence_length, 1),
        }

    def _detect_tone(self, email_content: str) -> EmailTone:
        """Auto-detect email tone from content."""
        content_lower = email_content.lower()
        
        # Urgent indicators
        if any(w in content_lower for w in ["urgent", "asap", "immediately", "critical", "time-sensitive"]):
            return EmailTone.URGENT
        
        # Formal indicators
        if any(w in content_lower for w in ["dear", "sincerely", "respectfully", "pursuant"]):
            return EmailTone.FORMAL
        
        # Friendly/informal indicators
        if any(w in content_lower for w in ["hey", "cheers", "catch up", "hope you're well", "!"]):
            return EmailTone.FRIENDLY
        
        # Default to direct
        return EmailTone.DIRECT

    def _get_default_patterns(self) -> Dict[str, Any]:
        """Return default patterns when no emails are indexed."""
        return {
            "common_greetings": [("Hi [Name]", 1), ("Hello", 1)],
            "common_sign_offs": [("Best,\nBunmi", 1), ("Thanks,\nBunmi", 1)],
            "avg_word_count": 150,
            "avg_paragraphs": 3,
            "sample_count": 0,
            "filter_applied": {},
        }

    def build_style_context(
        self,
        similar_emails: List[Dict[str, Any]],
        patterns: Dict[str, Any],
    ) -> str:
        """
        Build a style context string for injection into prompts.
        
        Args:
            similar_emails: Retrieved similar email examples
            patterns: Aggregated style patterns
            
        Returns:
            Formatted style context for prompt injection
        """
        context_parts = []
        
        # Add pattern summary
        context_parts.append("BUNMI'S EMAIL STYLE PATTERNS:")
        
        if patterns.get("common_greetings"):
            greetings = [g[0] for g in patterns["common_greetings"][:3]]
            context_parts.append(f"- Common greetings: {', '.join(greetings)}")
        
        if patterns.get("common_sign_offs"):
            sign_offs = [s[0] for s in patterns["common_sign_offs"][:3]]
            context_parts.append(f"- Common sign-offs: {', '.join(sign_offs)}")
        
        if patterns.get("avg_word_count"):
            context_parts.append(f"- Typical length: ~{patterns['avg_word_count']} words")
        
        if patterns.get("avg_paragraphs"):
            context_parts.append(f"- Typical structure: {patterns['avg_paragraphs']} paragraphs")
        
        # Add example emails
        if similar_emails:
            context_parts.append("\nREFERENCE EMAILS FROM BUNMI:")
            for i, email in enumerate(similar_emails[:2], 1):
                context_parts.append(f"\n--- Example {i} ---")
                # Truncate if too long
                text = email.get("text", "")
                if len(text) > 500:
                    text = text[:500] + "..."
                context_parts.append(text)
        
        return "\n".join(context_parts)


# Singleton instance
_email_style_service: Optional[EmailStyleService] = None


def get_email_style_service() -> EmailStyleService:
    """Get or create the email style service singleton."""
    global _email_style_service
    if _email_style_service is None:
        _email_style_service = EmailStyleService()
    return _email_style_service

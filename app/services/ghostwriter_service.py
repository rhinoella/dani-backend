"""
Ghostwriter Service.

Generates content (LinkedIn posts, emails, blog drafts, etc.) 
in Bunmi's voice based on RAG context from meetings/documents.

For emails, also retrieves past email examples to inject stylistic patterns.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, cast
from enum import Enum

from app.llm.ollama import OllamaClient
from app.services.retrieval_service import RetrievalService
from app.schemas.retrieval import MetadataFilter

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Supported content types for ghostwriting."""
    LINKEDIN_POST = "linkedin_post"
    EMAIL = "email"
    BLOG_DRAFT = "blog_draft"
    TWEET_THREAD = "tweet_thread"
    NEWSLETTER = "newsletter"
    MEETING_SUMMARY = "meeting_summary"


# Content-specific prompts that maintain Bunmi's voice
CONTENT_TEMPLATES = {
    ContentType.LINKEDIN_POST: """You are ghostwriting a LinkedIn post for DANI (a tech executive).

WRITING STYLE:
- Professional yet personable
- Start with a hook (insight, question, or bold statement)
- Share a specific lesson or insight from experience
- Include 1-2 relevant hashtags at the end
- Keep it between 150-300 words
- Use short paragraphs (1-3 sentences each)
- End with a question or call to action to encourage engagement
- NO AI disclaimers or mentions of being an assistant

TOPIC/CONTEXT FROM MEETINGS:
{context}

USER REQUEST:
{request}

Write the LinkedIn post now. Output ONLY the post content, ready to publish:""",

    ContentType.EMAIL: """You are drafting an email for DANI (a tech executive).

WRITING STYLE:
- Professional, direct, and confident
- Clear subject line
- Brief greeting
- Get to the point quickly (2-3 short paragraphs max)
- Specific call-to-action if needed
- Professional sign-off
- NO fluff, NO AI mentions
- Match the tone to the recipient relationship (formal/informal)

{style_context}

RELEVANT CONTEXT FROM MEETINGS:
{context}

USER REQUEST:
{request}

Write the email now. Include subject line. Output ONLY the email, ready to send:""",

    ContentType.BLOG_DRAFT: """You are drafting a blog post for DANI (a tech executive).

WRITING STYLE:
- Thought leadership tone - confident and insightful
- Start with a compelling hook
- Use headers to structure the content
- Include specific examples and insights from experience
- Actionable takeaways
- 500-800 words
- NO AI disclaimers

RELEVANT CONTEXT FROM MEETINGS/DOCUMENTS:
{context}

USER REQUEST:
{request}

Write the blog draft now. Include a title. Output ONLY the blog post:""",

    ContentType.TWEET_THREAD: """You are writing a Twitter/X thread for DANI (a tech executive).

WRITING STYLE:
- Each tweet max 280 characters
- Start with a hook tweet
- Number the tweets (1/, 2/, etc.)
- 4-8 tweets total
- Punchy, direct language
- End with a summary or call-to-action
- Include 1-2 relevant hashtags in the last tweet
- NO AI mentions

RELEVANT CONTEXT:
{context}

USER REQUEST:
{request}

Write the thread now. Output ONLY the tweets:""",

    ContentType.NEWSLETTER: """You are writing a newsletter section for DANI (a tech executive).

WRITING STYLE:
- Conversational but professional
- Personal insights and lessons
- Include specific stories/examples
- Actionable advice
- 300-500 words
- NO AI disclaimers

RELEVANT CONTEXT FROM MEETINGS:
{context}

USER REQUEST:
{request}

Write the newsletter section now:""",

    ContentType.MEETING_SUMMARY: """You are summarizing a meeting for DANI (a tech executive).

FORMAT:
- Executive Summary (2-3 sentences)
- Key Decisions Made (bullet points)
- Action Items (with owners if mentioned)
- Open Questions/Follow-ups
- Keep it concise and actionable

MEETING CONTEXT:
{context}

SPECIFIC FOCUS (if any):
{request}

Write the meeting summary now:""",
}


class GhostwriterService:
    """
    Generates content in Bunmi's voice using RAG context.
    
    For emails, also retrieves past email examples to learn and inject 
    stylistic patterns (greetings, sign-offs, tone, CTAs).
    """

    def __init__(self):
        self.llm = OllamaClient()
        self.retrieval = RetrievalService()
        self._email_style_service = None  # Lazy-loaded
    
    @property
    def email_style(self):
        """Lazy-load email style service."""
        if self._email_style_service is None:
            from app.services.email_style_service import EmailStyleService
            self._email_style_service = EmailStyleService()
        return self._email_style_service

    async def generate(
        self,
        content_type: ContentType,
        request: str,
        topic: Optional[str] = None,
        doc_type: Optional[str] = None,
        additional_context: Optional[str] = None,
        tone: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate content in Bunmi's voice.

        Args:
            content_type: Type of content to generate (linkedin_post, email, etc.)
            request: User's request/prompt for what to write about
            topic: Optional topic to search for in RAG (defaults to request)
            doc_type: Filter sources by type (meeting, email, document)
            additional_context: Extra context to include
            tone: Override tone (e.g., "formal", "casual", "urgent")
            max_length: Approximate max length in words

        Returns:
            Dict with generated content and metadata
        """
        import time
        start_time = time.time()

        # Get the template for this content type
        if content_type not in CONTENT_TEMPLATES:
            return {
                "error": f"Unsupported content type: {content_type}",
                "supported_types": [ct.value for ct in ContentType],
            }

        template = CONTENT_TEMPLATES[content_type]

        # Search for relevant context
        search_query = topic or request
        
        # Build metadata filter
        metadata_filter = None
        if doc_type and doc_type != "all":
            # Cast doc_type string to DocSourceType
            from app.schemas.retrieval import DocSourceType
            doc_type_literal = cast(DocSourceType, doc_type)
            metadata_filter = MetadataFilter(doc_type=doc_type_literal)

        logger.info(f"[GHOSTWRITER] Generating {content_type.value} for request: {request[:100]}...")
        
        # Retrieve relevant context
        retrieval_start = time.time()
        retrieval_result = await self.retrieval.search_with_confidence(
            query=search_query,
            limit=6,
            metadata_filter=metadata_filter,
        )
        chunks = retrieval_result["chunks"]
        confidence = retrieval_result["confidence"]
        retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

        logger.info(f"[GHOSTWRITER] Retrieved {len(chunks)} chunks in {retrieval_ms}ms")

        # Format context from chunks
        context_parts = []
        sources = []
        
        for chunk in chunks:
            title = chunk.get("title", "Untitled")
            text = chunk.get("text", "")
            date = chunk.get("date")
            speakers = chunk.get("speakers", [])
            
            # Format chunk for context
            chunk_header = f"From '{title}'"
            if speakers:
                chunk_header += f" (Speakers: {', '.join(speakers[:3])})"
            
            context_parts.append(f"{chunk_header}:\n{text}")
            
            # Track sources
            sources.append({
                "title": title,
                "date": date,
                "transcript_id": chunk.get("transcript_id"),
                "relevance_score": chunk.get("score", 0),
            })

        # Combine context
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No specific context available."
        
        # Add additional context if provided
        if additional_context:
            context += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"

        # For emails, retrieve learned style patterns from past email corpus
        style_context = ""
        if content_type == ContentType.EMAIL:
            try:
                style_context = await self._get_email_style_context(search_query)
                if style_context:
                    logger.info("[GHOSTWRITER] Injected email style patterns from past emails")
            except Exception as e:
                logger.warning(f"[GHOSTWRITER] Could not retrieve email style patterns: {e}")
                style_context = ""

        # Modify request with tone if specified
        modified_request = request
        if tone:
            modified_request = f"[Tone: {tone}] {request}"
        if max_length:
            modified_request += f" (approximately {max_length} words)"

        # Build the final prompt
        # For email, include style context; for others, just context and request
        if content_type == ContentType.EMAIL:
            prompt = template.format(
                context=context,
                request=modified_request,
                style_context=style_context,
            )
        else:
            prompt = template.format(
                context=context,
                request=modified_request,
            )

        # Generate content
        generation_start = time.time()
        try:
            # Embed system instruction into prompt (OllamaClient.generate only takes prompt and stream)
            full_prompt = f"""SYSTEM: You are DANI's ghostwriter. Write exactly what is requested, nothing more. No preambles, no explanations, no AI disclaimers. DO NOT use markdown formatting (no bold **, no headers #). Produce clean plain text.

{prompt}"""
            content = await self.llm.generate(
                prompt=full_prompt,
            )
            content = content.strip()
            generation_ms = round((time.time() - generation_start) * 1000, 2)
            
        except Exception as e:
            logger.error(f"[GHOSTWRITER] Generation failed: {e}")
            return {
                "error": f"Content generation failed: {str(e)}",
                "content_type": content_type.value,
            }

        total_ms = round((time.time() - start_time) * 1000, 2)
        word_count = len(content.split())

        logger.info(f"[GHOSTWRITER] Generated {word_count} words in {total_ms}ms")

        return {
            "content": content,
            "content_type": content_type.value,
            "word_count": word_count,
            "sources": sources,
            "confidence": confidence,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
            "metadata": {
                "topic": topic or request,
                "doc_type_filter": doc_type,
                "tone": tone,
                "chunks_used": len(chunks),
                "email_style_used": bool(style_context) if content_type == ContentType.EMAIL else None,
            },
        }

    async def _get_email_style_context(self, query: str) -> str:
        """
        Retrieve email style patterns from past email corpus.
        
        This retrieves similar emails and aggregates stylistic patterns
        to inject into the email prompt for consistency with Bunmi's style.
        
        Args:
            query: The search query (topic of the email being drafted)
            
        Returns:
            Formatted style context string for prompt injection
        """
        try:
            # Get similar emails from the corpus
            similar_emails = await self.email_style.get_similar_emails(
                query=query,
                limit=5,
            )
            
            if not similar_emails:
                logger.debug("[GHOSTWRITER] No similar emails found in corpus")
                return ""
            
            # Get aggregated style patterns
            patterns = await self.email_style.get_style_patterns()
            
            # Build the style context string
            style_context = self.email_style.build_style_context(
                similar_emails=similar_emails,
                patterns=patterns,
            )
            
            return style_context
            
        except Exception as e:
            logger.warning(f"[GHOSTWRITER] Email style retrieval failed: {e}")
            return ""

    async def index_email_example(
        self,
        subject: str,
        body: str,
        recipient_type: str = "colleague",
        email_purpose: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Index a sample email to train the email style learner.
        
        Args:
            subject: Email subject line
            body: Full email body text
            recipient_type: Type of recipient (colleague, client, partner, etc.)
            email_purpose: Purpose of email (follow-up, introduction, request, etc.)
            metadata: Additional metadata
            
        Returns:
            Dict with indexing result
        """
        try:
            result = await self.email_style.index_email(
                email_content=body,
                subject=subject,
                recipient_type=recipient_type,
            )
            return {
                "success": True,
                "email_id": result.get("email_id"),
                "patterns_extracted": result.get("patterns", {}),
            }
        except Exception as e:
            logger.error(f"[GHOSTWRITER] Failed to index email: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def index_email_batch(
        self,
        emails: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Index multiple sample emails in batch.
        
        Args:
            emails: List of email dicts with keys: subject, body, recipient_type, email_purpose
            
        Returns:
            Dict with batch indexing results
        """
        try:
            result = await self.email_style.index_emails_batch(emails)
            return {
                "success": True,
                "indexed_count": result.get("indexed_count", 0),
                "failed_count": result.get("failed_count", 0),
            }
        except Exception as e:
            logger.error(f"[GHOSTWRITER] Failed to batch index emails: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def refine(
        self,
        content: str,
        feedback: str,
        content_type: ContentType,
    ) -> Dict[str, Any]:
        """
        Refine generated content based on feedback.

        Args:
            content: Previously generated content
            feedback: User's feedback/refinement request
            content_type: Type of content being refined

        Returns:
            Dict with refined content
        """
        import time
        start_time = time.time()

        prompt = f"""You previously wrote this {content_type.value}:

---
{content}
---

USER FEEDBACK:
{feedback}

Please revise the content based on this feedback. Maintain DANI's voice - professional, direct, confident.
Output ONLY the revised content, nothing else:"""

        try:
            # Embed system instruction into prompt (OllamaClient.generate only takes prompt and stream)
            full_prompt = f"""SYSTEM: You are DANI's ghostwriter. Revise the content as requested. No explanations, no AI disclaimers. DO NOT use markdown formatting (no bold **, no headers #). Produce clean plain text.

{prompt}"""
            refined = await self.llm.generate(
                prompt=full_prompt,
            )
            refined = refined.strip()
            
        except Exception as e:
            logger.error(f"[GHOSTWRITER] Refinement failed: {e}")
            return {
                "error": f"Refinement failed: {str(e)}",
            }

        total_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "content": refined,
            "content_type": content_type.value,
            "word_count": len(refined.split()),
            "timing": {"total_ms": total_ms},
            "refined_from": content[:100] + "...",
            "feedback_applied": feedback,
        }

    def get_content_types(self) -> List[Dict[str, str]]:
        """Get list of supported content types with descriptions."""
        descriptions = {
            ContentType.LINKEDIN_POST: "Professional social media post for LinkedIn",
            ContentType.EMAIL: "Professional email draft",
            ContentType.BLOG_DRAFT: "Thought leadership blog post",
            ContentType.TWEET_THREAD: "Twitter/X thread (multiple tweets)",
            ContentType.NEWSLETTER: "Newsletter section or article",
            ContentType.MEETING_SUMMARY: "Executive summary of a meeting",
        }
        
        return [
            {"type": ct.value, "description": descriptions.get(ct, "")}
            for ct in ContentType
        ]


# Singleton instance
_ghostwriter: Optional[GhostwriterService] = None


def get_ghostwriter() -> GhostwriterService:
    """Get or create the ghostwriter singleton."""
    global _ghostwriter
    if _ghostwriter is None:
        _ghostwriter = GhostwriterService()
    return _ghostwriter

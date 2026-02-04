"""
Infographic Service.

Generates visual infographics by:
1. Using RAG to retrieve relevant context
2. Extracting structured data (headline, stats, facts)
3. Creating a visual prompt for image generation
4. Calling nano-banana MCP to generate the actual image
5. Persisting the infographic to S3 and database
"""

from __future__ import annotations

import logging
import time
import re
import json
import base64
import io
import os
import textwrap
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.llm.ollama import OllamaClient
from app.services.retrieval_service import RetrievalService
from app.services.storage_service import StorageService
from app.services.infographic_context import InfographicContextBuilder
from app.schemas.retrieval import MetadataFilter
from app.mcp.client import extract_all_content, extract_text
from app.utils.query_processor import normalize_relevance_score, get_relevance_label
from app.database.models.infographic import (
    Infographic as InfographicModel,
    InfographicStyle,
    InfographicStatus,
)
from app.core.config import settings

# Image generation dependencies
try:
    from google import genai as google_genai
    from PIL import Image, ImageDraw, ImageFont
    IMAGE_DEPS_AVAILABLE = True
except ImportError:
    IMAGE_DEPS_AVAILABLE = False
    logger.warning("Image generation dependencies not available - PIL and google-genai required")
from app.database.models.infographic import (
    Infographic as InfographicModel,
    InfographicStyle,
    InfographicStatus,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


# Style-specific visual prompts for image generation
# Updated: Force reload trigger
STYLE_PROMPTS = {
    InfographicStyle.MODERN: "modern sleek design, clean lines, gradient backgrounds, sans-serif typography, professional color palette with blue and white accents",
    InfographicStyle.CORPORATE: "corporate business style, professional navy and gray colors, formal typography, structured layout, executive presentation quality",
    InfographicStyle.MINIMAL: "minimalist design, lots of white space, simple icons, black and white with single accent color, clean typography",
    InfographicStyle.VIBRANT: "colorful vibrant design, bold colors, energetic layout, modern icons, dynamic visual elements, engaging and eye-catching",
    InfographicStyle.DARK: "dark theme infographic, dark background with bright accent colors, neon highlights, modern tech aesthetic, high contrast text",
}


# Prompt template for extracting structured infographic data
EXTRACTION_PROMPT = """Extract infographic data from the context below.

CONTEXT:
{context}

USER REQUEST: {request}

Return JSON with this EXACT structure:
{{
  "headline": "Short headline with a key number (5-8 words)",
  "subtitle": "Brief context (under 12 words)",
  "stats": [
    {{"value": "$X.XM", "label": "metric name", "icon": "💰"}},
    {{"value": "XX%", "label": "metric name", "icon": "📈"}},
    {{"value": "XX", "label": "metric name", "icon": "👥"}}
  ],
  "key_points": [
    "Key insight with a number",
    "Another insight with data"
  ],
  "source_summary": "Source name"
}}

RULES:
- Include 3-4 stats with numeric values
- Include 2-3 key points
- Use emojis: 💰📈🎯👥🌏📊🚀💡
- Output ONLY the JSON, no other text"""


class InfographicService:
    """
    Generates visual infographics using RAG + MCP image generation.
    
    Now supports full persistence:
    - Images stored in S3
    - Metadata stored in PostgreSQL
    """

    def __init__(self):
        self.llm = OllamaClient()
        self.retrieval = RetrievalService()
        self.storage = StorageService()
        self.context_builder = InfographicContextBuilder()
        self._mcp_registry = None

    @property
    def mcp_registry(self):
        """Lazy load MCP registry to avoid circular imports."""
        if self._mcp_registry is None:
            from app.mcp.registry import get_registry
            self._mcp_registry = get_registry()
        return self._mcp_registry

    async def generate(
        self,
        request: str,
        topic: Optional[str] = None,
        style: InfographicStyle = InfographicStyle.MODERN,
        doc_type: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        user_id: Optional[str] = None,
        db: Optional[AsyncSession] = None,
        persist: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a visual infographic.

        Args:
            request: User's request for what the infographic should show
            topic: Optional topic to search for in RAG
            style: Visual style for the infographic
            doc_type: Filter sources by type
            width: Image width in pixels
            height: Image height in pixels
            user_id: Optional user ID for ownership
            db: Database session for persistence
            persist: Whether to save to S3/database (default True)
            conversation_history: Optional list of previous messages in the conversation
                                  for better context understanding

        Returns:
            Dict with image data, structured content, and metadata
        """
        start_time = time.time()
        infographic_id = None
        
        logger.info(f"[INFOGRAPHIC] Generating infographic: {request[:100]}...")

        # Step 1: Retrieve relevant context using enhanced context builder
        retrieval_start = time.time()
        search_query = topic or request
        
        if doc_type == "meeting":
            # WORKAROUND: Legacy data in Qdrant is missing the 'doc_type' field.
            # Filtering by doc_type="meeting" returns 0 results.
            # specific to "meeting" serves no purpose if the collection is implicitly meetings.
            # We switch to "all" to bypass the filter.
            doc_type = "all"

        # Use enhanced context builder if conversation history is available
        if conversation_history and settings.ENHANCED_RETRIEVAL_ENABLED:
            enhanced_context = await self.context_builder.build_context(
                request=search_query,
                conversation_history=conversation_history,
                doc_type=doc_type if doc_type != "all" else None,
            )
            
            # Check for errors from context builder
            if "error" in enhanced_context:
                return {
                    "error": enhanced_context.get("error"),
                    "suggestion": "Try a different topic or broader search terms",
                }
            
            # Get structured data and sources from the enhanced context
            structured_data = enhanced_context.get("structured_data", {})
            raw_context = enhanced_context.get("raw_context", {})
            sources = enhanced_context.get("sources", [])
            chunks_used = enhanced_context.get("chunks_used", 0)
            retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)
            
            # Use raw context for any further processing
            context = raw_context.get("rag", "") + "\n\n" + raw_context.get("conversation", "")
            
            logger.info(f"[INFOGRAPHIC] Enhanced context built in {retrieval_ms}ms with {chunks_used} chunks")
            
            # If structured data already extracted, skip to image generation
            if structured_data and "error" not in structured_data:
                logger.info(f"[INFOGRAPHIC] Using pre-extracted structured data from enhanced context")
                
                # Step 4: Generate visual infographic via MCP
                image_start = time.time()
                image_result = await self._generate_image(
                    structured_data, 
                    style, 
                    width, 
                    height,
                    user_query=request,
                    context_summary=context,
                )
                image_ms = round((time.time() - image_start) * 1000, 2)

                if "error" in image_result:
                    result = {
                        "id": None,
                        "structured_data": structured_data,
                        "image": None,
                        "image_error": image_result["error"],
                        "sources": sources,
                        "timing": {
                            "retrieval_ms": retrieval_ms,
                            "extraction_ms": 0,  # Already extracted by context builder
                            "image_generation_ms": image_ms,
                            "total_ms": round((time.time() - start_time) * 1000, 2),
                        },
                    }
                else:
                    result = {
                        "id": None,
                        "structured_data": structured_data,
                        **image_result,
                        "sources": sources,
                        "timing": {
                            "retrieval_ms": retrieval_ms,
                            "extraction_ms": 0,
                            "image_generation_ms": image_ms,
                            "total_ms": round((time.time() - start_time) * 1000, 2),
                        },
                    }

                # Persist if requested
                if persist and db:
                    infographic_id = await self._persist_infographic(
                        db=db,
                        user_id=user_id,
                        request=request,
                        style=style,
                        structured_data=structured_data,
                        image_result=image_result,
                        sources=sources,
                        result=result,
                    )
                    result["id"] = infographic_id

                return result
        else:
            # Fallback to standard retrieval
            metadata_filter = None
            if doc_type and doc_type != "all":
                metadata_filter = MetadataFilter(
                    doc_type=doc_type,
                    speakers=None,
                    source_file=None,
                    date_from=None,
                    date_to=None,
                    transcript_id=None,
                )

            retrieval_result = await self.retrieval.search_with_confidence(
                query=search_query,
                limit=8,
                metadata_filter=metadata_filter,
            )
            chunks = retrieval_result["chunks"]
            confidence = retrieval_result["confidence"]
            retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)
            
            logger.info(f"[INFOGRAPHIC] Retrieved {len(chunks)} chunks in {retrieval_ms}ms")

            if not chunks:
                return {
                    "error": "No relevant context found for the infographic",
                    "suggestion": "Try a different topic or broader search terms",
                }

            # Step 2: Format context (standard flow)
            # IMPORTANT: Limit context size to prevent Ollama empty response
            MAX_CHUNKS = 5  # Use top 5 most relevant chunks
            MAX_CHUNK_TEXT = 1500  # Chars per chunk
            MAX_TOTAL_CONTEXT = 8000  # Total context chars
            
            context_parts = []
            sources = []
            
            for chunk in chunks[:MAX_CHUNKS]:
                title = chunk.get("title") or "Untitled"
                text = chunk.get("text", "")[:MAX_CHUNK_TEXT]  # Truncate chunk text
                date = chunk.get("date")
                raw_score = chunk.get("score", 0)
                
                context_parts.append(f"From '{title}' ({date or 'undated'}):\n{text}")
                sources.append({
                    "title": title,
                    "date": date,
                    "text_preview": text[:500],  # Preview is shorter
                    "score": raw_score,
                    "relevance_score": normalize_relevance_score(raw_score),  # Normalized 0-100%
                    "relevance_label": get_relevance_label(raw_score),  # Human-readable label
                    "meeting_category": chunk.get("meeting_category"),
                    "category_confidence": chunk.get("category_confidence"),
                })

            context = "\n\n---\n\n".join(context_parts)
            # Truncate total context if still too long
            if len(context) > MAX_TOTAL_CONTEXT:
                context = context[:MAX_TOTAL_CONTEXT] + "\n\n[Context truncated for processing]"
                logger.info(f"[INFOGRAPHIC] Context truncated to {MAX_TOTAL_CONTEXT} chars")

        # Step 3: Extract structured data
        extraction_start = time.time()
        structured_data = await self._extract_structured_data(context, request)
        extraction_ms = round((time.time() - extraction_start) * 1000, 2)
        
        if "error" in structured_data:
            return structured_data

        # Step 3.5: Validate and enhance quality
        structured_data = self._validate_and_enhance_quality(structured_data, chunks)
        
        logger.info(f"[INFOGRAPHIC] Extracted and enhanced structured data in {extraction_ms}ms")

        # Step 4: Generate visual infographic via MCP
        image_start = time.time()
        image_result = await self._generate_image(
            structured_data, 
            style, 
            width, 
            height,
            user_query=request,
            context_summary=context,
        )
        image_ms = round((time.time() - image_start) * 1000, 2)

        if "error" in image_result:
            # Return structured data even if image generation fails
            result = {
                "id": None,
                "structured_data": structured_data,
                "image": None,
                "image_error": image_result["error"],
                "sources": sources,
                "confidence": confidence,
                "timing": {
                    "retrieval_ms": retrieval_ms,
                    "extraction_ms": extraction_ms,
                    "image_ms": image_ms,
                    "total_ms": round((time.time() - start_time) * 1000, 2),
                },
                "metadata": {
                    "style": style.value,
                    "width": width,
                    "height": height,
                    "chunks_used": len(chunks),
                },
            }
            
            # Persist failed infographic if db session provided
            if persist and db:
                try:
                    infographic_id = await self._save_to_database(
                        db=db,
                        request=request,
                        topic=topic,
                        style=style,
                        width=width,
                        height=height,
                        user_id=user_id,
                        structured_data=structured_data,
                        sources=sources,
                        chunks_used=len(chunks),
                        confidence=confidence,
                        timing={
                            "retrieval_ms": retrieval_ms,
                            "extraction_ms": extraction_ms,
                            "image_ms": image_ms,
                            "total_ms": round((time.time() - start_time) * 1000, 2),
                        },
                        status=InfographicStatus.FAILED,
                        error_message=image_result["error"],
                    )
                    result["id"] = infographic_id
                except Exception as e:
                    logger.error(f"[INFOGRAPHIC] Failed to save error state: {e}")
            
            return result

        total_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[INFOGRAPHIC] Generated infographic in {total_ms}ms")

        # Step 5: Persist to S3 and database
        logger.info(f"[INFOGRAPHIC] image_result keys: {list(image_result.keys())}")
        logger.info(f"[INFOGRAPHIC] image_result.get('image') exists: {bool(image_result.get('image'))}")
        logger.info(f"[INFOGRAPHIC] persist={persist}, user_id={user_id}")
        
        s3_key = None
        s3_url = None
        image_size = None
        presigned_url = None
        
        if persist and image_result.get("image"):
            try:
                logger.info(f"[INFOGRAPHIC] Starting S3 upload...")
                # Decode base64 image
                image_bytes = base64.b64decode(image_result["image"])
                image_size = len(image_bytes)
                logger.info(f"[INFOGRAPHIC] Image decoded: {image_size} bytes")
                
                # Sanitize headline for S3 metadata (ASCII only)
                raw_headline = structured_data.get("headline", "")[:100]
                # Remove non-ASCII characters (including emojis)
                sanitized_headline = raw_headline.encode('ascii', 'ignore').decode('ascii')
                
                # Upload to S3
                s3_result = await self.storage.upload(
                    file_content=image_bytes,
                    filename=f"infographic_{int(time.time())}.png",
                    user_id=user_id,
                    content_type="image/png",
                    metadata={
                        "type": "infographic",
                        "style": style.value,
                        "headline": sanitized_headline,
                    },
                )
                s3_key = s3_result["key"]
                s3_url = s3_result["url"]
                logger.info(f"[INFOGRAPHIC] S3 upload complete: {s3_key}")
                
                # Generate presigned URL for browser access
                try:
                    presigned_url = await self.storage.get_presigned_url(
                        s3_key,
                        expiry=604800,  # 7 days (AWS S3 max)
                    )
                    logger.info(f"[INFOGRAPHIC] Generated presigned URL for {s3_key}")
                except Exception as e:
                    logger.error(f"[INFOGRAPHIC] Failed to generate presigned URL: {e}")
                    presigned_url = s3_url  # Fallback to regular URL
                
                logger.info(f"[INFOGRAPHIC] Uploaded to S3: {s3_key}")
            except Exception as e:
                logger.error(f"[INFOGRAPHIC] S3 upload failed: {e}", exc_info=True)
                presigned_url = None
                # Continue without persistence
        else:
            logger.warning(f"[INFOGRAPHIC] S3 upload SKIPPED: persist={persist}, has_image={bool(image_result.get('image'))}")
        
        # Save to database
        if persist and db:
            try:
                infographic_id = await self._save_to_database(
                    db=db,
                    request=request,
                    topic=topic,
                    style=style,
                    width=width,
                    height=height,
                    user_id=user_id,
                    structured_data=structured_data,
                    sources=sources,
                    chunks_used=len(chunks),
                    confidence=confidence,
                    timing={
                        "retrieval_ms": retrieval_ms,
                        "extraction_ms": extraction_ms,
                        "image_ms": image_ms,
                        "total_ms": total_ms,
                    },
                    status=InfographicStatus.COMPLETED,
                    s3_key=s3_key,
                    s3_url=s3_url,
                    image_size=image_size,
                )
                logger.info(f"[INFOGRAPHIC] Saved to database: {infographic_id}")
            except Exception as e:
                logger.error(f"[INFOGRAPHIC] Database save failed: {e}")
                infographic_id = None

        return {
            "id": infographic_id,
            "structured_data": structured_data,
            "image": image_result.get("image"),
            "image_url": presigned_url or image_result.get("url"),
            "s3_key": s3_key,
            "image_format": image_result.get("format", "png"),
            "sources": sources,
            "confidence": confidence,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "extraction_ms": extraction_ms,
                "image_ms": image_ms,
                "total_ms": total_ms,
            },
            "metadata": {
                "style": style.value,
                "width": width,
                "height": height,
                "chunks_used": len(chunks),
            },
        }

    async def _persist_infographic(
        self,
        db: AsyncSession,
        user_id: Optional[str],
        request: str,
        style: InfographicStyle,
        structured_data: Dict[str, Any],
        image_result: Dict[str, Any],
        sources: List[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> Optional[str]:
        """
        Persist infographic to S3 and database (used by enhanced context path).
        
        Args:
            db: Database session
            user_id: User ID for ownership
            request: Original request text
            style: Infographic style
            structured_data: Extracted structured data
            image_result: Image generation result with base64 data
            sources: Source references
            result: Result dict to update with s3_key and image_url
            
        Returns:
            Infographic ID if saved successfully, None otherwise
        """
        logger.info(f"[INFOGRAPHIC] _persist_infographic called with image_result keys: {list(image_result.keys())}")
        logger.info(f"[INFOGRAPHIC] image_result has 'image': {bool(image_result.get('image'))}")
        
        s3_key = None
        s3_url = None
        presigned_url = None
        image_size = None
        
        # Upload image to S3 if we have image data
        if image_result.get("image"):
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_result["image"])
                image_size = len(image_bytes)
                
                # Sanitize headline for S3 metadata (ASCII only)
                raw_headline = structured_data.get("headline", "")[:100]
                # Remove non-ASCII characters (including emojis)
                sanitized_headline = raw_headline.encode('ascii', 'ignore').decode('ascii')
                
                # Upload to S3
                s3_result = await self.storage.upload(
                    file_content=image_bytes,
                    filename=f"infographic_{int(time.time())}.png",
                    user_id=user_id,
                    content_type="image/png",
                    metadata={
                        "type": "infographic",
                        "style": style.value,
                        "headline": sanitized_headline,
                    },
                )
                s3_key = s3_result["key"]
                s3_url = s3_result["url"]
                
                # Generate presigned URL for browser access
                try:
                    presigned_url = await self.storage.get_presigned_url(
                        s3_key,
                        expiry=604800,  # 7 days (AWS S3 max)
                    )
                    logger.info(f"[INFOGRAPHIC] Generated presigned URL for {s3_key}")
                except Exception as e:
                    logger.error(f"[INFOGRAPHIC] Failed to generate presigned URL: {e}")
                    presigned_url = s3_url  # Fallback to regular URL
                
                logger.info(f"[INFOGRAPHIC] Uploaded to S3: {s3_key}")
                
                # Update result dict with S3 info
                result["s3_key"] = s3_key
                result["image_url"] = presigned_url or image_result.get("url")
                
            except Exception as e:
                logger.error(f"[INFOGRAPHIC] S3 upload failed: {e}")
                # Continue without S3 persistence
        
        # Save to database
        try:
            timing = result.get("timing", {})
            infographic_id = await self._save_to_database(
                db=db,
                request=request,
                topic=None,  # Topic not always available in enhanced path
                style=style,
                width=1024,
                height=1024,
                user_id=user_id,
                structured_data=structured_data,
                sources=sources,
                chunks_used=len(sources),
                confidence=0.8,  # Default confidence for enhanced path
                timing=timing,
                status=InfographicStatus.COMPLETED,
                s3_key=s3_key,
                s3_url=s3_url,
                image_size=image_size,
            )
            logger.info(f"[INFOGRAPHIC] Saved to database: {infographic_id}")
            return infographic_id
        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Database save failed: {e}")
            return None

    async def _extract_structured_data(
        self, 
        context: str, 
        request: str
    ) -> Dict[str, Any]:
        """Extract structured infographic data from context using LLM."""
        # Log context size for debugging
        logger.info(f"[INFOGRAPHIC] Context size: {len(context)} chars, request: {len(request)} chars")
        
        # If context is still too large, truncate more aggressively
        MAX_CONTEXT_FOR_EXTRACTION = 6000
        if len(context) > MAX_CONTEXT_FOR_EXTRACTION:
            context = context[:MAX_CONTEXT_FOR_EXTRACTION] + "\n[...truncated]"
            logger.warning(f"[INFOGRAPHIC] Context truncated to {MAX_CONTEXT_FOR_EXTRACTION} chars for extraction")
        
        prompt = EXTRACTION_PROMPT.format(context=context, request=request)
        logger.info(f"[INFOGRAPHIC] Extraction prompt size: {len(prompt)} chars")
        
        try:
            # Use higher num_predict to ensure complete JSON response
            # The extraction response needs ~800-1000 tokens for complete JSON
            response = await self.llm.generate(
                prompt=prompt,
                options={"num_predict": 1024, "temperature": 0.3}
            )
            
            # Clean and parse JSON
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?\n?", "", response)
                response = response.rstrip("`").strip()
            
            # Find JSON object by looking for { and }
            # This handles cases where LLM adds text before/after JSON
            json_start = response.find('{')
            if json_start != -1:
                json_end = response.rfind('}')
                if json_end != -1 and json_end > json_start:
                    response = response[json_start:json_end+1]
            
            # Try to fix common JSON errors from LLMs
            data = self._parse_json_with_recovery(response)
            
            # Validate required fields
            required = ["headline", "stats"]
            for field in required:
                if field not in data:
                    return {"error": f"Missing required field: {field}"}
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"[INFOGRAPHIC] JSON parse error: {e}")
            logger.error(f"[INFOGRAPHIC] Response was: {response[:500]}")
            return {"error": f"Failed to parse structured data: {str(e)}"}
        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Extraction failed: {e}")
            return {"error": f"Data extraction failed: {str(e)}"}
    
    def _parse_json_with_recovery(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON with automatic recovery from common LLM formatting errors.
        
        Handles:
        - Missing commas between elements
        - Trailing commas
        - Single quotes instead of double quotes
        - Comments (// style)
        - Unquoted keys
        """
        # First, try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Apply fixes iteratively
        fixed = response
        
        # Remove comments (// style)
        fixed = re.sub(r'//.*?(?=\n|$)', '', fixed)
        
        # Replace single quotes with double quotes (but not inside strings)
        # This is a simplified approach - handles most cases
        fixed = re.sub(r"(?<![a-zA-Z])'([^']*)'(?![a-zA-Z])", r'"\1"', fixed)
        
        # Fix missing commas between array elements or object properties
        # Pattern: }\s*{ or ]\s*[ or }\s*" or ]\s*" or "\s*{ or "\s*[
        fixed = re.sub(r'(\})\s*(\{)', r'\1,\2', fixed)
        fixed = re.sub(r'(\])\s*(\[)', r'\1,\2', fixed)
        fixed = re.sub(r'(\})\s*(")', r'\1,\2', fixed)
        fixed = re.sub(r'(\])\s*(")', r'\1,\2', fixed)
        fixed = re.sub(r'(")\s*(\{)', r'\1,\2', fixed)
        fixed = re.sub(r'(")\s*(\[)', r'\1,\2', fixed)
        
        # Fix missing commas after quoted strings followed by quoted keys
        # "value"\n    "key" -> "value",\n    "key"
        fixed = re.sub(r'(")\s*\n(\s*")', r'\1,\n\2', fixed)
        
        # Remove trailing commas before ] or }
        fixed = re.sub(r',(\s*[\}\]])', r'\1', fixed)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Last resort: try to extract key-value pairs manually for a minimal valid structure
        logger.warning("[INFOGRAPHIC] Attempting manual JSON reconstruction")
        logger.info(f"[INFOGRAPHIC] Raw response length: {len(response)} chars")
        
        # Extract headline
        headline_match = re.search(r'"headline"\s*:\s*"([^"]*)"', response)
        headline = headline_match.group(1) if headline_match else "Data Summary"
        
        # Extract subtitle
        subtitle_match = re.search(r'"subtitle"\s*:\s*"([^"]*)"', response)
        subtitle = subtitle_match.group(1) if subtitle_match else ""
        
        # Extract stats array - handle both complete and truncated responses
        stats = []
        
        # Try full stat pattern first (with icon)
        stats_matches = re.findall(
            r'\{\s*"value"\s*:\s*"([^"]*)"\s*,\s*"label"\s*:\s*"([^"]*)"\s*(?:,\s*"icon"\s*:\s*"([^"]*)")?\s*\}',
            response
        )
        for match in stats_matches:
            stat = {"value": match[0], "label": match[1]}
            if match[2]:
                stat["icon"] = match[2]
            stats.append(stat)
        
        # If no complete stats found, try to extract partial stats from truncated JSON
        if not stats:
            # Look for value/label pairs even if not in complete object
            partial_matches = re.findall(r'"value"\s*:\s*"([^"]*)"\s*,\s*"label"\s*:\s*"([^"]*)', response)
            for match in partial_matches:
                stats.append({"value": match[0], "label": match[1], "icon": "📊"})
        
        # Extract key points
        key_points = []
        kp_match = re.search(r'"key_points"\s*:\s*\[([^\]]*)\]?', response, re.DOTALL)
        if kp_match:
            kp_text = kp_match.group(1)
            kp_items = re.findall(r'"([^"]*)"', kp_text)
            key_points = kp_items[:4]  # Max 4 points
        
        if not stats:
            # If still no stats found, try one more pattern for truncated responses
            # Look for any "$X" or "X%" values
            value_matches = re.findall(r'([\$€£]?[\d,.]+[%BMKx]?)', response)
            if value_matches:
                for i, val in enumerate(value_matches[:4]):
                    stats.append({"value": val, "label": f"Metric {i+1}", "icon": "📊"})
        
        if not stats:
            # If no stats found at all, raise the original error
            raise json.JSONDecodeError("Could not extract valid JSON structure", response, 0)
        
        logger.info(f"[INFOGRAPHIC] Recovered {len(stats)} stats from truncated response")
        
        # Build minimal valid structure
        result = {
            "headline": headline,
            "subtitle": subtitle,
            "stats": stats,
            "key_points": key_points if key_points else ["See detailed data below"]
        }
        
        logger.info(f"[INFOGRAPHIC] Reconstructed JSON with headline='{headline}', {len(stats)} stats")
        return result

    async def _generate_image(
        self,
        structured_data: Dict[str, Any],
        style: InfographicStyle,
        width: int,
        height: int,
        user_query: Optional[str] = None,
        context_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate visual infographic image via direct Gemini API or MCP fallback.
        
        Args:
            structured_data: Extracted headline, stats, key_points etc.
            style: Visual style for the infographic
            width: Image width
            height: Image height
            user_query: Original user request for context
            context_summary: Summary of retrieved chunks for richer prompt
        """
        
        # Build the image generation prompt with full context
        prompt = self._build_image_prompt(
            structured_data, 
            style, 
            user_query=user_query, 
            context_summary=context_summary
        )
        
        try:
            # Try direct Gemini API first (bypasses broken nano-banana MCP)
            logger.info("[INFOGRAPHIC] Attempting direct Gemini image generation")
            image_data = self._generate_image_direct(prompt, width=width, height=height)
            
            logger.info(f"[INFOGRAPHIC] _generate_image_direct returned: {type(image_data)}, len={len(image_data) if image_data else 0}")
            
            if image_data:
                # Convert bytes to base64 for consistency with existing code
                base64_data = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"[INFOGRAPHIC] Returning image dict with base64 data len={len(base64_data)}")
                return {
                    "image": base64_data,
                    "format": "png",  # Gemini typically returns PNG
                }
            
            logger.warning("[INFOGRAPHIC] Direct Gemini generation failed, trying MCP fallback")
            
            # Fallback to MCP if direct API fails
            registry = self.mcp_registry
            if not registry or "nano-banana" not in registry.connected_servers:
                logger.warning("[INFOGRAPHIC] nano-banana MCP not connected")
                return {"error": "Image generation service not available"}

            # Call nano-banana's generate_image tool
            result = await registry.call_tool(
                server_name="nano-banana",
                tool_name="generate_image",
                arguments={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                },
            )

            # MCPToolResult has is_error flag, use extract helpers for content
            if result.is_error:
                error_text = extract_text(result)
                return {"error": error_text or "Image generation failed"}

            # Try to extract image first (most common response type)
            from app.mcp.client import extract_image
            image_data = extract_image(result)
            if image_data:
                base64_data, mime_type = image_data
                return {
                    "image": base64_data,
                    "format": mime_type.split("/")[-1] if "/" in mime_type else "png",
                }
            
            # Fallback: try to extract text (might be JSON or URL)
            text_content = extract_text(result)
            if text_content:
                # Try to parse as JSON
                try:
                    data = json.loads(text_content)
                    if isinstance(data, dict):
                        return {
                            "image": data.get("image") or data.get("data"),
                            "url": data.get("url"),
                            "format": data.get("format", "png"),
                        }
                except json.JSONDecodeError:
                    pass
                
                # Could be a URL
                if text_content.startswith("http"):
                    return {"url": text_content, "format": "png"}
                # Could be raw base64
                return {"image": text_content, "format": "png"}
            
            return {"error": "No image content in response"}

        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Image generation failed: {e}")
            return {"error": f"Image generation failed: {str(e)}"}

    def _generate_image_direct(self, prompt: str, width: int = 1024, height: int = 768) -> Optional[bytes]:
        """
        Generate an image using Google Imagen API directly, with PIL fallback.

        Args:
            prompt: The text prompt for image generation
            width: Desired image width
            height: Desired image height

        Returns:
            Image data as bytes, or None if generation fails
        """
        if not IMAGE_DEPS_AVAILABLE:
            logger.warning("Image generation dependencies not available")
            return self._generate_placeholder_image(prompt, width, height)

        # Try Gemini image generation models
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("GEMINI_API_KEY not found - will use PIL fallback for image generation")
            return self._generate_placeholder_image(prompt, width, height)

        # Keep prompt simple - don't add more text, it slows generation
        enhanced_prompt = prompt
        logger.info(f"[INFOGRAPHIC] Prompt length: {len(prompt)} chars")

        # Models to try in order of preference
        image_models = [
            "gemini-2.5-flash-image",      # Nano Banana - fast, good quality
            "gemini-3-pro-image-preview",  # Nano Banana Pro - higher quality
        ]
        
        try:
            client = google_genai.Client(api_key=api_key)
            logger.info("[INFOGRAPHIC] Gemini client initialized successfully")
            
            for model_name in image_models:
                try:
                    logger.info(f"[INFOGRAPHIC] Generating image with {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=enhanced_prompt,
                    )
                    
                    logger.info(f"[INFOGRAPHIC] Response received from {model_name}: {type(response)}")
                    
                    if response and response.candidates:
                        logger.info(f"[INFOGRAPHIC] Found {len(response.candidates)} candidates")
                        for candidate in response.candidates:
                            if candidate.content and candidate.content.parts:
                                logger.info(f"[INFOGRAPHIC] Found {len(candidate.content.parts)} parts")
                                for part in candidate.content.parts:
                                    logger.info(f"[INFOGRAPHIC] Part type: {type(part)}, has inline_data: {hasattr(part, 'inline_data')}")
                                    if hasattr(part, 'inline_data') and part.inline_data:
                                        # Get image data
                                        data = part.inline_data.data
                                        if isinstance(data, str):
                                            data = base64.b64decode(data)
                                        logger.info(f"[INFOGRAPHIC] Image generated successfully with {model_name} ({len(data)} bytes)")
                                        return data
                    else:
                        logger.warning(f"[INFOGRAPHIC] {model_name} response has no candidates: {response}")
                    
                    logger.warning(f"[INFOGRAPHIC] {model_name} did not return image data")
                    
                except Exception as model_e:
                    logger.warning(f"[INFOGRAPHIC] {model_name} failed: {model_e}", exc_info=True)
                    continue
            
            logger.warning("[INFOGRAPHIC] All Gemini image models failed, using PIL fallback")
            
        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Gemini client initialization failed: {e}", exc_info=True)

        # Fallback to PIL-based image generation
        logger.info("[INFOGRAPHIC] Using PIL fallback for image generation")
        pil_result = self._generate_placeholder_image(prompt, width, height)
        logger.info(f"[INFOGRAPHIC] PIL fallback returned: {type(pil_result)}, len={len(pil_result) if pil_result else 0}")
        return pil_result

    def _generate_placeholder_image(self, prompt: str, width: int = 1024, height: int = 768) -> Optional[bytes]:
        """
        Generate a placeholder infographic image using PIL.

        Args:
            prompt: Text prompt (used for content)
            width: Image width
            height: Image height

        Returns:
            Image data as bytes
        """
        logger.info(f"[INFOGRAPHIC] _generate_placeholder_image called with width={width}, height={height}")
        try:
            # Create a new image with a professional color scheme
            img = Image.new('RGB', (width, height), color='#f8f9fa')
            draw = ImageDraw.Draw(img)
            logger.info("[INFOGRAPHIC] PIL Image created successfully")

            # Try to use a default font, fallback to basic if not available
            try:
                font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 48)
                font_body = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
                logger.info("[INFOGRAPHIC] Using Arial font")
            except:
                font_title = ImageFont.load_default()
                font_body = ImageFont.load_default()
                logger.info("[INFOGRAPHIC] Using default font")

            # Colors
            primary_color = '#1a73e8'  # Google Blue
            secondary_color = '#34a853'  # Green
            text_color = '#202124'  # Dark gray
            accent_color = '#ea4335'  # Red

            # Draw header bar
            draw.rectangle([0, 0, width, 120], fill=primary_color)

            # Title
            title_text = "Infographic"
            draw.text((width//2, 60), title_text, fill='white', font=font_title, anchor='mm')

            # Extract key information from prompt for content
            lines = textwrap.wrap(prompt[:200] + "..." if len(prompt) > 200 else prompt, width=60)

            # Draw content boxes
            y_offset = 140
            for i, line in enumerate(lines[:8]):  # Limit to 8 lines
                if y_offset > height - 100:
                    break

                # Alternate background colors
                bg_color = '#ffffff' if i % 2 == 0 else '#f1f3f4'
                draw.rectangle([50, y_offset, width-50, y_offset+40], fill=bg_color, outline=primary_color, width=1)
                draw.text((60, y_offset+20), line, fill=text_color, font=font_body, anchor='lm')
                y_offset += 45

            # Add some visual elements
            # Circle
            draw.ellipse([width-150, height-150, width-50, height-50], fill=secondary_color, outline=primary_color, width=3)

            # Triangle
            draw.polygon([(100, height-100), (150, height-150), (150, height-50)], fill=accent_color)

            # Footer
            draw.rectangle([0, height-40, width, height], fill=primary_color)
            footer_text = "Generated Infographic"
            draw.text((width//2, height-20), footer_text, fill='white', font=font_body, anchor='mm')

            # Convert to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            result = buffer.getvalue()
            logger.info(f"[INFOGRAPHIC] PIL placeholder image created: {len(result)} bytes")
            return result

        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Failed to generate placeholder image: {e}", exc_info=True)
            return None

    def _validate_and_enhance_quality(
        self,
        structured_data: Dict[str, Any],
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate and enhance structured data quality.
        
        Fixes common issues:
        1. Generic headlines without numbers
        2. Stats with vague values like 'High', 'Good'
        3. Missing icons on stats
        4. Key points without specific data
        
        Returns enhanced structured data.
        """
        enhanced = structured_data.copy()
        
        # Extract all numbers from context for reference
        context_text = " ".join([c.get("text", "") for c in context_chunks])
        context_numbers = re.findall(
            r'\$[\d,.]+[MBK]?|\d+(?:\.\d+)?%|\d+(?:\.\d+)?x|\d{1,3}(?:,\d{3})*(?:\.\d+)?',
            context_text
        )
        
        # Fix headline if too generic
        headline = enhanced.get("headline", "")
        generic_words = ["summary", "overview", "report", "information", "update", "meeting"]
        if any(word in headline.lower() for word in generic_words) or len(headline.split()) < 4:
            # Try to make headline more impactful with numbers
            if context_numbers:
                top_numbers = context_numbers[:2]
                if "%" in str(top_numbers):
                    enhanced["headline"] = f"Key Metrics: {top_numbers[0]} Achievement"
                elif "$" in str(top_numbers):
                    enhanced["headline"] = f"Revenue Highlights: {top_numbers[0]}"
                else:
                    enhanced["headline"] = f"Performance Summary: {top_numbers[0]} Results"
        
        # Fix stats - ensure numeric values and icons
        stats = enhanced.get("stats", [])
        icon_defaults = ["💰", "📈", "🎯", "👥", "🌏", "📊", "🚀", "💡", "✅", "🏆"]
        vague_values = ["high", "good", "strong", "low", "improved", "increased", "significant"]
        
        fixed_stats = []
        used_numbers = set()
        
        for i, stat in enumerate(stats):
            value = str(stat.get("value", "")).strip()
            label = stat.get("label", "Metric")
            icon = stat.get("icon", "")
            
            # Fix vague values
            if value.lower() in vague_values or not re.search(r'\d', value):
                # Try to find an unused number from context
                for num in context_numbers:
                    if num not in used_numbers:
                        value = num
                        used_numbers.add(num)
                        break
            else:
                used_numbers.add(value)
            
            # Add icon if missing
            if not icon or icon.strip() == "":
                icon = icon_defaults[i % len(icon_defaults)]
            
            fixed_stats.append({
                "value": value,
                "label": label,
                "icon": icon,
                **{k: v for k, v in stat.items() if k not in ["value", "label", "icon"]}
            })
        
        enhanced["stats"] = fixed_stats
        
        # Fix key points - ensure they have specifics
        key_points = enhanced.get("key_points", [])
        fixed_points = []
        
        for point in key_points:
            point_str = str(point)
            # If point is too vague, try to add context
            if len(point_str.split()) < 5:
                point_str = f"{point_str} based on recent data"
            fixed_points.append(point_str)
        
        enhanced["key_points"] = fixed_points
        
        return enhanced

    def _build_image_prompt(
        self, 
        structured_data: Dict[str, Any], 
        style: InfographicStyle,
        user_query: Optional[str] = None,
        context_summary: Optional[str] = None,
    ) -> str:
        """
        Build a SIMPLIFIED prompt for image generation.
        
        IMPORTANT: Gemini image models work best with SHORT, focused prompts.
        Long prompts with too much text cause slow generation and poor results.
        
        Args:
            structured_data: Extracted headline, stats, key_points etc.
            style: Visual style for the infographic
            user_query: Original user request/query for context (not used - too verbose)
            context_summary: Summary of retrieved chunks (not used - too verbose)
        """
        
        headline = structured_data.get("headline", "Infographic")
        stats = structured_data.get("stats", [])
        
        # Format only TOP 4 stats as simple text
        stats_text = ", ".join([
            f"{s.get('value')} {s.get('label')}"
            for s in stats[:4]
        ])
        
        # Get style-specific visual instructions (shortened)
        style_short = {
            InfographicStyle.MODERN: "modern, clean, gradient blue",
            InfographicStyle.CORPORATE: "corporate, navy and gray, professional",
            InfographicStyle.MINIMAL: "minimalist, white space, simple",
            InfographicStyle.VIBRANT: "colorful, bold, energetic",
            InfographicStyle.DARK: "dark theme, neon accents, tech style",
        }
        style_desc = style_short.get(style, "modern professional")
        
        # SIMPLIFIED PROMPT - much shorter for faster generation
        prompt = f"""Create a {style_desc} infographic titled "{headline}".

Show these stats visually with icons and charts: {stats_text}

Style: Clean data visualization, no photos of people, high contrast text."""

        logger.info(f"[INFOGRAPHIC] Image prompt length: {len(prompt)} chars")
        return prompt
    
    async def _save_to_database(
        self,
        db: AsyncSession,
        request: str,
        topic: Optional[str],
        style: InfographicStyle,
        width: int,
        height: int,
        user_id: Optional[str],
        structured_data: Dict[str, Any],
        sources: List[Dict[str, Any]],
        chunks_used: int,
        confidence: Dict[str, Any],
        timing: Dict[str, float],
        status: InfographicStatus,
        s3_key: Optional[str] = None,
        s3_url: Optional[str] = None,
        image_size: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Save infographic metadata to database.
        
        Returns the generated infographic ID.
        """
        infographic = InfographicModel(
            user_id=user_id,
            request=request,
            topic=topic,
            style=style,
            width=width,
            height=height,
            headline=structured_data.get("headline"),
            subtitle=structured_data.get("subtitle"),
            structured_data=structured_data,
            s3_key=s3_key,
            s3_bucket=settings.S3_BUCKET_NAME if s3_key else None,
            image_url=s3_url,
            image_format="png",
            image_size_bytes=image_size,
            sources=sources,
            chunks_used=chunks_used,
            confidence_score=confidence.get("score"),
            confidence_level=confidence.get("level"),
            retrieval_ms=timing.get("retrieval_ms"),
            extraction_ms=timing.get("extraction_ms"),
            image_gen_ms=timing.get("image_ms"),
            total_ms=timing.get("total_ms"),
            status=status,
            error_message=error_message,
        )
        
        db.add(infographic)
        await db.commit()
        await db.refresh(infographic)
        
        return infographic.id
    
    async def get_by_id(
        self,
        infographic_id: str,
        db: AsyncSession,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an infographic by ID.
        
        Args:
            infographic_id: The infographic ID
            db: Database session
            user_id: Optional user ID to filter by (for authorization)
            
        Returns:
            Infographic data dict or None if not found
        """
        query = select(InfographicModel).where(
            InfographicModel.id == infographic_id,
            InfographicModel.deleted_at.is_(None),
        )
        
        if user_id:
            query = query.where(InfographicModel.user_id == user_id)
        
        result = await db.execute(query)
        infographic = result.scalar_one_or_none()
        
        if not infographic:
            return None
        
        return self._model_to_dict(infographic)
    
    async def list_by_user(
        self,
        user_id: str,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0,
        status: Optional[InfographicStatus] = None,
    ) -> List[Dict[str, Any]]:
        """
        List infographics for a user.
        
        Args:
            user_id: The user ID
            db: Database session
            limit: Maximum results to return
            offset: Pagination offset
            status: Optional status filter
            
        Returns:
            List of infographic data dicts
        """
        query = select(InfographicModel).where(
            InfographicModel.user_id == user_id,
            InfographicModel.deleted_at.is_(None),
        )
        
        if status:
            query = query.where(InfographicModel.status == status)
        
        query = query.order_by(desc(InfographicModel.created_at))
        query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        infographics = result.scalars().all()
        
        return [self._model_to_dict(i) for i in infographics]
    
    async def delete(
        self,
        infographic_id: str,
        db: AsyncSession,
        user_id: Optional[str] = None,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete an infographic (soft delete by default).
        
        Args:
            infographic_id: The infographic ID
            db: Database session
            user_id: Optional user ID for authorization
            hard_delete: If True, permanently delete (including S3)
            
        Returns:
            True if deleted, False if not found
        """
        query = select(InfographicModel).where(
            InfographicModel.id == infographic_id,
        )
        
        if user_id:
            query = query.where(InfographicModel.user_id == user_id)
        
        result = await db.execute(query)
        infographic = result.scalar_one_or_none()
        
        if not infographic:
            return False
        
        if hard_delete:
            # Delete from S3
            if infographic.s3_key:
                try:
                    await self.storage.delete(infographic.s3_key)
                except Exception as e:
                    logger.error(f"[INFOGRAPHIC] S3 delete failed: {e}")
            
            await db.delete(infographic)
        else:
            # Soft delete
            from datetime import datetime, timezone
            infographic.deleted_at = datetime.now(timezone.utc)
        
        await db.commit()
        return True
    
    async def get_presigned_url(
        self,
        infographic_id: str,
        db: AsyncSession,
        user_id: Optional[str] = None,
        expiry_seconds: int = 3600,
    ) -> Optional[str]:
        """
        Get a presigned URL for downloading the infographic image.
        
        Args:
            infographic_id: The infographic ID
            db: Database session
            user_id: Optional user ID for authorization
            expiry_seconds: URL expiry time in seconds
            
        Returns:
            Presigned URL or None if not found
        """
        query = select(InfographicModel).where(
            InfographicModel.id == infographic_id,
            InfographicModel.deleted_at.is_(None),
        )
        
        if user_id:
            query = query.where(InfographicModel.user_id == user_id)
        
        result = await db.execute(query)
        infographic = result.scalar_one_or_none()
        
        if not infographic or not infographic.s3_key:
            return None
        
        try:
            return await self.storage.get_presigned_url(
                infographic.s3_key,
                expiry=expiry_seconds,
            )
        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Failed to generate presigned URL: {e}")
            return None
    
    def _model_to_dict(self, infographic: InfographicModel) -> Dict[str, Any]:
        """Convert database model to response dict."""
        return {
            "id": infographic.id,
            "user_id": infographic.user_id,
            "request": infographic.request,
            "topic": infographic.topic,
            "style": infographic.style.value if infographic.style else None,
            "width": infographic.width,
            "height": infographic.height,
            "headline": infographic.headline,
            "subtitle": infographic.subtitle,
            "structured_data": infographic.structured_data,
            "s3_key": infographic.s3_key,
            "image_url": infographic.image_url,
            "image_format": infographic.image_format,
            "image_size_bytes": infographic.image_size_bytes,
            "sources": infographic.sources,
            "chunks_used": infographic.chunks_used,
            "confidence": {
                "score": infographic.confidence_score,
                "level": infographic.confidence_level,
            },
            "timing": {
                "retrieval_ms": infographic.retrieval_ms,
                "extraction_ms": infographic.extraction_ms,
                "image_ms": infographic.image_gen_ms,
                "total_ms": infographic.total_ms,
            },
            "status": infographic.status.value if infographic.status else None,
            "error_message": infographic.error_message,
            "created_at": infographic.created_at.isoformat() if infographic.created_at else None,
            "updated_at": infographic.updated_at.isoformat() if infographic.updated_at else None,
        }

    def get_styles(self) -> List[Dict[str, str]]:
        """Get list of supported infographic styles."""
        return [
            {"style": style.value, "description": STYLE_PROMPTS[style][:100] + "..."}
            for style in InfographicStyle
        ]

    async def generate_schema_only(
        self,
        request: str,
        topic: Optional[str] = None,
        style: InfographicStyle = InfographicStyle.MODERN,
        doc_type: Optional[str] = None,
        user_id: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Generate only the structured schema without image generation.
        
        This is faster and cheaper when you only need the spec for
        external rendering (e.g., Claude, Gemini).
        
        Args:
            request: User's request for what the infographic should show
            topic: Optional topic to search for in RAG
            style: Visual style for the infographic
            doc_type: Filter sources by type
            user_id: Optional user ID for ownership
            db: Database session for persistence
            
        Returns:
            Dict with structured data, sources, and metadata
        """
        start_time = time.time()
        
        logger.info(f"[INFOGRAPHIC] Generating schema only: {request[:100]}...")

        # Step 1: Retrieve relevant context
        retrieval_start = time.time()
        search_query = topic or request
        
        metadata_filter = None
        if doc_type and doc_type != "all":
            metadata_filter = MetadataFilter(
                doc_type=doc_type,
                speakers=None,
                source_file=None,
                date_from=None,
                date_to=None,
                transcript_id=None,
            )

        retrieval_result = await self.retrieval.search_with_confidence(
            query=search_query,
            limit=8,
            metadata_filter=metadata_filter,
        )
        chunks = retrieval_result["chunks"]
        confidence = retrieval_result["confidence"]
        retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

        if not chunks:
            return {
                "error": "No relevant context found for the infographic",
                "suggestion": "Try a different topic or broader search terms",
            }

        # Step 2: Format context
        context_parts = []
        sources = []
        
        for chunk in chunks:
            title = chunk.get("title", "Untitled")
            text = chunk.get("text", "")
            date = chunk.get("date")
            
            context_parts.append(f"From '{title}' ({date or 'undated'}):\n{text}")
            sources.append({
                "title": title,
                "date": date,
                "score": chunk.get("score", 0),
            })

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Extract structured data
        extraction_start = time.time()
        structured_data = await self._extract_structured_data(context, request)
        extraction_ms = round((time.time() - extraction_start) * 1000, 2)
        
        if "error" in structured_data:
            return structured_data

        total_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[INFOGRAPHIC] Generated schema in {total_ms}ms")

        return {
            "id": None,
            "structured_data": structured_data,
            "image": None,
            "image_url": None,
            "s3_key": None,
            "sources": sources,
            "confidence": confidence,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "extraction_ms": extraction_ms,
                "image_ms": 0,
                "total_ms": total_ms,
            },
            "metadata": {
                "style": style.value,
                "width": None,
                "height": None,
                "chunks_used": len(chunks),
                "output_format": "schema",
            },
        }

    async def export_to_claude_deck(
        self,
        structured_data: Dict[str, Any],
        sources: List[Dict[str, Any]],
        slides_count: int = 5,
    ) -> Dict[str, Any]:
        """
        Convert structured infographic data to Claude deck storyboard format.
        
        This format is optimized for Claude's artifact generation to create
        presentation decks.
        
        Args:
            structured_data: The extracted infographic data
            sources: Source documents used
            slides_count: Target number of slides
            
        Returns:
            Claude deck storyboard format
        """
        headline = structured_data.get("headline", "Presentation")
        subtitle = structured_data.get("subtitle", "")
        stats = structured_data.get("stats", [])
        key_points = structured_data.get("key_points", [])
        source_summary = structured_data.get("source_summary", "")
        
        slides = []
        slide_num = 1
        
        # Slide 1: Title slide
        slides.append({
            "slide_number": slide_num,
            "title": headline,
            "content_type": "title",
            "main_text": subtitle or "Key Insights & Data",
            "bullet_points": None,
            "visual_suggestion": "Full-width title with gradient background, company logo in corner",
            "speaker_notes": f"Welcome and introduction. This deck covers {headline.lower()}.",
        })
        slide_num += 1
        
        # Slide 2-3: Stats slides (split if many stats)
        if stats:
            stats_per_slide = max(2, len(stats) // 2) if len(stats) > 4 else len(stats)
            for i in range(0, len(stats), stats_per_slide):
                batch = stats[i:i + stats_per_slide]
                if slide_num > slides_count - 1:
                    break
                slides.append({
                    "slide_number": slide_num,
                    "title": "Key Metrics" if i == 0 else "Additional Metrics",
                    "content_type": "stats",
                    "main_text": "Critical numbers driving our decisions",
                    "bullet_points": [f"{s.get('icon', '📊')} {s.get('value')}: {s.get('label')}" for s in batch],
                    "visual_suggestion": "Large stat cards with icons, colorful backgrounds, side-by-side layout",
                    "speaker_notes": f"These metrics highlight {', '.join(s.get('label', '') for s in batch)}.",
                })
                slide_num += 1
        
        # Slide 4: Key points
        if key_points and slide_num < slides_count:
            slides.append({
                "slide_number": slide_num,
                "title": "Key Insights",
                "content_type": "key_points",
                "main_text": "Important findings from our analysis",
                "bullet_points": key_points[:5],
                "visual_suggestion": "Bullet list with custom icons, progressive reveal animation",
                "speaker_notes": "These are the main takeaways. " + (key_points[0] if key_points else ""),
            })
            slide_num += 1
        
        # Fill remaining slides with comparison or additional content
        while slide_num < slides_count:
            if slide_num == slides_count - 1:
                # Summary slide
                slides.append({
                    "slide_number": slide_num,
                    "title": "Summary",
                    "content_type": "summary",
                    "main_text": headline,
                    "bullet_points": [
                        f"Key stat: {stats[0].get('value')} {stats[0].get('label')}" if stats else "Data-driven insights",
                        key_points[0] if key_points else "Strategic recommendations",
                        "Next steps and action items",
                    ],
                    "visual_suggestion": "Clean summary layout, call-to-action button, contact info",
                    "speaker_notes": "Wrap up with key takeaways and next steps.",
                })
            else:
                slides.append({
                    "slide_number": slide_num,
                    "title": f"Deep Dive: {key_points[slide_num - 3] if len(key_points) > slide_num - 3 else 'Analysis'}",
                    "content_type": "comparison",
                    "main_text": "Detailed analysis and context",
                    "bullet_points": key_points[slide_num - 2:slide_num] if key_points else ["Analysis point"],
                    "visual_suggestion": "Split layout with chart on left, bullets on right",
                    "speaker_notes": "Deeper exploration of the data.",
                })
            slide_num += 1
        
        # Source attribution
        source_titles = [s.get("title", "Source") for s in sources[:3]]
        source_attr = f"Based on: {', '.join(source_titles)}"
        if source_summary:
            source_attr = source_summary
        
        return {
            "deck_title": headline,
            "deck_subtitle": subtitle,
            "total_slides": len(slides),
            "slides": slides,
            "theme_suggestion": "Modern professional theme with blue/gray palette, clean sans-serif fonts",
            "source_attribution": source_attr,
            "generation_prompt": f"Create a {len(slides)}-slide presentation about '{headline}' with professional styling, data visualizations for stats, and engaging visuals.",
        }

    async def export_to_gemini_visual(
        self,
        structured_data: Dict[str, Any],
        width: int = 1024,
        height: int = 1024,
    ) -> Dict[str, Any]:
        """
        Convert structured infographic data to Gemini visual instruction schema.
        
        This format provides detailed positioning and styling instructions
        optimized for Gemini's image generation capabilities.
        
        Args:
            structured_data: The extracted infographic data
            width: Canvas width in pixels
            height: Canvas height in pixels
            
        Returns:
            Gemini visual instruction schema
        """
        headline = structured_data.get("headline", "Infographic")
        subtitle = structured_data.get("subtitle", "")
        stats = structured_data.get("stats", [])
        key_points = structured_data.get("key_points", [])
        
        elements = []
        priority = 1
        
        # Title element (top 15% of canvas)
        elements.append({
            "element_type": "title",
            "content": headline,
            "position": {"x": 5, "y": 3, "width": 90, "height": 10},
            "style": {
                "font_size": 48,
                "color": "#1a1a2e",
                "background": "transparent",
                "alignment": "center",
                "font_weight": "bold",
            },
            "priority": priority,
        })
        priority += 1
        
        # Subtitle (below title)
        if subtitle:
            elements.append({
                "element_type": "subtitle",
                "content": subtitle,
                "position": {"x": 10, "y": 13, "width": 80, "height": 5},
                "style": {
                    "font_size": 24,
                    "color": "#4a4a6a",
                    "background": "transparent",
                    "alignment": "center",
                    "font_weight": "normal",
                },
                "priority": priority,
            })
            priority += 1
        
        # Divider
        elements.append({
            "element_type": "divider",
            "content": "",
            "position": {"x": 20, "y": 19, "width": 60, "height": 1},
            "style": {
                "color": "#e0e0e0",
                "background": "#e0e0e0",
            },
            "priority": priority,
        })
        priority += 1
        
        # Stats cards (middle section, 2x2 or 2x3 grid)
        stats_to_show = stats[:6]
        if stats_to_show:
            cols = 2 if len(stats_to_show) <= 4 else 3
            rows = (len(stats_to_show) + cols - 1) // cols
            card_width = 90 // cols - 2
            card_height = 25 // rows
            
            for i, stat in enumerate(stats_to_show):
                row = i // cols
                col = i % cols
                x = 5 + col * (card_width + 2)
                y = 22 + row * (card_height + 2)
                
                elements.append({
                    "element_type": "stat_card",
                    "content": f"{stat.get('icon', '📊')} {stat.get('value')}\n{stat.get('label')}",
                    "position": {"x": x, "y": y, "width": card_width, "height": card_height},
                    "style": {
                        "font_size": 28,
                        "color": "#ffffff",
                        "background": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#43e97b"][i % 6],
                        "alignment": "center",
                        "border_radius": 12,
                    },
                    "priority": priority,
                })
                priority += 1
        
        # Key points (bottom section)
        if key_points:
            points_text = "\n".join([f"• {p}" for p in key_points[:5]])
            elements.append({
                "element_type": "bullet_list",
                "content": points_text,
                "position": {"x": 5, "y": 70, "width": 90, "height": 22},
                "style": {
                    "font_size": 18,
                    "color": "#2d2d2d",
                    "background": "#f8f9fa",
                    "alignment": "left",
                    "padding": 16,
                    "border_radius": 8,
                },
                "priority": priority,
            })
            priority += 1
        
        # Footer
        elements.append({
            "element_type": "footer",
            "content": structured_data.get("source_summary", "Generated with DANI"),
            "position": {"x": 5, "y": 94, "width": 90, "height": 4},
            "style": {
                "font_size": 12,
                "color": "#888888",
                "background": "transparent",
                "alignment": "center",
            },
            "priority": priority,
        })
        
        # Determine layout type based on content
        layout_type = "dashboard" if len(stats) >= 4 else "hero" if len(stats) <= 2 else "grid"
        
        return {
            "canvas_width": width,
            "canvas_height": height,
            "background_color": "#ffffff",
            "color_palette": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#1a1a2e"],
            "elements": elements,
            "layout_type": layout_type,
            "generation_instructions": f"""Generate a {width}x{height} professional infographic with the following specifications:
- Clean, modern design with clear visual hierarchy
- Title prominently displayed at top
- Stat cards arranged in a {layout_type} layout
- Key points in a clean bullet list
- Use the specified color palette for consistency
- Ensure all text is clearly legible
- Add subtle shadows and rounded corners for depth
- Professional business/corporate aesthetic""",
        }

    def to_spec_schema(
        self,
        structured_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert structured infographic data to PROJECT PLAN spec schema format.
        
        The spec schema format is:
        {
            "title": "...",
            "sections": [{"header": "...", "bullets": [...]}],
            "recommended_visuals": "..."
        }
        
        This format is designed for downstream visual generation tools and
        matches the exact specification in the project documentation.
        
        Args:
            structured_data: The extracted infographic data (headline, stats, key_points, etc.)
            
        Returns:
            Dict in spec schema format
        """
        headline = structured_data.get("headline", "Infographic")
        subtitle = structured_data.get("subtitle", "")
        stats = structured_data.get("stats", [])
        key_points = structured_data.get("key_points", [])
        source_summary = structured_data.get("source_summary", "")
        
        sections = []
        
        # Section 1: Overview/Context (if subtitle exists)
        if subtitle:
            sections.append({
                "header": "Overview",
                "bullets": [subtitle],
            })
        
        # Section 2: Key Metrics (stats)
        if stats:
            stat_bullets = [
                f"{stat.get('icon', '📊')} {stat.get('value')}: {stat.get('label')}"
                for stat in stats
            ]
            sections.append({
                "header": "Key Metrics",
                "bullets": stat_bullets,
            })
        
        # Section 3: Key Insights (key_points)
        if key_points:
            sections.append({
                "header": "Key Insights",
                "bullets": key_points,
            })
        
        # Section 4: Source (if available)
        if source_summary:
            sections.append({
                "header": "Source",
                "bullets": [source_summary],
            })
        
        # Determine recommended visuals based on content
        visual_recommendations = []
        
        if stats:
            if len(stats) >= 4:
                visual_recommendations.append("Dashboard layout with metric cards")
            elif len(stats) >= 2:
                visual_recommendations.append("Side-by-side stat comparison")
            else:
                visual_recommendations.append("Hero stat highlight")
            
            # Check for numeric trends
            has_percentages = any('%' in str(s.get('value', '')) for s in stats)
            has_currencies = any('$' in str(s.get('value', '')) for s in stats)
            
            if has_percentages:
                visual_recommendations.append("Pie chart or progress bars for percentages")
            if has_currencies:
                visual_recommendations.append("Bar chart for financial comparisons")
        
        if key_points and len(key_points) >= 3:
            visual_recommendations.append("Bullet point list or numbered cards")
        
        if not visual_recommendations:
            visual_recommendations.append("Clean text-based infographic with icons")
        
        recommended_visuals = "; ".join(visual_recommendations)
        
        return {
            "title": headline,
            "sections": sections,
            "recommended_visuals": recommended_visuals,
        }

    async def generate_spec_schema(
        self,
        request: str,
        topic: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate infographic data directly in PROJECT PLAN spec schema format.
        
        This is a convenience method that generates structured data and 
        immediately converts it to spec schema format.
        
        Args:
            request: User's request for what the infographic should show
            topic: Optional topic to search for in RAG
            doc_type: Filter sources by type
            
        Returns:
            Dict with spec schema and metadata
        """
        start_time = time.time()
        
        logger.info(f"[INFOGRAPHIC] Generating spec schema: {request[:100]}...")

        # Retrieve relevant context
        retrieval_start = time.time()
        search_query = topic or request
        
        metadata_filter = None
        if doc_type and doc_type != "all":
            metadata_filter = MetadataFilter(
                doc_type=doc_type,
                speakers=None,
                source_file=None,
                date_from=None,
                date_to=None,
                transcript_id=None,
            )

        retrieval_result = await self.retrieval.search_with_confidence(
            query=search_query,
            limit=8,
            metadata_filter=metadata_filter,
        )
        chunks = retrieval_result["chunks"]
        confidence = retrieval_result["confidence"]
        retrieval_ms = round((time.time() - retrieval_start) * 1000, 2)

        if not chunks:
            return {
                "error": "No relevant context found for the infographic",
                "suggestion": "Try a different topic or broader search terms",
            }

        # Format context
        context_parts = []
        sources = []
        
        for chunk in chunks:
            title = chunk.get("title", "Untitled")
            text = chunk.get("text", "")
            date = chunk.get("date")
            
            context_parts.append(f"From '{title}' ({date or 'undated'}):\n{text}")
            sources.append({
                "title": title,
                "date": date,
                "score": chunk.get("score", 0),
            })

        context = "\n\n---\n\n".join(context_parts)

        # Extract structured data
        extraction_start = time.time()
        structured_data = await self._extract_structured_data(context, request)
        extraction_ms = round((time.time() - extraction_start) * 1000, 2)
        
        if "error" in structured_data:
            return structured_data

        # Convert to spec schema
        spec_schema = self.to_spec_schema(structured_data)
        
        total_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"[INFOGRAPHIC] Generated spec schema in {total_ms}ms")

        return {
            "spec_schema": spec_schema,
            "raw_structured_data": structured_data,
            "sources": sources,
            "confidence": confidence,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "extraction_ms": extraction_ms,
                "total_ms": total_ms,
            },
            "metadata": {
                "chunks_used": len(chunks),
                "output_format": "spec_schema",
            },
        }


# Singleton instance
_infographic_service: Optional[InfographicService] = None


def get_infographic_service() -> InfographicService:
    """Get or create the infographic service singleton."""
    global _infographic_service
    if _infographic_service is None:
        _infographic_service = InfographicService()
    return _infographic_service

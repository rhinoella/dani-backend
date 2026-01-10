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
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.llm.ollama import OllamaClient
from app.services.retrieval_service import RetrievalService
from app.services.storage_service import StorageService
from app.schemas.retrieval import MetadataFilter
from app.mcp.client import extract_all_content, extract_text
from app.database.models.infographic import (
    Infographic as InfographicModel,
    InfographicStyle as InfographicStyleDB,
    InfographicStatus,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class InfographicStyle(str, Enum):
    """Supported infographic visual styles."""
    MODERN = "modern"
    CORPORATE = "corporate"
    MINIMAL = "minimal"
    VIBRANT = "vibrant"
    DARK = "dark"


# Style-specific visual prompts for image generation
STYLE_PROMPTS = {
    InfographicStyle.MODERN: "modern sleek design, clean lines, gradient backgrounds, sans-serif typography, professional color palette with blue and white accents",
    InfographicStyle.CORPORATE: "corporate business style, professional navy and gray colors, formal typography, structured layout, executive presentation quality",
    InfographicStyle.MINIMAL: "minimalist design, lots of white space, simple icons, black and white with single accent color, clean typography",
    InfographicStyle.VIBRANT: "colorful vibrant design, bold colors, energetic layout, modern icons, dynamic visual elements, engaging and eye-catching",
    InfographicStyle.DARK: "dark theme infographic, dark background with bright accent colors, neon highlights, modern tech aesthetic, high contrast text",
}


# Prompt template for extracting structured infographic data
EXTRACTION_PROMPT = """You are extracting data for an infographic from meeting/document context.

CONTEXT:
{context}

USER REQUEST:
{request}

Extract the following in JSON format:
{{
    "headline": "A catchy headline (8 words max)",
    "subtitle": "Brief context line (15 words max)", 
    "stats": [
        {{"value": "number or metric", "label": "what it represents", "icon": "suggested emoji"}},
        // 4-6 stats total
    ],
    "key_points": [
        "Key insight or fact 1",
        "Key insight or fact 2",
        // 3-5 points
    ],
    "source_summary": "Brief note on data source (meeting name/date)"
}}

Rules:
- Use ONLY facts from the context
- Numbers and metrics are preferred for stats
- Keep all text concise and impactful
- If data is insufficient, use what's available

Output ONLY valid JSON:"""


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
        self._mcp_registry = None

    @property
    def mcp_registry(self):
        """Lazy load MCP registry to avoid circular imports."""
        if self._mcp_registry is None:
            from app.mcp.registry import get_registry
            self._mcp_registry = get_registry()
        return self._mcp_registry
    
    def _style_to_db(self, style: InfographicStyle) -> InfographicStyleDB:
        """Convert service style enum to database enum."""
        return InfographicStyleDB(style.value)

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

        Returns:
            Dict with image data, structured content, and metadata
        """
        start_time = time.time()
        infographic_id = None
        
        logger.info(f"[INFOGRAPHIC] Generating infographic: {request[:100]}...")

        # Step 1: Retrieve relevant context
        retrieval_start = time.time()
        search_query = topic or request
        
        if doc_type == "meeting":
            # WORKAROUND: Legacy data in Qdrant is missing the 'doc_type' field.
            # Filtering by doc_type="meeting" returns 0 results.
            # specific to "meeting" serves no purpose if the collection is implicitly meetings.
            # We switch to "all" to bypass the filter.
            doc_type = "all"

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

        # Step 2: Format context
        context_parts = []
        sources = []
        
        for chunk in chunks:
            title = chunk.get("title") or "Untitled"
            text = chunk.get("text", "")
            date = chunk.get("date")
            
            context_parts.append(f"From '{title}' ({date or 'undated'}):\n{text}")
            sources.append({
                "title": title,
                "date": date,
                "text_preview": text[:1000],
                "score": chunk.get("score", 0),
            })

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Extract structured data
        extraction_start = time.time()
        structured_data = await self._extract_structured_data(context, request)
        extraction_ms = round((time.time() - extraction_start) * 1000, 2)
        
        if "error" in structured_data:
            return structured_data

        logger.info(f"[INFOGRAPHIC] Extracted structured data in {extraction_ms}ms")

        # Step 4: Generate visual infographic via MCP
        image_start = time.time()
        image_result = await self._generate_image(structured_data, style, width, height)
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
        s3_key = None
        s3_url = None
        image_size = None
        
        if persist and image_result.get("image"):
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_result["image"])
                image_size = len(image_bytes)
                
                # Upload to S3
                s3_result = await self.storage.upload(
                    file_content=image_bytes,
                    filename=f"infographic_{int(time.time())}.png",
                    user_id=user_id,
                    content_type="image/png",
                    metadata={
                        "type": "infographic",
                        "style": style.value,
                        "headline": structured_data.get("headline", "")[:100],
                    },
                )
                s3_key = s3_result["key"]
                s3_url = s3_result["url"]
                logger.info(f"[INFOGRAPHIC] Uploaded to S3: {s3_key}")
            except Exception as e:
                logger.error(f"[INFOGRAPHIC] S3 upload failed: {e}")
                # Continue without persistence
        
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
            "image_url": s3_url or image_result.get("url"),
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

    async def _extract_structured_data(
        self, 
        context: str, 
        request: str
    ) -> Dict[str, Any]:
        """Extract structured infographic data from context using LLM."""
        prompt = EXTRACTION_PROMPT.format(context=context, request=request)
        
        try:
            # OllamaClient.generate() uses settings for temperature/tokens
            # We include extraction instructions directly in the prompt
            response = await self.llm.generate(prompt=prompt)
            
            # Clean and parse JSON
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?\n?", "", response)
                response = response.rstrip("`").strip()
            
            data = json.loads(response)
            
            # Validate required fields
            required = ["headline", "stats"]
            for field in required:
                if field not in data:
                    return {"error": f"Missing required field: {field}"}
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"[INFOGRAPHIC] JSON parse error: {e}")
            return {"error": f"Failed to parse structured data: {str(e)}"}
        except Exception as e:
            logger.error(f"[INFOGRAPHIC] Extraction failed: {e}")
            return {"error": f"Data extraction failed: {str(e)}"}

    async def _generate_image(
        self,
        structured_data: Dict[str, Any],
        style: InfographicStyle,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        """Generate visual infographic image via MCP."""
        
        # Build the image generation prompt
        prompt = self._build_image_prompt(structured_data, style)
        
        try:
            # Check if MCP is available
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

    def _build_image_prompt(
        self, 
        structured_data: Dict[str, Any], 
        style: InfographicStyle
    ) -> str:
        """Build a detailed prompt for image generation."""
        
        headline = structured_data.get("headline", "Infographic")
        subtitle = structured_data.get("subtitle", "")
        stats = structured_data.get("stats", [])
        key_points = structured_data.get("key_points", [])
        
        # Format stats for the prompt
        stats_text = "\n".join([
            f"- {s.get('icon', '📊')} {s.get('value')}: {s.get('label')}"
            for s in stats[:6]
        ])
        
        # Format key points
        points_text = "\n".join([f"- {p}" for p in key_points[:5]])
        
        # Get style-specific visual instructions
        style_instructions = STYLE_PROMPTS.get(style, STYLE_PROMPTS[InfographicStyle.MODERN])
        
        prompt = f"""Create a professional infographic image with the following content:

TITLE: {headline}
{f'SUBTITLE: {subtitle}' if subtitle else ''}

KEY STATISTICS:
{stats_text}

{'KEY POINTS:' if points_text else ''}
{points_text}

VISUAL STYLE: {style_instructions}

DESIGN REQUIREMENTS:
- Clean, readable typography
- Professional infographic layout
- Visual hierarchy with title prominent
- Stats displayed with icons or visual elements
- Balanced composition
- High quality, presentation-ready
- Text should be clearly legible
- Use visual elements like charts, icons, or graphics where appropriate

Generate a complete infographic image, NOT a photograph or illustration of people."""

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
            style=self._style_to_db(style),
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
                expiry_seconds=expiry_seconds,
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

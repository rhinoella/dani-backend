"""
Infographic API routes.

Endpoints for generating visual infographics from meeting/document context.
Supports full persistence with S3 storage and database metadata.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Literal
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import base64
from enum import Enum

from app.services.infographic_service import (
    InfographicService,
    InfographicStyle,
    get_infographic_service,
)
from app.api.deps import get_optional_user, get_db
from app.database.models import User
from app.database.models.infographic import InfographicStatus

router = APIRouter(prefix="/infographic", tags=["infographic"])
logger = logging.getLogger(__name__)


# ============== Request/Response Schemas ==============

class OutputFormat(str, Enum):
    """Output format for infographic generation."""
    VISUAL = "visual"           # Generate visual image (default)
    SCHEMA = "schema"           # Return structured spec only (no image)
    BOTH = "both"               # Return both spec and image


class InfographicRequest(BaseModel):
    """Request to generate an infographic."""
    request: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="What you want the infographic to show (e.g., 'Q1 mobile strategy key metrics')"
    )
    topic: Optional[str] = Field(
        None,
        description="Optional topic to search for context (defaults to request)"
    )
    style: Optional[str] = Field(
        "modern",
        description="Visual style: modern, corporate, minimal, vibrant, dark"
    )
    doc_type: Optional[str] = Field(
        None,
        description="Filter sources: meeting, email, document, note, or all"
    )
    width: Optional[int] = Field(
        1024,
        ge=512,
        le=2048,
        description="Image width in pixels"
    )
    height: Optional[int] = Field(
        1024,
        ge=512,
        le=2048,
        description="Image height in pixels"
    )
    output_format: Optional[str] = Field(
        "visual",
        description="Output format: visual (image only), schema (spec only), both (image + spec)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request": "Create an infographic showing Q1 mobile app strategy and key milestones",
                "style": "modern",
                "doc_type": "meeting",
                "width": 1024,
                "height": 1024,
                "output_format": "both"
            }
        }
    }


class InfographicStat(BaseModel):
    """A single stat/metric in the infographic."""
    value: str
    label: str
    icon: Optional[str] = None


class StructuredData(BaseModel):
    """Structured data extracted for the infographic."""
    headline: str
    subtitle: Optional[str] = None
    stats: List[InfographicStat]
    key_points: Optional[List[str]] = None
    source_summary: Optional[str] = None


class InfographicResponse(BaseModel):
    """Response with generated infographic."""
    id: Optional[str] = Field(None, description="Infographic ID (if persisted)")
    structured_data: StructuredData
    image: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to the generated image")
    s3_key: Optional[str] = Field(None, description="S3 storage key")
    image_format: str = Field("png", description="Image format")
    sources: List[dict]
    confidence: dict
    timing: dict
    metadata: dict


class InfographicErrorResponse(BaseModel):
    """Error response when infographic generation fails."""
    error: str
    suggestion: Optional[str] = None
    structured_data: Optional[StructuredData] = None


class StyleInfo(BaseModel):
    """Information about an infographic style."""
    style: str
    description: str


# ============== Export Format Schemas ==============

class ClaudeSlide(BaseModel):
    """A single slide in Claude deck storyboard format."""
    slide_number: int
    title: str
    content_type: Literal["title", "stats", "key_points", "comparison", "summary"]
    main_text: str
    bullet_points: Optional[List[str]] = None
    visual_suggestion: str
    speaker_notes: str


class ClaudeDeckExport(BaseModel):
    """Claude deck storyboard export format."""
    deck_title: str
    deck_subtitle: Optional[str] = None
    total_slides: int
    slides: List[ClaudeSlide]
    theme_suggestion: str
    source_attribution: str
    generation_prompt: str


class GeminiVisualElement(BaseModel):
    """Visual element in Gemini instruction schema."""
    element_type: Literal["title", "subtitle", "stat_card", "icon", "chart", "bullet_list", "divider", "footer"]
    content: str
    position: dict = Field(description="{x, y, width, height} as percentages")
    style: dict = Field(description="{font_size, color, background, alignment}")
    priority: int = Field(ge=1, le=10, description="Visual priority 1-10")


class GeminiVisualExport(BaseModel):
    """Gemini visual instruction schema export format."""
    canvas_width: int
    canvas_height: int
    background_color: str
    color_palette: List[str]
    elements: List[GeminiVisualElement]
    layout_type: Literal["grid", "flow", "hero", "dashboard"]
    generation_instructions: str


# ============== PROJECT PLAN Spec Schema ==============

class SpecSchemaSection(BaseModel):
    """A section in the spec schema format."""
    header: str = Field(description="Section header/title")
    bullets: List[str] = Field(description="Bullet points in this section")


class SpecSchema(BaseModel):
    """PROJECT PLAN spec schema format for infographic data."""
    title: str = Field(description="Infographic title")
    sections: List[SpecSchemaSection] = Field(description="Content sections with headers and bullets")
    recommended_visuals: str = Field(description="Recommended visualization types")


class SpecSchemaResponse(BaseModel):
    """Response with infographic data in PROJECT PLAN spec schema format."""
    spec_schema: SpecSchema
    raw_structured_data: StructuredData
    sources: List[dict]
    confidence: dict
    timing: dict
    metadata: dict


# ============== Endpoints ==============

@router.get("/styles", response_model=List[StyleInfo])
async def list_styles():
    """
    List available infographic styles.
    
    Returns descriptions of each visual style that can be used
    when generating infographics.
    """
    service = get_infographic_service()
    return service.get_styles()


@router.post("/generate", response_model=InfographicResponse)
async def generate_infographic(
    req: InfographicRequest,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a visual infographic.

    This endpoint:
    1. Searches for relevant context from meetings/documents
    2. Extracts structured data (headline, stats, key points)
    3. Generates a visual infographic image via MCP (if output_format is visual or both)
    4. Persists the image to S3 and metadata to database

    Output formats:
    - visual (default): Returns image with structured data
    - schema: Returns only structured spec (no image generation)
    - both: Returns both image and full structured spec

    The response includes the infographic ID, structured data, and the generated image.
    """
    user_id = current_user.id if current_user else None
    logger.info(f"[INFOGRAPHIC API] Generate request: {req.request[:100]}, user={user_id or 'anonymous'}")

    # Validate output format
    output_format = req.output_format or "visual"
    try:
        output_fmt = OutputFormat(output_format)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid output_format: {output_format}",
                "valid_formats": [f.value for f in OutputFormat],
            }
        )

    # Validate style
    try:
        style = InfographicStyle(req.style) if req.style else InfographicStyle.MODERN
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid style: {req.style}",
                "valid_styles": [s.value for s in InfographicStyle],
            }
        )

    service = get_infographic_service()
    
    # Schema-only mode: just extract structured data
    if output_fmt == OutputFormat.SCHEMA:
        result = await service.generate_schema_only(
            request=req.request,
            topic=req.topic,
            style=style,
            doc_type=req.doc_type,
            user_id=user_id,
            db=db,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result)
        return result
    
    # Visual or Both mode: generate full infographic
    result = await service.generate(
        request=req.request,
        topic=req.topic,
        style=style,
        doc_type=req.doc_type,
        width=req.width or 1024,
        height=req.height or 1024,
        user_id=user_id,
        db=db,
        persist=True,
    )

    if "error" in result and "structured_data" not in result:
        raise HTTPException(status_code=400, detail=result)

    # If we have structured data but image failed, still return partial success
    if "image_error" in result:
        logger.warning(f"[INFOGRAPHIC API] Image generation failed: {result['image_error']}")

    return result


@router.post("/generate/image", responses={200: {"content": {"image/png": {}}}})
async def generate_infographic_image(
    req: InfographicRequest,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate infographic and return the image directly.

    Returns the PNG image as binary data for direct download/display.
    The infographic is still persisted to S3 and database.
    """
    user_id = current_user.id if current_user else None
    logger.info(f"[INFOGRAPHIC API] Direct image request: {req.request[:100]}")

    try:
        style = InfographicStyle(req.style) if req.style else InfographicStyle.MODERN
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style: {req.style}"
        )

    service = get_infographic_service()
    
    result = await service.generate(
        request=req.request,
        topic=req.topic,
        style=style,
        doc_type=req.doc_type,
        width=req.width or 1024,
        height=req.height or 1024,
        user_id=user_id,
        db=db,
        persist=True,
    )

    if "error" in result and not result.get("image"):
        raise HTTPException(status_code=400, detail=result.get("error"))

    image_data = result.get("image")
    if not image_data:
        raise HTTPException(
            status_code=500,
            detail="Image generation failed - no image data returned"
        )

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to decode image: {str(e)}"
        )

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="infographic.png"'
        }
    )


@router.post("/extract", response_model=StructuredData)
async def extract_infographic_data(
    req: InfographicRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Extract structured infographic data without generating an image.

    Useful for previewing what data would be included in the infographic
    before committing to image generation.
    """
    logger.info(f"[INFOGRAPHIC API] Extract request: {req.request[:100]}")

    service = get_infographic_service()
    
    # Use internal method to just extract data
    from app.schemas.retrieval import MetadataFilter
    
    metadata_filter = None
    if req.doc_type and req.doc_type != "all":
        metadata_filter = MetadataFilter(
            doc_type=req.doc_type,
            speakers=None,
            source_file=None,
            date_from=None,
            date_to=None,
            transcript_id=None,
        )

    retrieval_result = await service.retrieval.search_with_confidence(
        query=req.topic or req.request,
        limit=8,
        metadata_filter=metadata_filter,
    )
    
    chunks = retrieval_result["chunks"]
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant context found for extraction"
        )

    # Format context
    context_parts = []
    for chunk in chunks:
        title = chunk.get("title", "Untitled")
        text = chunk.get("text", "")
        date = chunk.get("date")
        context_parts.append(f"From '{title}' ({date or 'undated'}):\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # Extract structured data
    structured = await service._extract_structured_data(context, req.request)
    
    if "error" in structured:
        raise HTTPException(status_code=400, detail=structured["error"])

    return structured


@router.post("/generate/spec-schema", response_model=SpecSchemaResponse)
async def generate_spec_schema(
    req: InfographicRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Generate infographic data in PROJECT PLAN spec schema format.
    
    Returns the infographic data in the exact format specified in the 
    project documentation:
    
    ```json
    {
        "title": "...",
        "sections": [{"header": "...", "bullets": [...]}],
        "recommended_visuals": "..."
    }
    ```
    
    This format is designed for downstream visual generation tools and
    provides clear structure for presentation/documentation purposes.
    
    Also returns the raw structured data and metadata for additional context.
    """
    logger.info(f"[INFOGRAPHIC API] Spec schema request: {req.request[:100]}")
    
    service = get_infographic_service()
    
    result = await service.generate_spec_schema(
        request=req.request,
        topic=req.topic,
        doc_type=req.doc_type,
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    
    return result


@router.post("/convert/spec-schema", response_model=SpecSchema)
async def convert_to_spec_schema(
    structured_data: StructuredData,
):
    """
    Convert existing structured data to PROJECT PLAN spec schema format.
    
    This is a utility endpoint that converts already-extracted infographic
    data (headline, stats, key_points) to the spec schema format without
    making any RAG calls.
    
    Useful when you have structured data from another endpoint and want
    it in spec schema format.
    """
    logger.info(f"[INFOGRAPHIC API] Convert to spec schema: {structured_data.headline}")
    
    service = get_infographic_service()
    
    # Convert Pydantic model to dict
    data = structured_data.model_dump()
    
    spec_schema = service.to_spec_schema(data)
    
    return spec_schema


# ============== Persistence Endpoints ==============

class InfographicListItem(BaseModel):
    """Summary item for infographic listing."""
    id: str
    headline: Optional[str] = None
    style: Optional[str] = None
    status: str
    image_url: Optional[str] = None
    created_at: Optional[str] = None


class InfographicListResponse(BaseModel):
    """Response for listing infographics."""
    items: List[InfographicListItem]
    total: int
    limit: int
    offset: int


@router.get("/", response_model=InfographicListResponse)
async def list_infographics(
    limit: int = Query(20, ge=1, le=100, description="Maximum items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(None, description="Filter by status: completed, failed, pending"),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List infographics for the current user.
    
    Returns a paginated list of user's generated infographics.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = InfographicStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid: pending, generating, completed, failed"
            )
    
    service = get_infographic_service()
    
    items = await service.list_by_user(
        user_id=current_user.id,
        db=db,
        limit=limit,
        offset=offset,
        status=status_filter,
    )
    
    return {
        "items": [
            {
                "id": item["id"],
                "headline": item.get("headline"),
                "style": item.get("style"),
                "status": item.get("status"),
                "image_url": item.get("image_url"),
                "created_at": item.get("created_at"),
            }
            for item in items
        ],
        "total": len(items),  # In real implementation, get count from DB
        "limit": limit,
        "offset": offset,
    }


@router.get("/{infographic_id}")
async def get_infographic(
    infographic_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific infographic by ID.
    
    Returns full infographic details including structured data and sources.
    """
    service = get_infographic_service()
    
    user_id = current_user.id if current_user else None
    infographic = await service.get_by_id(
        infographic_id=infographic_id,
        db=db,
        user_id=user_id,
    )
    
    if not infographic:
        raise HTTPException(status_code=404, detail="Infographic not found")
    
    return infographic


@router.get("/{infographic_id}/download", responses={200: {"content": {"image/png": {}}}})
async def download_infographic(
    infographic_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Download infographic image as PNG.
    
    Redirects to a presigned S3 URL for direct download.
    """
    from fastapi.responses import RedirectResponse
    
    service = get_infographic_service()
    
    user_id = current_user.id if current_user else None
    presigned_url = await service.get_presigned_url(
        infographic_id=infographic_id,
        db=db,
        user_id=user_id,
        expiry_seconds=3600,
    )
    
    if not presigned_url:
        raise HTTPException(status_code=404, detail="Infographic not found or no image available")
    
    return RedirectResponse(url=presigned_url)


class RegenerateUrlRequest(BaseModel):
    """Request to regenerate a presigned URL from an S3 key."""
    s3_key: str = Field(..., description="The S3 key for the image")
    expiry_seconds: int = Field(3600 * 24, ge=60, le=604800, description="URL expiry in seconds (default 24h, max 7 days)")


class RegenerateUrlResponse(BaseModel):
    """Response containing the new presigned URL."""
    url: str
    expires_in_seconds: int


@router.post("/regenerate-url", response_model=RegenerateUrlResponse)
async def regenerate_presigned_url(
    req: RegenerateUrlRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Regenerate a presigned URL from an S3 key.
    
    This is useful when a stored presigned URL has expired but the S3 key
    is still available in the message metadata.
    """
    from app.services.storage_service import StorageService
    
    try:
        storage = StorageService()
        url = await storage.get_presigned_url(
            key=req.s3_key,
            expiry_seconds=req.expiry_seconds,
        )
        return RegenerateUrlResponse(
            url=url,
            expires_in_seconds=req.expiry_seconds,
        )
    except Exception as e:
        logger.error(f"Failed to regenerate presigned URL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate URL: {str(e)}"
        )


@router.delete("/{infographic_id}")
async def delete_infographic(
    infographic_id: str,
    hard_delete: bool = Query(False, description="Permanently delete (including S3)"),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an infographic.
    
    By default, performs a soft delete. Use hard_delete=true to permanently
    remove from database and S3.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    service = get_infographic_service()
    
    deleted = await service.delete(
        infographic_id=infographic_id,
        db=db,
        user_id=current_user.id,
        hard_delete=hard_delete,
    )
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Infographic not found")
    
    return {"message": "Infographic deleted", "id": infographic_id}


# ============== Export Format Endpoints ==============

class ExportRequest(BaseModel):
    """Request to export an existing infographic to a specific format."""
    infographic_id: Optional[str] = Field(
        None,
        description="ID of existing infographic to export"
    )
    request: Optional[str] = Field(
        None,
        description="Generate new infographic spec from this request"
    )
    topic: Optional[str] = Field(
        None,
        description="Optional topic filter for RAG search"
    )
    doc_type: Optional[str] = Field(
        None,
        description="Filter sources by type"
    )
    slides_count: Optional[int] = Field(
        5,
        ge=3,
        le=15,
        description="Number of slides for deck export"
    )


@router.post("/export/claude", response_model=ClaudeDeckExport)
async def export_to_claude(
    req: ExportRequest,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Export infographic as Claude deck storyboard format.
    
    This format is optimized for Claude's artifact generation capabilities,
    producing a structured storyboard that can be used to generate 
    presentation decks via Claude.
    
    Provide either:
    - infographic_id: Export an existing infographic
    - request: Generate new spec from RAG context
    """
    user_id = current_user.id if current_user else None
    service = get_infographic_service()
    
    # Get or generate structured data
    structured_data = None
    sources = []
    
    if req.infographic_id:
        # Fetch existing infographic
        infographic = await service.get_by_id(
            infographic_id=req.infographic_id,
            db=db,
            user_id=user_id,
        )
        if not infographic:
            raise HTTPException(status_code=404, detail="Infographic not found")
        structured_data = infographic.get("structured_data", {})
        sources = infographic.get("sources", [])
    elif req.request:
        # Generate new spec
        result = await service.generate_schema_only(
            request=req.request,
            topic=req.topic,
            doc_type=req.doc_type,
            user_id=user_id,
            db=db,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        structured_data = result.get("structured_data", {})
        sources = result.get("sources", [])
    else:
        raise HTTPException(
            status_code=400,
            detail="Either infographic_id or request must be provided"
        )
    
    # Convert to Claude deck format
    claude_deck = await service.export_to_claude_deck(
        structured_data=structured_data,
        sources=sources,
        slides_count=req.slides_count or 5,
    )
    
    return claude_deck


@router.post("/export/gemini", response_model=GeminiVisualExport)
async def export_to_gemini(
    req: ExportRequest,
    width: int = Query(1024, ge=512, le=2048),
    height: int = Query(1024, ge=512, le=2048),
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Export infographic as Gemini visual instruction schema.
    
    This format is optimized for Gemini's image generation capabilities,
    producing a detailed visual layout specification with element positions,
    styles, and rendering instructions.
    
    Provide either:
    - infographic_id: Export an existing infographic
    - request: Generate new spec from RAG context
    """
    user_id = current_user.id if current_user else None
    service = get_infographic_service()
    
    # Get or generate structured data
    structured_data = None
    
    if req.infographic_id:
        # Fetch existing infographic
        infographic = await service.get_by_id(
            infographic_id=req.infographic_id,
            db=db,
            user_id=user_id,
        )
        if not infographic:
            raise HTTPException(status_code=404, detail="Infographic not found")
        structured_data = infographic.get("structured_data", {})
    elif req.request:
        # Generate new spec
        result = await service.generate_schema_only(
            request=req.request,
            topic=req.topic,
            doc_type=req.doc_type,
            user_id=user_id,
            db=db,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        structured_data = result.get("structured_data", {})
    else:
        raise HTTPException(
            status_code=400,
            detail="Either infographic_id or request must be provided"
        )
    
    # Convert to Gemini visual format
    gemini_spec = await service.export_to_gemini_visual(
        structured_data=structured_data,
        width=width,
        height=height,
    )
    
    return gemini_spec

"""
Ghostwriter API routes.

Endpoints for generating content in Bunmi's voice.
"""

from __future__ import annotations

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.ghostwriter_service import (
    GhostwriterService,
    ContentType,
    get_ghostwriter,
)
from app.api.deps import get_optional_user
from app.database.models import User

router = APIRouter(prefix="/ghostwriter", tags=["ghostwriter"])
logger = logging.getLogger(__name__)


# ============== Request/Response Schemas ==============

class GhostwriteRequest(BaseModel):
    """Request to generate content."""
    content_type: str = Field(
        ...,
        description="Type of content: linkedin_post, email, blog_draft, tweet_thread, newsletter, meeting_summary"
    )
    request: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="What you want written (e.g., 'Write about the Q1 product strategy')"
    )
    topic: Optional[str] = Field(
        None,
        description="Optional topic to search for context (defaults to request)"
    )
    doc_type: Optional[str] = Field(
        None,
        description="Filter sources: meeting, email, document, note, or all"
    )
    additional_context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Extra context to include in the generation"
    )
    tone: Optional[str] = Field(
        None,
        description="Tone override: formal, casual, urgent, inspirational"
    )
    max_length: Optional[int] = Field(
        None,
        ge=50,
        le=2000,
        description="Approximate max length in words"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content_type": "linkedin_post",
                "request": "Write about lessons learned from our mobile app launch",
                "doc_type": "meeting",
                "tone": "inspirational"
            }
        }


class RefineRequest(BaseModel):
    """Request to refine previously generated content."""
    content: str = Field(
        ...,
        min_length=10,
        description="The previously generated content to refine"
    )
    feedback: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Feedback or refinement instructions"
    )
    content_type: str = Field(
        ...,
        description="Type of content being refined"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Just launched our mobile app...",
                "feedback": "Make it more personal and add a question at the end",
                "content_type": "linkedin_post"
            }
        }


class GhostwriteResponse(BaseModel):
    """Response with generated content."""
    content: str
    content_type: str
    word_count: int
    sources: List[dict]
    confidence: dict
    timing: dict
    metadata: dict


class RefineResponse(BaseModel):
    """Response with refined content."""
    content: str
    content_type: str
    word_count: int
    timing: dict
    refined_from: str
    feedback_applied: str


class ContentTypeInfo(BaseModel):
    """Information about a content type."""
    type: str
    description: str


# ============== Endpoints ==============

@router.get("/types", response_model=List[ContentTypeInfo])
async def list_content_types():
    """
    List all supported content types for ghostwriting.
    """
    service = get_ghostwriter()
    return service.get_content_types()


@router.post("/generate", response_model=GhostwriteResponse)
async def generate_content(
    req: GhostwriteRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Generate content in DANI's voice.

    This endpoint uses RAG to find relevant context from meetings and documents,
    then generates the requested content type (LinkedIn post, email, blog, etc.)
    in Bunmi's authentic voice.

    The generated content is ready to use - no AI disclaimers or preambles.
    """
    logger.info(f"[GHOSTWRITER API] Generate request: type={req.content_type}, user={current_user.id if current_user else 'anonymous'}")

    # Validate content type
    try:
        content_type = ContentType(req.content_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid content type: {req.content_type}",
                "valid_types": [ct.value for ct in ContentType],
            }
        )

    service = get_ghostwriter()
    
    result = await service.generate(
        content_type=content_type,
        request=req.request,
        topic=req.topic,
        doc_type=req.doc_type,
        additional_context=req.additional_context,
        tone=req.tone,
        max_length=req.max_length,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    logger.info(f"[GHOSTWRITER API] Generated {result['word_count']} words in {result['timing']['total_ms']}ms")
    
    return result


@router.post("/refine", response_model=RefineResponse)
async def refine_content(
    req: RefineRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Refine previously generated content based on feedback.

    Use this to iterate on content - make it shorter, change tone, add specific points, etc.
    """
    logger.info(f"[GHOSTWRITER API] Refine request: type={req.content_type}")

    # Validate content type
    try:
        content_type = ContentType(req.content_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid content type: {req.content_type}",
                "valid_types": [ct.value for ct in ContentType],
            }
        )

    service = get_ghostwriter()
    
    result = await service.refine(
        content=req.content,
        feedback=req.feedback,
        content_type=content_type,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return result


class QuickGenerateRequest(BaseModel):
    """Quick request for shortcut endpoints."""
    request: str = Field(..., min_length=5, description="What to write about")
    topic: Optional[str] = Field(None, description="Optional topic for context search")
    tone: Optional[str] = Field(None, description="Tone: formal, casual, urgent")


@router.post("/linkedin", response_model=GhostwriteResponse)
async def generate_linkedin_post(
    req: QuickGenerateRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Quick endpoint to generate a LinkedIn post.

    Shortcut for /generate with content_type=linkedin_post.
    """
    service = get_ghostwriter()
    
    result = await service.generate(
        content_type=ContentType.LINKEDIN_POST,
        request=req.request,
        topic=req.topic,
        tone=req.tone,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return result


@router.post("/email", response_model=GhostwriteResponse)
async def generate_email(
    req: QuickGenerateRequest,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Quick endpoint to generate an email draft.

    Shortcut for /generate with content_type=email.
    """
    service = get_ghostwriter()
    
    result = await service.generate(
        content_type=ContentType.EMAIL,
        request=req.request,
        topic=req.topic,
        tone=req.tone,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return result

"""
Message schemas.
"""

from pydantic import BaseModel, Field, AliasChoices
from typing import Optional, List, Any, Union
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SourceReference(BaseModel):
    """Source reference for a message."""
    # Meeting/Transcript information (primary source fields)
    title: Optional[str] = None
    date: Optional[Union[str, int]] = None
    transcript_id: Optional[str] = None
    speakers: Optional[List[str]] = []
    text: Optional[str] = None
    text_preview: Optional[str] = None  # Alias for text, for frontend compatibility
    relevance_score: Optional[float] = None
    raw_score: Optional[float] = None
    meeting_category: Optional[str] = None  # Inferred meeting category
    category_confidence: Optional[float] = None  # Category confidence score
    
    # Document-based sources (for uploaded documents)
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    filename: Optional[str] = None
    content_preview: Optional[str] = None
    score: Optional[float] = None
    
    model_config = {"extra": "allow"}  # Allow additional fields


class MessageBase(BaseModel):
    """Base message fields."""
    role: MessageRole
    content: str


class MessageCreate(MessageBase):
    """Schema for creating a message."""
    sources: Optional[List[SourceReference]] = None
    confidence_score: Optional[float] = None
    metadata: Optional[dict] = None


class MessageUpdate(BaseModel):
    """Schema for updating a message (limited updates allowed)."""
    metadata: Optional[dict] = None


class MessageResponse(BaseModel):
    """Message response schema."""
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    sources: Optional[List[SourceReference]] = None
    confidence_score: Optional[float] = None
    # Use AliasChoices to accept both explicit 'metadata=' and ORM 'metadata_' attribute
    metadata: Optional[dict] = Field(None, validation_alias=AliasChoices('metadata', 'metadata_'))
    created_at: datetime
    
    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    """Paginated message list."""
    messages: List[MessageResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class ConversationContext(BaseModel):
    """Conversation context for LLM."""
    messages: List[MessageBase]
    summary: Optional[str] = None
    total_messages: int
    context_token_count: int
    truncated: bool = False


class MessageSearchRequest(BaseModel):
    """Search messages request."""
    query: str = Field(..., min_length=1, max_length=500)
    conversation_id: Optional[str] = None
    role: Optional[MessageRole] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class MessageSearchResult(BaseModel):
    """Search result for a message."""
    message: MessageResponse
    conversation_id: str
    conversation_title: Optional[str] = None
    relevance_score: float


class MessageSearchResponse(BaseModel):
    """Message search response."""
    results: List[MessageSearchResult]
    total: int
    page: int
    page_size: int
    has_more: bool


class RegenerateRequest(BaseModel):
    """Request to regenerate an assistant message."""
    message_id: str
    additional_context: Optional[str] = None

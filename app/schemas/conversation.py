"""
Conversation schemas.
"""

from pydantic import BaseModel, Field, AliasChoices
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ConversationStatus(str, Enum):
    """Conversation status."""
    ACTIVE = "active"
    ARCHIVED = "archived"


class ConversationBase(BaseModel):
    """Base conversation fields."""
    title: Optional[str] = None
    metadata: Optional[dict] = None


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation."""
    pass


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    title: Optional[str] = None
    metadata: Optional[dict] = None


class ConversationSummary(BaseModel):
    """Conversation summary for lists."""
    id: str
    title: Optional[str] = None
    message_count: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_message_preview: Optional[str] = None
    
    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Full conversation response."""
    id: str
    user_id: str
    title: Optional[str] = None
    summary: Optional[str] = None
    message_count: int = 0
    # Use AliasChoices to accept both explicit 'metadata=' and ORM 'metadata_' attribute
    metadata: Optional[dict] = Field(None, validation_alias=AliasChoices('metadata', 'metadata_'))
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class ConversationWithMessages(ConversationResponse):
    """Conversation with messages."""
    messages: List["MessageResponse"] = []


class ConversationListResponse(BaseModel):
    """Paginated conversation list."""
    conversations: List[ConversationSummary]
    total: int
    page: int
    page_size: int
    has_more: bool


class ConversationSearchRequest(BaseModel):
    """Search conversations request."""
    query: str = Field(..., min_length=1, max_length=500)
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class ConversationExportResponse(BaseModel):
    """Conversation export format."""
    id: str
    title: Optional[str] = None
    created_at: datetime
    messages: List[dict]
    metadata: Optional[dict] = None


# Forward reference for circular import
from app.schemas.message import MessageResponse
ConversationWithMessages.model_rebuild()

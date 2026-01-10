from __future__ import annotations

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


# Document source types for filtering
DocSourceType = Literal["meeting", "email", "document", "note", "all"]


class ChatRequest(BaseModel):
    query: str
    verbose: Optional[bool] = False
    output_format: Optional[Literal[
        "summary", 
        "decisions", 
        "tasks", 
        "insights",
        "email",
        "whatsapp",
        "slides",
        "infographic"
    ]] = Field(None, description="Request structured output format")
    # New fields for conversation memory
    conversation_id: Optional[str] = Field(None, description="ID of existing conversation to continue")
    include_history: Optional[bool] = Field(True, description="Include conversation history in context")
    # Document type filter
    doc_type: Optional[DocSourceType] = Field(
        None, 
        description="Filter sources by type: meeting, email, document, note, or all"
    )
    # Uploaded documents - full content will be injected into chat context
    document_ids: Optional[List[str]] = Field(
        None,
        description="List of uploaded document IDs to include in chat context. Full document content will be injected."
    )


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    output_format: Optional[str] = None
    debug: Optional[Dict] = None
    # New fields for conversation tracking
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    confidence: Optional[float] = None


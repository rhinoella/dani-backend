from __future__ import annotations

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# Document source types for filtering
DocSourceType = Literal["meeting", "email", "document", "note", "all"]

# Meeting category types for filtering
MeetingCategory = Literal[
    "board",      # Board meetings, governance
    "1on1",       # One-on-one meetings
    "standup",    # Daily standups, scrums
    "client",     # Client/customer meetings
    "internal",   # Internal team meetings
    "external",   # Meetings with external participants
    "all"         # No category filter
]


class MetadataFilter(BaseModel):
    """Metadata filtering options for retrieval."""
    organizer_email: Optional[str] = None
    speakers: Optional[List[str]] = Field(default=None, description="Filter by speaker names")
    source_file: Optional[str] = Field(default=None, description="Filter by specific source file")
    date_from: Optional[int] = Field(default=None, description="Filter meetings from this date (Unix timestamp ms)")
    date_to: Optional[int] = Field(default=None, description="Filter meetings until this date (Unix timestamp ms)")
    transcript_id: Optional[str] = Field(default=None, description="Filter by specific transcript ID")
    doc_type: Optional[DocSourceType] = Field(
        default=None, 
        description="Filter by document source type: meeting, email, document, note, or all"
    )
    meeting_category: Optional[MeetingCategory] = Field(
        default=None,
        description="Filter by meeting category: board, 1on1, standup, client, internal, external, or all"
    )
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by specific document IDs")


class RetrievalRequest(BaseModel):
    """Request for retrieval preview or search."""
    query: str = Field(..., min_length=2, description="Search query")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of results")
    filters: Optional[MetadataFilter] = Field(None, description="Metadata filters")


class RetrievalResult(BaseModel):
    """A single retrieval result."""
    score: float
    transcript_id: Optional[str] = None
    title: Optional[str] = None
    date: Optional[int] = None
    organizer_email: Optional[str] = None
    speakers: Optional[List[str]] = None
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    text: str
    text_preview: Optional[str] = Field(None, description="Truncated preview of text")
    meeting_category: Optional[str] = Field(
        None, 
        description="Inferred meeting category: board, 1on1, standup, client, internal, external"
    )
    category_confidence: Optional[float] = Field(
        None,
        description="Confidence score for the inferred meeting category (0.0-1.0)"
    )


class ConfidenceMetrics(BaseModel):
    """Confidence scoring metrics for retrieval quality."""
    score: float = Field(..., description="Overall confidence score (0-1)")
    level: str = Field(..., description="Confidence level: high, medium, low, very_low, none")
    reason: str = Field(..., description="Reason for the confidence level")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Detailed metrics")


class QueryAnalysis(BaseModel):
    """Query analysis results."""
    intent: str = Field(..., description="Detected query intent type")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    time_references: List[str] = Field(default_factory=list, description="Extracted time references")
    processed_query: str = Field(..., description="Processed/compressed query")


class RetrievalResponse(BaseModel):
    """Response from retrieval with enhanced metrics."""
    query: str
    limit: int
    filters_applied: Optional[MetadataFilter] = None
    results_count: int
    results: List[RetrievalResult]
    confidence: Optional[Dict[str, Any]] = Field(None, description="Confidence scoring metrics")
    query_analysis: Optional[Dict[str, Any]] = Field(None, description="Query analysis results")
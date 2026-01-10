from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class FirefliesSyncRequest(BaseModel):
    """Request schema for Fireflies sync endpoint."""
    from_date: Optional[str] = Field(
        None,
        description="Start date for transcript filtering (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    to_date: Optional[str] = Field(
        None,
        description="End date for transcript filtering (YYYY-MM-DD)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    force_reingest: bool = Field(
        False,
        description="Force re-ingestion even if transcript already exists"
    )


class FirefliesSyncResponse(BaseModel):
    """Response schema for Fireflies sync endpoint."""
    status: str
    transcripts_found: int
    transcripts_ingested: int
    transcripts_skipped: int
    chunks_created: int


class IngestionStatus(BaseModel):
    """Response schema for ingestion status endpoint."""
    total_transcripts: int
    total_chunks: int
    last_ingested: Optional[str] = None
    collection_name: str

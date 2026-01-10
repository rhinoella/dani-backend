"""
Document schemas for upload and retrieval APIs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============== Request Schemas ==============

class DocumentUploadRequest(BaseModel):
    """Request metadata for document upload."""
    title: Optional[str] = Field(None, max_length=500, description="Document title")
    description: Optional[str] = Field(None, max_length=2000, description="Document description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Q4 Strategy Document",
                "description": "Strategic planning document for Q4 2025"
            }
        }


class DocumentUpdateRequest(BaseModel):
    """Request to update document metadata."""
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)


class DocumentListRequest(BaseModel):
    """Request parameters for listing documents."""
    status: Optional[DocumentStatus] = Field(None, description="Filter by status")
    file_type: Optional[DocumentType] = Field(None, description="Filter by file type")
    search: Optional[str] = Field(None, max_length=200, description="Search in filename/title")
    skip: int = Field(0, ge=0, description="Number of documents to skip")
    limit: int = Field(20, ge=1, le=100, description="Maximum documents to return")


# ============== Response Schemas ==============

class DocumentResponse(BaseModel):
    """Single document response."""
    id: str
    filename: str
    file_type: DocumentType
    file_size: int
    file_size_mb: float
    mime_type: Optional[str]
    title: Optional[str]
    description: Optional[str]
    status: DocumentStatus
    error_message: Optional[str]
    chunk_count: int
    total_tokens: int
    metadata: Dict[str, Any]
    storage_key: Optional[str] = None
    download_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]
    user_id: Optional[str]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "strategy_doc.pdf",
                "file_type": "pdf",
                "file_size": 1048576,
                "file_size_mb": 1.0,
                "mime_type": "application/pdf",
                "title": "Q4 Strategy Document",
                "description": "Strategic planning for Q4",
                "status": "completed",
                "chunk_count": 25,
                "total_tokens": 12500,
                "metadata": {"page_count": 10, "author": "John Doe"},
                "created_at": "2025-01-04T10:00:00Z",
                "updated_at": "2025-01-04T10:05:00Z",
                "processed_at": "2025-01-04T10:05:00Z",
                "user_id": "user-123"
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response after successful upload."""
    id: str
    filename: str
    file_type: DocumentType
    file_size: int
    status: DocumentStatus
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "strategy_doc.pdf",
                "file_type": "pdf",
                "file_size": 1048576,
                "status": "processing",
                "message": "Document uploaded successfully and is being processed"
            }
        }


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int
    has_more: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [],
                "total": 50,
                "skip": 0,
                "limit": 20,
                "has_more": True
            }
        }


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    id: str
    deleted: bool
    chunks_removed: int
    message: str


class DocumentProcessingStatus(BaseModel):
    """Status of document processing."""
    id: str
    filename: str
    status: DocumentStatus
    progress: Optional[int] = Field(None, description="Processing progress (0-100)")
    error_message: Optional[str]
    chunk_count: int
    processed_at: Optional[datetime]


class DocumentStatsResponse(BaseModel):
    """Document statistics for a user or system."""
    total_documents: int
    documents_by_status: Dict[str, int]
    documents_by_type: Dict[str, int]
    total_chunks: int
    total_tokens: int
    total_size_mb: float


# ============== Chunk-related Schemas ==============

class DocumentChunkResponse(BaseModel):
    """A chunk from a document stored in the vector database."""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any]
    score: Optional[float] = Field(None, description="Relevance score if from search")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk-123",
                "document_id": "doc-456",
                "text": "This is the content of the chunk...",
                "chunk_index": 0,
                "token_count": 350,
                "metadata": {"page": 1, "section": "Introduction"},
                "score": 0.89
            }
        }


class DocumentChunksResponse(BaseModel):
    """List of chunks from a document."""
    document_id: str
    filename: str
    chunks: List[DocumentChunkResponse]
    total_chunks: int


class DocumentDownloadUrlResponse(BaseModel):
    """Response containing a presigned download URL."""
    id: str
    filename: str
    download_url: str
    expires_in_seconds: int = Field(description="URL expiration time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "strategy_doc.pdf",
                "download_url": "https://bucket.s3.amazonaws.com/documents/...",
                "expires_in_seconds": 3600
            }
        }

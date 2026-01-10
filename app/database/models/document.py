"""
Document model for tracking uploaded files (PDF, DOCX, TXT).

Stores metadata about uploaded documents and links to vector store chunks.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import String, Text, Integer, BigInteger, DateTime, Index, func, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from app.database.models.base import Base

if TYPE_CHECKING:
    from app.database.models.user import User


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


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


class Document(Base):
    """
    Document model for tracking uploaded files.
    
    Each document is linked to the user who uploaded it
    and tracks the chunks stored in the vector database.
    """
    
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )
    
    # User who uploaded the document (optional for system uploads)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        comment="User who uploaded the document",
    )
    
    # File information
    filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Original filename",
    )
    file_type: Mapped[DocumentType] = mapped_column(
        SQLEnum(DocumentType, name="document_type"),
        nullable=False,
        index=True,
        comment="File type (pdf, docx, txt)",
    )
    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="File size in bytes",
    )
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="MIME type of the file",
    )
    
    # S3 Storage
    storage_key: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
        comment="S3 object key for the original file",
    )
    storage_bucket: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="S3 bucket name",
    )
    
    # Content metadata
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Document title (extracted or user-provided)",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="User-provided description",
    )
    
    # Processing status
    status: Mapped[DocumentStatus] = mapped_column(
        SQLEnum(DocumentStatus, name="document_status"),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
        comment="Processing status",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if processing failed",
    )
    
    # Vector store information
    collection_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="documents",
        comment="Qdrant collection name where chunks are stored",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of chunks created from this document",
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total tokens in the document",
    )
    
    # Additional metadata
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
        comment="Additional document metadata (pages, author, etc.)",
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When processing completed",
    )
    
    # Soft delete
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_documents_user_status", "user_id", "status"),
        Index("ix_documents_created_at", "created_at"),
        Index("ix_documents_filename", "filename"),
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
    
    @property
    def file_size_mb(self) -> float:
        """Return file size in MB."""
        return self.file_size / (1024 * 1024)
    
    @property
    def is_processed(self) -> bool:
        """Check if document has been processed successfully."""
        return self.status == DocumentStatus.COMPLETED

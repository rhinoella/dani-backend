"""
Document repository for database operations.
"""

from __future__ import annotations

from typing import Optional, List
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository
from app.database.models.document import Document, DocumentType, DocumentStatus


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(Document, session)
    
    async def get_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
        status: Optional[DocumentStatus] = None,
        file_type: Optional[DocumentType] = None,
        search: Optional[str] = None,
    ) -> List[Document]:
        """Get documents for a specific user with filtering."""
        query = select(Document).where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None)
        )
        
        if status:
            query = query.where(Document.status == status)
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        if search:
            search_filter = or_(
                Document.filename.ilike(f"%{search}%"),
                Document.title.ilike(f"%{search}%"),
            )
            query = query.where(search_filter)
        
        query = query.order_by(Document.created_at.desc())
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_by_user(
        self,
        user_id: str,
        status: Optional[DocumentStatus] = None,
        file_type: Optional[DocumentType] = None,
        search: Optional[str] = None,
    ) -> int:
        """Count documents for a user with filtering."""
        query = select(func.count()).select_from(Document).where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None)
        )
        
        if status:
            query = query.where(Document.status == status)
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        if search:
            search_filter = or_(
                Document.filename.ilike(f"%{search}%"),
                Document.title.ilike(f"%{search}%"),
            )
            query = query.where(search_filter)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_all_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        status: Optional[DocumentStatus] = None,
        file_type: Optional[DocumentType] = None,
        search: Optional[str] = None,
    ) -> List[Document]:
        """Get all documents with filtering (admin view)."""
        query = select(Document).where(Document.deleted_at.is_(None))
        
        if status:
            query = query.where(Document.status == status)
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        if search:
            search_filter = or_(
                Document.filename.ilike(f"%{search}%"),
                Document.title.ilike(f"%{search}%"),
            )
            query = query.where(search_filter)
        
        query = query.order_by(Document.created_at.desc())
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_all(
        self,
        status: Optional[DocumentStatus] = None,
        file_type: Optional[DocumentType] = None,
        search: Optional[str] = None,
    ) -> int:
        """Count all documents with filtering."""
        query = select(func.count()).select_from(Document).where(
            Document.deleted_at.is_(None)
        )
        
        if status:
            query = query.where(Document.status == status)
        
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        if search:
            search_filter = or_(
                Document.filename.ilike(f"%{search}%"),
                Document.title.ilike(f"%{search}%"),
            )
            query = query.where(search_filter)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> Optional[Document]:
        """Update document processing status."""
        from datetime import datetime, timezone
        
        update_data = {"status": status}
        
        if error_message is not None:
            update_data["error_message"] = error_message
        
        if chunk_count is not None:
            update_data["chunk_count"] = chunk_count
        
        if total_tokens is not None:
            update_data["total_tokens"] = total_tokens
        
        if status == DocumentStatus.COMPLETED:
            update_data["processed_at"] = datetime.now(timezone.utc)
        
        return await self.update(document_id, **update_data)
    
    async def get_user_stats(self, user_id: str) -> dict:
        """Get document statistics for a user."""
        # Total count
        total = await self.count_by_user(user_id)
        
        # Count by status
        status_counts = {}
        for status in DocumentStatus:
            count = await self.count_by_user(user_id, status=status)
            status_counts[status.value] = count
        
        # Count by type
        type_counts = {}
        for file_type in DocumentType:
            count = await self.count_by_user(user_id, file_type=file_type)
            type_counts[file_type.value] = count
        
        # Sum chunks and tokens
        query = select(
            func.sum(Document.chunk_count),
            func.sum(Document.total_tokens),
            func.sum(Document.file_size),
        ).where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None),
            Document.status == DocumentStatus.COMPLETED,
        )
        
        result = await self.session.execute(query)
        row = result.one()
        
        return {
            "total_documents": total,
            "documents_by_status": status_counts,
            "documents_by_type": type_counts,
            "total_chunks": row[0] or 0,
            "total_tokens": row[1] or 0,
            "total_size_mb": (row[2] or 0) / (1024 * 1024),
        }
    
    async def get_pending_documents(self, limit: int = 10) -> List[Document]:
        """Get documents waiting to be processed."""
        query = select(Document).where(
            Document.status == DocumentStatus.PENDING,
            Document.deleted_at.is_(None),
        ).order_by(Document.created_at.asc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())

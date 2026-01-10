"""
Tests for Document Repository.

Tests CRUD operations for document records.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import uuid

from app.repositories.document_repository import DocumentRepository
from app.database.models.document import Document, DocumentType, DocumentStatus


# ============== Fixtures ==============

@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_document():
    """Create a mock document entity."""
    doc = MagicMock(spec=Document)
    doc.id = str(uuid.uuid4())
    doc.user_id = str(uuid.uuid4())
    doc.filename = "test_doc_abc123.pdf"
    doc.original_filename = "test_document.pdf"
    doc.title = "Test Document"
    doc.description = "A test document"
    doc.file_type = DocumentType.PDF
    doc.file_size = 1024
    doc.mime_type = "application/pdf"
    doc.status = DocumentStatus.COMPLETED
    doc.chunk_count = 5
    doc.total_tokens = 500
    doc.collection_name = "documents"
    doc.metadata = {}
    doc.created_at = datetime.now(timezone.utc)
    doc.updated_at = datetime.now(timezone.utc)
    doc.deleted_at = None
    doc.error_message = None
    return doc


@pytest.fixture
def document_repository(mock_session):
    """Create DocumentRepository instance."""
    return DocumentRepository(mock_session)


# ============== Tests ==============

class TestDocumentRepositoryInit:
    """Tests for DocumentRepository initialization."""
    
    def test_init_stores_session(self, mock_session):
        """Test that init stores session."""
        repo = DocumentRepository(mock_session)
        assert repo.session == mock_session
        assert repo.model == Document


class TestDocumentRepositoryGetByUser:
    """Tests for get_by_user method."""
    
    @pytest.mark.asyncio
    async def test_get_by_user_returns_documents(self, document_repository, mock_session, mock_document):
        """Test get_by_user returns user's documents."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_by_user(mock_document.user_id)
        
        assert result == [mock_document]
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_user_with_status_filter(self, document_repository, mock_session, mock_document):
        """Test get_by_user with status filter."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_by_user(
            mock_document.user_id,
            status=DocumentStatus.COMPLETED
        )
        
        assert result == [mock_document]
    
    @pytest.mark.asyncio
    async def test_get_by_user_with_file_type_filter(self, document_repository, mock_session, mock_document):
        """Test get_by_user with file type filter."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_by_user(
            mock_document.user_id,
            file_type=DocumentType.PDF
        )
        
        assert result == [mock_document]
    
    @pytest.mark.asyncio
    async def test_get_by_user_with_pagination(self, document_repository, mock_session, mock_document):
        """Test get_by_user with pagination."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_by_user(
            mock_document.user_id,
            skip=10,
            limit=20
        )
        
        assert result == [mock_document]
    
    @pytest.mark.asyncio
    async def test_get_by_user_empty_result(self, document_repository, mock_session):
        """Test get_by_user returns empty list when no documents."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_by_user("nonexistent-user")
        
        assert result == []


class TestDocumentRepositoryCountByUser:
    """Tests for count_by_user method."""
    
    @pytest.mark.asyncio
    async def test_count_by_user_returns_count(self, document_repository, mock_session):
        """Test count_by_user returns document count."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.count_by_user("user-123")
        
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_count_by_user_with_status_filter(self, document_repository, mock_session):
        """Test count_by_user with status filter."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.count_by_user(
            "user-123",
            status=DocumentStatus.COMPLETED
        )
        
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_count_by_user_zero(self, document_repository, mock_session):
        """Test count_by_user returns 0 when no documents."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.count_by_user("nonexistent-user")
        
        assert result == 0


class TestDocumentRepositoryUpdateStatus:
    """Tests for update_status method."""
    
    @pytest.mark.asyncio
    async def test_update_status_success(self, document_repository, mock_session, mock_document):
        """Test update_status updates document status."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.update_status(
            mock_document.id,
            DocumentStatus.PROCESSING
        )
        
        assert result is not None
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_status_with_error(self, document_repository, mock_session, mock_document):
        """Test update_status with error message."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.update_status(
            mock_document.id,
            DocumentStatus.FAILED,
            error_message="Processing failed"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_update_status_not_found(self, document_repository, mock_session):
        """Test update_status returns None when document not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.update_status(
            "nonexistent-id",
            DocumentStatus.PROCESSING
        )
        
        assert result is None


class TestDocumentRepositoryGetUserStats:
    """Tests for get_user_stats method."""
    
    @pytest.mark.asyncio
    async def test_get_user_stats_returns_stats(self, document_repository, mock_session):
        """Test get_user_stats returns statistics."""
        # Mock the results for counts by status
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (DocumentStatus.COMPLETED, 5),
            (DocumentStatus.PENDING, 2),
            (DocumentStatus.FAILED, 1),
        ]
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_user_stats("user-123")
        
        assert isinstance(result, dict)


class TestDocumentRepositoryGetPendingDocuments:
    """Tests for get_pending_documents method."""
    
    @pytest.mark.asyncio
    async def test_get_pending_documents(self, document_repository, mock_session, mock_document):
        """Test get_pending_documents returns pending documents."""
        mock_document.status = DocumentStatus.PENDING
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_pending_documents()
        
        assert result == [mock_document]
    
    @pytest.mark.asyncio
    async def test_get_pending_documents_with_limit(self, document_repository, mock_session, mock_document):
        """Test get_pending_documents with limit."""
        mock_document.status = DocumentStatus.PENDING
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_document]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.get_pending_documents(limit=5)
        
        assert result == [mock_document]


class TestDocumentRepositorySoftDelete:
    """Tests for soft_delete method inherited from BaseRepository."""
    
    @pytest.mark.asyncio
    async def test_soft_delete_sets_deleted_at(self, document_repository, mock_session, mock_document):
        """Test soft_delete sets deleted_at timestamp."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_document
        mock_session.execute.return_value = mock_result
        
        result = await document_repository.soft_delete(mock_document.id)
        
        assert result is True
        assert mock_document.deleted_at is not None

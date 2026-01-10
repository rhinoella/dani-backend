"""
Tests for Document Service.

Tests business logic for document upload and processing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import uuid

from app.services.document_service import DocumentService
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
    return session


@pytest.fixture
def mock_document():
    """Create a mock document entity with proper values."""
    doc = MagicMock(spec=Document)
    doc.id = str(uuid.uuid4())
    doc.user_id = "user-123"
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
    doc.metadata_ = {}  # SQLAlchemy uses metadata_
    doc.created_at = datetime.now(timezone.utc)
    doc.updated_at = datetime.now(timezone.utc)
    doc.deleted_at = None
    doc.error_message = None
    doc.processing_started_at = None
    doc.processing_completed_at = None
    return doc


@pytest.fixture
def document_service(mock_session):
    """Create DocumentService instance with mocked dependencies."""
    with patch('app.services.document_service.DocumentRepository') as mock_repo_class, \
         patch('app.services.document_service.OllamaEmbeddingClient') as mock_embedder_class, \
         patch('app.services.document_service.QdrantStore') as mock_store_class, \
         patch('app.services.document_service.TokenChunker') as mock_chunker_class:
        
        # Setup mocks
        mock_repo = AsyncMock()
        mock_repo_class.return_value = mock_repo
        
        mock_embedder = AsyncMock()
        mock_embedder_class.return_value = mock_embedder
        
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        
        mock_chunker = MagicMock()
        mock_chunker_class.return_value = mock_chunker
        
        service = DocumentService(mock_session)
        service.repository = mock_repo
        service.embedder = mock_embedder
        service.store = mock_store
        service.chunker = mock_chunker
        
        return service


# ============== Tests ==============

class TestDocumentServiceInit:
    """Tests for DocumentService initialization."""
    
    def test_init_creates_dependencies(self, mock_session):
        """Test that init creates all required dependencies."""
        with patch('app.services.document_service.DocumentRepository') as mock_repo, \
             patch('app.services.document_service.OllamaEmbeddingClient'), \
             patch('app.services.document_service.QdrantStore'), \
             patch('app.services.document_service.TokenChunker'):
            
            service = DocumentService(mock_session)
            
            # Service stores session
            assert service.session == mock_session
            # Repository was created
            mock_repo.assert_called_once_with(mock_session)


class TestDocumentServiceDetectFileType:
    """Tests for file type detection."""
    
    def test_detect_file_type_pdf(self, document_service):
        """Test file type detection for PDF."""
        file_type = document_service._detect_file_type(
            "document.pdf",
            "application/pdf"
        )
        assert file_type == DocumentType.PDF
    
    def test_detect_file_type_docx(self, document_service):
        """Test file type detection for DOCX."""
        file_type = document_service._detect_file_type(
            "document.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert file_type == DocumentType.DOCX
    
    def test_detect_file_type_txt(self, document_service):
        """Test file type detection for TXT."""
        file_type = document_service._detect_file_type(
            "document.txt",
            "text/plain"
        )
        assert file_type == DocumentType.TXT
    
    def test_detect_file_type_by_extension(self, document_service):
        """Test file type detection by extension when mime is unknown."""
        file_type = document_service._detect_file_type(
            "document.pdf",
            "application/octet-stream"
        )
        assert file_type == DocumentType.PDF
    
    def test_detect_file_type_unsupported(self, document_service):
        """Test unsupported file type raises error."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            document_service._detect_file_type(
                "document.xyz",
                "application/unknown"
            )


class TestDocumentServiceUpload:
    """Tests for document upload."""
    
    @pytest.mark.asyncio
    async def test_upload_document_empty_file(self, document_service):
        """Test upload with empty file raises error."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            await document_service.upload_document(
                filename="test.txt",
                file_content=b"",
                content_type="text/plain"
            )
    
    @pytest.mark.asyncio
    async def test_upload_document_unsupported_type(self, document_service):
        """Test upload with unsupported file type raises error."""
        with pytest.raises(ValueError, match="Unsupported"):
            await document_service.upload_document(
                filename="test.xyz",
                file_content=b"content",
                content_type="application/unknown"
            )


class TestDocumentServiceGetDocument:
    """Tests for getting documents."""
    
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, document_service):
        """Test get_document returns None when not found."""
        document_service.repository.get_by_id = AsyncMock(return_value=None)
        
        result = await document_service.get_document("nonexistent-id")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_document_wrong_user(self, document_service, mock_document):
        """Test get_document returns None for wrong user."""
        mock_document.user_id = "other-user"
        document_service.repository.get_by_id = AsyncMock(return_value=mock_document)
        
        result = await document_service.get_document(
            mock_document.id,
            user_id="user-123"
        )
        
        assert result is None


class TestDocumentServiceListDocuments:
    """Tests for listing documents."""
    
    @pytest.mark.asyncio
    async def test_list_documents_empty(self, document_service):
        """Test list_documents returns empty when no documents."""
        document_service.repository.get_by_user = AsyncMock(return_value=[])
        document_service.repository.count_by_user = AsyncMock(return_value=0)
        
        result = await document_service.list_documents(user_id="user-123")
        
        assert result.total == 0
        assert result.documents == []


class TestDocumentServiceDeleteDocument:
    """Tests for deleting documents."""
    
    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, document_service):
        """Test delete_document returns None when not found."""
        document_service.repository.get_by_id = AsyncMock(return_value=None)
        
        result = await document_service.delete_document(
            "nonexistent-id",
            user_id="user-123"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_document_wrong_user(self, document_service, mock_document):
        """Test delete_document returns None for wrong user."""
        mock_document.user_id = "other-user"
        document_service.repository.get_by_id = AsyncMock(return_value=mock_document)
        
        result = await document_service.delete_document(
            mock_document.id,
            user_id="user-123"
        )
        
        assert result is None


class TestDocumentServiceGetChunks:
    """Tests for getting document chunks."""
    
    @pytest.mark.asyncio
    async def test_get_document_chunks_not_found(self, document_service):
        """Test get_document_chunks returns None when document not found."""
        document_service.repository.get_by_id = AsyncMock(return_value=None)
        
        result = await document_service.get_document_chunks(
            "nonexistent-id",
            user_id="user-123"
        )
        
        assert result is None

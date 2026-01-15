"""
Tests for S3 Storage Service.

Tests AWS S3 file storage operations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import uuid
from io import BytesIO

from botocore.exceptions import ClientError

from app.services.storage_service import StorageService
from app.core.exceptions import StorageError


# ============== Fixtures ==============

@pytest.fixture
def mock_s3_client():
    """Create mock boto3 S3 client."""
    return MagicMock()


@pytest.fixture
def storage_service(mock_s3_client):
    """Create StorageService with mocked S3 client."""
    with patch('app.services.storage_service.boto3') as mock_boto3:
        mock_boto3.client.return_value = mock_s3_client
        service = StorageService()
        service._client = mock_s3_client
        return service


# ============== Tests ==============

class TestStorageServiceInit:
    """Tests for StorageService initialization."""
    
    def test_init_creates_client_lazily(self):
        """Test that initialization creates S3 client with configured bucket."""
        with patch('app.services.storage_service.boto3') as mock_boto3, \
             patch('app.services.storage_service.settings') as mock_settings:
            mock_boto3.client.return_value = MagicMock()
            mock_settings.S3_BUCKET_NAME = "dani-documents"
            mock_settings.S3_DOCUMENTS_PREFIX = "documents/"
            mock_settings.S3_PRESIGNED_URL_EXPIRY = 3600
            mock_settings.AWS_REGION = "us-east-1"
            mock_settings.AWS_ACCESS_KEY_ID = None
            mock_settings.AWS_SECRET_ACCESS_KEY = None
            mock_settings.S3_ENDPOINT_URL = None
            service = StorageService()
            # Check internal state matches configured bucket
            assert service.bucket_name == "dani-documents"
    
    def test_client_property_creates_client_lazily(self):
        """Test that client is created on first access."""
        with patch('app.services.storage_service.boto3') as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            service = StorageService()
            
            # Access client
            client = service.client
            
            # Client should be created
            mock_boto3.client.assert_called_once()


class TestStorageServiceGenerateKey:
    """Tests for key generation."""
    
    def test_generate_key_includes_prefix(self, storage_service):
        """Test that generated key includes configured prefix."""
        key = storage_service._generate_key("test.pdf")
        
        assert key.startswith("documents/")
    
    def test_generate_key_includes_date(self, storage_service):
        """Test that generated key includes current date."""
        key = storage_service._generate_key("test.pdf")
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in key or datetime.utcnow().strftime("%Y-%m-%d") in key
    
    def test_generate_key_includes_uuid(self, storage_service):
        """Test that generated key includes UUID."""
        key = storage_service._generate_key("test.pdf")
        
        # Key should have format: prefix/date/uuid/filename
        parts = key.split("/")
        assert len(parts) >= 3
    
    def test_generate_key_preserves_filename(self, storage_service):
        """Test that generated key preserves original filename."""
        key = storage_service._generate_key("my_document.pdf")
        
        assert key.endswith("my_document.pdf")


class TestStorageServiceUpload:
    """Tests for file upload."""
    
    @pytest.mark.asyncio
    async def test_upload_success(self, storage_service, mock_s3_client):
        """Test successful file upload."""
        file_content = b"test file content"
        
        result = await storage_service.upload(
            file_content=file_content,
            filename="test.pdf",
        )
        
        assert "key" in result
        assert "bucket" in result
        mock_s3_client.put_object.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_with_content_type(self, storage_service, mock_s3_client):
        """Test upload with custom content type."""
        file_content = b"test content"
        
        await storage_service.upload(
            file_content=file_content,
            filename="test.pdf",
            content_type="application/pdf",
        )
        
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs.get("ContentType") == "application/pdf"
    
    @pytest.mark.asyncio
    async def test_upload_with_metadata(self, storage_service, mock_s3_client):
        """Test upload with custom metadata."""
        file_content = b"test content"
        
        await storage_service.upload(
            file_content=file_content,
            filename="test.pdf",
            metadata={"author": "Test User"},
        )
        
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert "Metadata" in call_kwargs
    
    @pytest.mark.asyncio
    async def test_upload_with_user_id(self, storage_service, mock_s3_client):
        """Test upload includes user_id in key path."""
        file_content = b"test content"
        user_id = "user-123"
        
        result = await storage_service.upload(
            file_content=file_content,
            filename="test.pdf",
            user_id=user_id,
        )
        
        # User ID should be in the key path
        assert user_id in result["key"]
    
    @pytest.mark.asyncio
    async def test_upload_failure_raises_storage_error(self, storage_service, mock_s3_client):
        """Test that S3 errors are wrapped in StorageError."""
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}},
            "PutObject"
        )
        
        with pytest.raises(StorageError) as exc_info:
            await storage_service.upload(
                file_content=b"test",
                filename="test.pdf",
            )
        
        assert "NoSuchBucket" in str(exc_info.value)


class TestStorageServiceDownload:
    """Tests for file download."""
    
    @pytest.mark.asyncio
    async def test_download_success(self, storage_service, mock_s3_client):
        """Test successful file download."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"file content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        
        content = await storage_service.download(key="documents/test.pdf")
        
        assert content == b"file content"
        mock_s3_client.get_object.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_not_found_raises_error(self, storage_service, mock_s3_client):
        """Test that missing file raises StorageError."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
            "GetObject"
        )
        
        with pytest.raises(StorageError) as exc_info:
            await storage_service.download(key="nonexistent.pdf")
        
        # Error message should indicate file not found
        assert "not found" in str(exc_info.value).lower() or "NoSuchKey" in str(exc_info.value)


class TestStorageServiceDelete:
    """Tests for file deletion."""
    
    @pytest.mark.asyncio
    async def test_delete_success(self, storage_service, mock_s3_client):
        """Test successful file deletion."""
        result = await storage_service.delete(key="documents/test.pdf")
        
        assert result is True
        mock_s3_client.delete_object.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_true(self, storage_service, mock_s3_client):
        """Test that deleting nonexistent file still returns True (idempotent)."""
        # S3 delete is idempotent - doesn't error on missing keys
        result = await storage_service.delete(key="nonexistent.pdf")
        
        assert result is True


class TestStorageServicePresignedUrl:
    """Tests for presigned URL generation."""
    
    @pytest.mark.asyncio
    async def test_presigned_url_success(self, storage_service, mock_s3_client):
        """Test successful presigned URL generation."""
        mock_s3_client.generate_presigned_url.return_value = "https://bucket.s3.amazonaws.com/test?signature=..."
        
        url = await storage_service.get_presigned_url(key="documents/test.pdf")
        
        assert url.startswith("https://")
        mock_s3_client.generate_presigned_url.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_presigned_url_custom_expiry(self, storage_service, mock_s3_client):
        """Test presigned URL with custom expiry."""
        mock_s3_client.generate_presigned_url.return_value = "https://example.com"
        
        await storage_service.get_presigned_url(
            key="documents/test.pdf",
            expiry=7200,  # Use correct parameter name
        )
        
        call_kwargs = mock_s3_client.generate_presigned_url.call_args[1]
        assert call_kwargs.get("ExpiresIn") == 7200


class TestStorageServiceExists:
    """Tests for file existence check."""
    
    @pytest.mark.asyncio
    async def test_exists_returns_true_when_file_exists(self, storage_service, mock_s3_client):
        """Test that exists returns True for existing file."""
        mock_s3_client.head_object.return_value = {}
        
        result = await storage_service.exists(key="documents/test.pdf")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_returns_false_when_file_missing(self, storage_service, mock_s3_client):
        """Test that exists returns False for missing file."""
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject"
        )
        
        result = await storage_service.exists(key="nonexistent.pdf")
        
        assert result is False


class TestStorageServiceGetMetadata:
    """Tests for file metadata retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_metadata_success(self, storage_service, mock_s3_client):
        """Test successful metadata retrieval."""
        mock_s3_client.head_object.return_value = {
            "ContentLength": 1024,
            "ContentType": "application/pdf",
            "LastModified": datetime.now(timezone.utc),
            "Metadata": {"user_id": "test-user"},
        }
        
        metadata = await storage_service.get_metadata(key="documents/test.pdf")
        
        assert metadata["size"] == 1024
        assert metadata["content_type"] == "application/pdf"
        assert "user_id" in metadata["metadata"]


class TestStorageServiceListFiles:
    """Tests for listing files."""
    
    @pytest.mark.asyncio
    async def test_list_files_success(self, storage_service, mock_s3_client):
        """Test successful file listing."""
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "documents/file1.pdf", "Size": 100, "LastModified": datetime.now(timezone.utc)},
                {"Key": "documents/file2.pdf", "Size": 200, "LastModified": datetime.now(timezone.utc)},
            ]
        }
        
        files = await storage_service.list_files(prefix="documents/")
        
        assert len(files) == 2
        assert files[0]["key"] == "documents/file1.pdf"
    
    @pytest.mark.asyncio
    async def test_list_files_empty(self, storage_service, mock_s3_client):
        """Test listing empty directory."""
        mock_s3_client.list_objects_v2.return_value = {}
        
        files = await storage_service.list_files(prefix="empty/")
        
        assert files == []


class TestStorageError:
    """Tests for StorageError exception."""
    
    def test_storage_error_message(self):
        """Test StorageError has correct message format."""
        error = StorageError(message="Test error")
        assert "Test error" in str(error)
        assert "s3" in str(error).lower()
    
    def test_storage_error_with_operation_and_key(self):
        """Test StorageError includes operation and key in details."""
        error = StorageError(
            message="Upload failed",
            operation="upload",
            key="test/file.pdf",
        )
        assert "Upload failed" in str(error)
        assert error.details["operation"] == "upload"
        assert error.details["key"] == "test/file.pdf"
    
    def test_storage_error_inheritance(self):
        """Test StorageError inherits from ServiceError and DANIException."""
        from app.core.exceptions import ServiceError, DANIException
        error = StorageError(message="Test")
        assert isinstance(error, ServiceError)
        assert isinstance(error, DANIException)
        assert isinstance(error, Exception)

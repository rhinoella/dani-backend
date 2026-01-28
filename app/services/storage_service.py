"""
AWS S3 Storage Service.

Handles document file upload, download, and deletion from S3.
"""

from __future__ import annotations

import logging
import uuid
import mimetypes
from typing import Optional, BinaryIO, Dict, Any
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from app.core.config import settings
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class StorageService:
    """
    Service for managing file storage in AWS S3.
    
    Supports:
    - Uploading documents with automatic key generation
    - Downloading documents by key
    - Generating presigned URLs for direct access
    - Deleting documents
    - Listing documents by prefix
    """
    
    def __init__(self):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.prefix = settings.S3_DOCUMENTS_PREFIX
        self.presigned_expiry = settings.S3_PRESIGNED_URL_EXPIRY
        
        # Configure S3 client
        config = Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        
        client_kwargs: Dict[str, Any] = {
            'service_name': 's3',
            'region_name': settings.AWS_REGION,
            'config': config,
        }
        
        # Add credentials if provided (otherwise uses IAM role / env vars)
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            client_kwargs['aws_access_key_id'] = settings.AWS_ACCESS_KEY_ID
            client_kwargs['aws_secret_access_key'] = settings.AWS_SECRET_ACCESS_KEY
        
        # Support S3-compatible services (MinIO, LocalStack)
        if settings.S3_ENDPOINT_URL:
            client_kwargs['endpoint_url'] = settings.S3_ENDPOINT_URL
        
        self.client = boto3.client(**client_kwargs)
        
        logger.info(f"StorageService initialized for bucket: {self.bucket_name}")
    
    def _generate_key(self, filename: str, user_id: Optional[str] = None) -> str:
        """
        Generate a unique S3 key for the file.
        
        Format: {prefix}{user_id}/{date}/{uuid}_{filename}
        Example: documents/user123/2025-01-04/abc123_report.pdf
        """
        date_path = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        unique_id = str(uuid.uuid4())[:8]
        
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-").strip()
        if not safe_filename:
            safe_filename = "file"
        
        if user_id:
            return f"{self.prefix}{user_id}/{date_path}/{unique_id}_{safe_filename}"
        return f"{self.prefix}{date_path}/{unique_id}_{safe_filename}"
    
    def _guess_content_type(self, filename: str) -> str:
        """Guess MIME type from filename."""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"
    
    async def upload(
        self,
        file_content: bytes,
        filename: str,
        user_id: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to S3.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            user_id: Optional user ID for path organization
            content_type: MIME type (auto-detected if not provided)
            metadata: Additional metadata to store with the file
            
        Returns:
            Dict with upload details (key, url, size, etc.)
            
        Raises:
            StorageError: If upload fails
        """
        key = self._generate_key(filename, user_id)
        content_type = content_type or self._guess_content_type(filename)
        
        extra_args: Dict[str, Any] = {
            'ContentType': content_type,
        }
        
        # Add custom metadata
        if metadata:
            extra_args['Metadata'] = {
                k: str(v)[:1024] for k, v in metadata.items()  # S3 metadata value limit
            }
        
        try:
            logger.info(f"Uploading to S3: {key} ({len(file_content)} bytes)")
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_content,
                **extra_args,
            )
            
            # Get the object URL
            if settings.S3_ENDPOINT_URL:
                url = f"{settings.S3_ENDPOINT_URL}/{self.bucket_name}/{key}"
            else:
                url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
            
            logger.info(f"Upload successful: {key}")
            
            return {
                "key": key,
                "bucket": self.bucket_name,
                "url": url,
                "size": len(file_content),
                "content_type": content_type,
                "filename": filename,
            }
            
        except NoCredentialsError as e:
            logger.error(f"S3 credentials not configured: {e}")
            raise StorageError(
                message="S3 credentials not configured",
                operation="upload",
                key=key,
            ) from e
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"S3 upload failed [{error_code}]: {e}")
            raise StorageError(
                message=f"Failed to upload file: {error_code}",
                operation="upload",
                key=key,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected upload error: {e}")
            raise StorageError(
                message=f"Upload failed: {str(e)}",
                operation="upload",
                key=key,
            ) from e
    
    async def download(self, key: str) -> bytes:
        """
        Download a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            File content as bytes
            
        Raises:
            StorageError: If download fails or file not found
        """
        try:
            logger.info(f"Downloading from S3: {key}")
            
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=key,
            )
            
            content = response['Body'].read()
            logger.info(f"Downloaded {len(content)} bytes from {key}")
            
            return content
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                logger.warning(f"File not found: {key}")
                raise StorageError(
                    message=f"File not found: {key}",
                    operation="download",
                    key=key,
                ) from e
            logger.error(f"S3 download failed [{error_code}]: {e}")
            raise StorageError(
                message=f"Failed to download file: {error_code}",
                operation="download",
                key=key,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected download error: {e}")
            raise StorageError(
                message=f"Download failed: {str(e)}",
                operation="download",
                key=key,
            ) from e
    
    async def get_presigned_url(
        self,
        key: str,
        expiry: Optional[int] = None,
        filename: Optional[str] = None,
        inline: bool = False,
    ) -> str:
        """
        Generate a presigned URL for direct download or inline viewing.
        
        Args:
            key: S3 object key
            expiry: URL expiration time in seconds (default from config)
            filename: Optional filename for Content-Disposition header
            inline: If True, use 'inline' disposition (for viewing in browser)
                   If False, use 'attachment' disposition (for download)
            
        Returns:
            Presigned URL string
            
        Raises:
            StorageError: If URL generation fails
        """
        expiry = expiry or self.presigned_expiry
        
        params: Dict[str, Any] = {
            'Bucket': self.bucket_name,
            'Key': key,
        }
        
        # Add Content-Disposition header
        if filename:
            disposition = 'inline' if inline else 'attachment'
            params['ResponseContentDisposition'] = f'{disposition}; filename="{filename}"'
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params=params,
                ExpiresIn=expiry,
            )
            
            logger.debug(f"Generated presigned URL for {key} (expires in {expiry}s)")
            return url
            
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise StorageError(
                message="Failed to generate download URL",
                operation="presign",
                key=key,
            ) from e
    
    async def delete(self, key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageError: If deletion fails
        """
        try:
            logger.info(f"Deleting from S3: {key}")
            
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=key,
            )
            
            logger.info(f"Deleted: {key}")
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                logger.warning(f"File not found for deletion: {key}")
                return False
            logger.error(f"S3 delete failed [{error_code}]: {e}")
            raise StorageError(
                message=f"Failed to delete file: {error_code}",
                operation="delete",
                key=key,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected delete error: {e}")
            raise StorageError(
                message=f"Delete failed: {str(e)}",
                operation="delete",
                key=key,
            ) from e
    
    async def exists(self, key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if exists, False otherwise
        """
        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=key,
            )
            return True
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                return False
            raise StorageError(
                message="Failed to check file existence",
                operation="exists",
                key=key,
            ) from e
    
    async def get_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata for a file without downloading content.
        
        Args:
            key: S3 object key
            
        Returns:
            Dict with file metadata
            
        Raises:
            StorageError: If file not found or request fails
        """
        try:
            response = self.client.head_object(
                Bucket=self.bucket_name,
                Key=key,
            )
            
            return {
                "key": key,
                "size": response.get('ContentLength', 0),
                "content_type": response.get('ContentType', 'application/octet-stream'),
                "last_modified": response.get('LastModified'),
                "etag": response.get('ETag', '').strip('"'),
                "metadata": response.get('Metadata', {}),
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code in ('404', 'NoSuchKey'):
                raise StorageError(
                    message=f"File not found: {key}",
                    operation="get_metadata",
                    key=key,
                ) from e
            raise StorageError(
                message=f"Failed to get metadata: {error_code}",
                operation="get_metadata",
                key=key,
            ) from e
    
    async def list_files(
        self,
        prefix: Optional[str] = None,
        max_keys: int = 1000,
    ) -> list[Dict[str, Any]]:
        """
        List files in the bucket with optional prefix filter.
        
        Args:
            prefix: Key prefix to filter by
            max_keys: Maximum number of results
            
        Returns:
            List of file info dicts
        """
        prefix = prefix or self.prefix
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified'],
                    "etag": obj.get('ETag', '').strip('"'),
                })
            
            return files
            
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            raise StorageError(
                message="Failed to list files",
                operation="list",
                key=prefix,
            ) from e
    
    def ensure_bucket_exists(self) -> bool:
        """
        Check if bucket exists, optionally create it.
        
        Returns:
            True if bucket exists or was created
        """
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket exists: {self.bucket_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                logger.warning(f"Bucket does not exist: {self.bucket_name}")
                return False
            elif error_code == '403':
                logger.error(f"Access denied to bucket: {self.bucket_name}")
                raise StorageError(
                    message="Access denied to S3 bucket",
                    operation="check_bucket",
                )
            raise StorageError(
                message=f"Failed to check bucket: {error_code}",
                operation="check_bucket",
            ) from e


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service

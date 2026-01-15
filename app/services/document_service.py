"""
Document service for handling file uploads and processing.

Orchestrates:
- File validation
- S3 storage
- Text extraction (PDF, DOCX, TXT)
- Chunking
- Embedding
- Vector storage
- Database tracking
"""

from __future__ import annotations

import logging
import mimetypes
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, BinaryIO
from uuid import uuid5, NAMESPACE_DNS

from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.http import models as qm

from app.core.config import settings
from app.core.exceptions import StorageError
from app.database.models.document import Document, DocumentType, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docx_loader import DOCXLoader
from app.ingestion.loaders.txt_loader import TXTLoader
from app.ingestion.chunker import TokenChunker
from app.embeddings.client import OllamaEmbeddingClient
from app.vectorstore.qdrant import QdrantStore
from app.services.storage_service import StorageService, get_storage_service
from app.schemas.document import (
    DocumentResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentStatsResponse,
    DocumentChunkResponse,
    DocumentChunksResponse,
    DocumentType as SchemaDocumentType,
    DocumentStatus as SchemaDocumentStatus,
)

logger = logging.getLogger(__name__)


# MIME type to DocumentType mapping
MIME_TO_DOCTYPE = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/msword": DocumentType.DOCX,
    "text/plain": DocumentType.TXT,
    "text/markdown": DocumentType.TXT,
}

# Extension to DocumentType mapping
EXT_TO_DOCTYPE = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".doc": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".text": DocumentType.TXT,
    ".md": DocumentType.TXT,
    ".markdown": DocumentType.TXT,
}

# Maximum file sizes (in bytes)
MAX_FILE_SIZES = {
    DocumentType.PDF: 50 * 1024 * 1024,    # 50MB
    DocumentType.DOCX: 25 * 1024 * 1024,   # 25MB
    DocumentType.TXT: 10 * 1024 * 1024,    # 10MB
}


def _stable_point_id(*parts: str) -> str:
    """Generate a stable UUID5 from multiple string parts."""
    composite = ":".join(parts)
    return str(uuid5(NAMESPACE_DNS, composite))


class DocumentService:
    """
    Service for handling document uploads and processing.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = DocumentRepository(session)
        
        # Loaders
        self.pdf_loader = PDFLoader()
        self.docx_loader = DOCXLoader()
        self.txt_loader = TXTLoader()
        
        # Processing components
        self.chunker = TokenChunker(chunk_size=350, overlap=100)
        self.embedder = OllamaEmbeddingClient()
        self.store = QdrantStore()
        self.collection = settings.QDRANT_COLLECTION_DOCUMENTS
        
        # S3 Storage (lazy initialization)
        self._storage: Optional[StorageService] = None
    
    @property
    def storage(self) -> StorageService:
        """Get or create storage service."""
        if self._storage is None:
            self._storage = get_storage_service()
        return self._storage
    
    def _detect_file_type(self, filename: str, content_type: Optional[str]) -> DocumentType:
        """Detect document type from filename and content type."""
        # Try extension first
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in EXT_TO_DOCTYPE:
            return EXT_TO_DOCTYPE[ext]
        
        # Try content type
        if content_type and content_type in MIME_TO_DOCTYPE:
            return MIME_TO_DOCTYPE[content_type]
        
        # Try guessing from filename
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type and guessed_type in MIME_TO_DOCTYPE:
            return MIME_TO_DOCTYPE[guessed_type]
        
        raise ValueError(f"Unsupported file type: {filename}")
    
    def _validate_file(self, filename: str, file_size: int, file_type: DocumentType) -> None:
        """Validate file before processing."""
        # Check size
        max_size = MAX_FILE_SIZES.get(file_type, 10 * 1024 * 1024)
        if file_size > max_size:
            raise ValueError(
                f"File too large: {file_size / (1024*1024):.1f}MB "
                f"(max: {max_size / (1024*1024):.0f}MB for {file_type.value})"
            )
        
        # Check for empty file
        if file_size == 0:
            raise ValueError("File is empty")
    
    async def upload_document(
        self,
        filename: str,
        file_content: bytes,
        content_type: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> DocumentUploadResponse:
        """
        Uploads a document to S3 and creates a database record.
        
        Args:
            filename: Original filename
            file_content: Raw file bytes
            content_type: MIME type (optional)
            title: Document title (optional)
            description: Document description (optional)
            user_id: ID of uploading user (optional)
            
        Returns:
            DocumentUploadResponse with document info (status: PENDING)
        """
        logger.info(f"Uploading document: {filename} ({len(file_content)} bytes)")
        
        # Detect file type
        file_type = self._detect_file_type(filename, content_type)
        
        # Validate
        self._validate_file(filename, len(file_content), file_type)
        
        # Determine MIME type
        mime_type = content_type or mimetypes.guess_type(filename)[0]
        
        # Upload to S3 first
        storage_key = None
        storage_bucket = None
        try:
            storage_result = await self.storage.upload(
                file_content=file_content,
                filename=filename,
                user_id=user_id,
                content_type=mime_type,
                metadata={
                    "original_filename": filename,
                    "file_type": file_type.value,
                },
            )
            storage_key = storage_result["key"]
            storage_bucket = storage_result["bucket"]
            logger.info(f"File uploaded to S3: {storage_key}")
        except StorageError as e:
            logger.error(f"S3 upload failed: {e}")
            raise ValueError(f"Failed to store file: {str(e)}") from e
        
        # Create document record (status: PENDING)
        document = await self.repository.create(
            filename=filename,
            original_filename=filename,
            file_type=file_type,
            file_size=len(file_content),
            mime_type=mime_type,
            title=title or filename.rsplit(".", 1)[0],
            description=description,
            user_id=user_id,
            status=DocumentStatus.PENDING,
            collection_name=self.collection,
            storage_key=storage_key,
            storage_bucket=storage_bucket,
        )
        
        await self.session.commit()
        
        logger.info(f"Document record created: {document.id} (PENDING)")
        
        return DocumentUploadResponse(
            id=document.id,
            filename=document.filename,
            file_type=SchemaDocumentType(document.file_type.value),
            file_size=document.file_size,
            status=SchemaDocumentStatus(document.status.value),
            message="Document uploaded, processing queued",
        )
    
    async def process_document(self, document: Document, file_content: bytes) -> None:
        """Process document: extract text, chunk, embed, store."""
        try:
            logger.info(f"Processing document: {document.id} ({document.file_type})")
            
            # Update status to processing
            await self.repository.update_status(document.id, DocumentStatus.PROCESSING)
            await self.session.commit() # Commit needed since this runs in background
            
            # 1) Extract text based on file type
            extracted = self._extract_text(document, file_content)
            text = extracted["text"]
            doc_metadata = extracted.get("metadata", {})
            
            if not text.strip():
                raise ValueError("No text could be extracted from document")
            
            logger.info(f"Extracted {len(text)} chars from document")
            
            # Update document metadata from extraction
            if doc_metadata:
                current_meta = document.metadata_ or {}
                current_meta.update(doc_metadata)
                await self.repository.update(document.id, metadata_=current_meta)
            
            # 2) Create chunk records
            chunk_records = self._create_chunks(document, extracted)
            
            if not chunk_records:
                raise ValueError("No chunks generated from document")
            
            logger.info(f"Created {len(chunk_records)} chunks")
            
            # 3) Embed chunks with contextual enrichment and proper prefix
            # Enrich text with metadata for better semantic matching
            texts = []
            for c in chunk_records:
                title = document.title or document.filename
                # Include context in the text for better embedding quality
                enriched_text = f"Document: {title}. Content: {c['text']}"
                texts.append(enriched_text)
            
            # Use embed_documents() which adds the search_document: prefix for nomic-embed-text
            vectors = await self.embedder.embed_documents(texts)
            vector_size = len(vectors[0])
            
            # 4) Ensure collection exists
            self.store.ensure_collection(self.collection, vector_size)
            
            # 5) Build and upsert points
            points: List[qm.PointStruct] = []
            total_tokens = 0
            
            for i, (chunk, vector) in enumerate(zip(chunk_records, vectors)):
                metadata = chunk.get("metadata", {})
                token_count = metadata.get("token_count", 0)
                total_tokens += token_count
                
                payload = {
                    "source": "document",
                    "doc_type": "document",  # Document type for filtering
                    "document_id": document.id,
                    "filename": document.filename,
                    "file_type": document.file_type.value,
                    "title": document.title,
                    "user_id": document.user_id,
                    "chunk_index": i,
                    "token_count": token_count,
                    "text": chunk["text"],
                    "speakers": [],  # Documents don't have speakers
                    **{k: v for k, v in metadata.items() if k not in ["token_count"]},
                }
                
                # Generate stable point ID
                point_id = _stable_point_id("document", document.id, str(i))
                points.append(qm.PointStruct(id=point_id, vector=vector, payload=payload))
            
            # 6) Upsert to Qdrant
            self.store.upsert(self.collection, points)
            
            logger.info(f"Upserted {len(points)} vectors to Qdrant")
            
            # 7) Update document status
            await self.repository.update_status(
                document.id,
                DocumentStatus.COMPLETED,
                chunk_count=len(points),
                total_tokens=total_tokens,
            )
            await self.session.commit()
            
            logger.info(f"Document {document.id} processed successfully")
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            await self.repository.update_status(
                document.id,
                DocumentStatus.FAILED,
                error_message=str(e),
            )
            await self.session.commit()
    
    def _extract_text(self, document: Document, file_content: bytes) -> Dict[str, Any]:
        """Extract text from document based on type."""
        if document.file_type == DocumentType.PDF:
            return self.pdf_loader.load(file_content, document.filename)
        elif document.file_type == DocumentType.DOCX:
            return self.docx_loader.load(file_content, document.filename)
        elif document.file_type == DocumentType.TXT:
            return self.txt_loader.load(file_content, document.filename)
        else:
            raise ValueError(f"Unsupported file type: {document.file_type}")
    
    def _create_chunks(self, document: Document, extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from extracted text."""
        text = extracted["text"]
        doc_metadata = extracted.get("metadata", {})
        
        # Build base metadata for chunks
        base_metadata = {
            "document_id": document.id,
            "filename": document.filename,
            "source_file": f"document:{document.id}",
            **doc_metadata,
        }
        
        # Create a record for chunking
        record = {
            "text": text,
            "metadata": base_metadata,
        }
        
        # Use the chunker
        chunks = self.chunker.chunk(record)
        
        return chunks
    
    async def get_document(self, document_id: str, user_id: Optional[str] = None) -> Optional[DocumentResponse]:
        """Get a single document by ID."""
        document = await self.repository.get_by_id(document_id)
        
        if not document:
            return None
        
        # Check user access if user_id provided
        if user_id and document.user_id and document.user_id != user_id:
            return None
        
        return self._to_response(document)
    
    async def list_documents(
        self,
        user_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        file_type: Optional[DocumentType] = None,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> DocumentListResponse:
        """List documents with filtering and pagination."""
        if user_id:
            documents = await self.repository.get_by_user(
                user_id=user_id,
                skip=skip,
                limit=limit,
                status=status,
                file_type=file_type,
                search=search,
            )
            total = await self.repository.count_by_user(
                user_id=user_id,
                status=status,
                file_type=file_type,
                search=search,
            )
        else:
            documents = await self.repository.get_all_documents(
                skip=skip,
                limit=limit,
                status=status,
                file_type=file_type,
                search=search,
            )
            total = await self.repository.count_all(
                status=status,
                file_type=file_type,
                search=search,
            )
        
        return DocumentListResponse(
            documents=[self._to_response(doc) for doc in documents],
            total=total,
            skip=skip,
            limit=limit,
            has_more=(skip + limit) < total,
        )
    
    async def delete_document(
        self,
        document_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[DocumentDeleteResponse]:
        """Delete a document and its chunks from vector store and S3."""
        document = await self.repository.get_by_id(document_id)
        
        if not document:
            return None
        
        # Check user access
        if user_id and document.user_id and document.user_id != user_id:
            return None
        
        chunks_removed = 0
        
        # Remove from vector store
        try:
            # Delete points by document_id filter
            self.store.client.delete(
                collection_name=self.collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="document_id",
                                match=qm.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            chunks_removed = document.chunk_count
            logger.info(f"Removed {chunks_removed} chunks from vector store")
        except Exception as e:
            logger.warning(f"Failed to remove chunks from vector store: {e}")
        
        # Delete from S3
        if document.storage_key:
            try:
                await self.storage.delete(document.storage_key)
                logger.info(f"Deleted file from S3: {document.storage_key}")
            except StorageError as e:
                logger.warning(f"Failed to delete file from S3: {e}")
        
        # Soft delete from database
        await self.repository.soft_delete(document_id)
        await self.session.commit()
        
        return DocumentDeleteResponse(
            id=document_id,
            deleted=True,
            chunks_removed=chunks_removed,
            message="Document and associated chunks deleted successfully",
        )
    
    async def get_document_stats(self, user_id: Optional[str] = None) -> DocumentStatsResponse:
        """Get document statistics."""
        if user_id:
            stats = await self.repository.get_user_stats(user_id)
        else:
            # System-wide stats would need a different implementation
            stats = await self.repository.get_user_stats(user_id) if user_id else {
                "total_documents": await self.repository.count_all(),
                "documents_by_status": {},
                "documents_by_type": {},
                "total_chunks": 0,
                "total_tokens": 0,
                "total_size_mb": 0,
            }
        
        return DocumentStatsResponse(**stats)
    
    async def get_document_chunks(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> Optional[DocumentChunksResponse]:
        """Get chunks for a specific document."""
        document = await self.repository.get_by_id(document_id)
        
        if not document:
            return None
        
        # Check user access
        if user_id and document.user_id and document.user_id != user_id:
            return None
        
        # Query vector store for document chunks
        try:
            results = self.store.client.scroll(
                collection_name=self.collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            points = results[0]
            
            chunks = []
            for point in points:
                payload = point.payload or {}
                chunks.append(DocumentChunkResponse(
                    chunk_id=str(point.id),
                    document_id=document_id,
                    text=payload.get("text", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    token_count=payload.get("token_count", 0),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["text", "chunk_index", "token_count", "document_id"]
                    },
                ))
            
            # Sort by chunk_index
            chunks.sort(key=lambda c: c.chunk_index)
            
            return DocumentChunksResponse(
                document_id=document_id,
                filename=document.filename,
                chunks=chunks,
                total_chunks=len(chunks),
            )
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return DocumentChunksResponse(
                document_id=document_id,
                filename=document.filename,
                chunks=[],
                total_chunks=0,
            )
    
    async def get_download_url(
        self,
        document_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a presigned download URL for a document.
        
        Args:
            document_id: Document ID
            user_id: Optional user ID for access check
            
        Returns:
            Presigned URL or None if not found/unauthorized
        """
        document = await self.repository.get_by_id(document_id)
        
        if not document or not document.storage_key:
            return None
        
        # Check user access
        if user_id and document.user_id and document.user_id != user_id:
            return None
        
        try:
            url = await self.storage.get_presigned_url(
                key=document.storage_key,
                filename=document.filename,
            )
            return url
        except StorageError as e:
            logger.error(f"Failed to generate download URL: {e}")
            return None
    
    async def download_file(
        self,
        document_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Download the original file content from S3.
        
        Args:
            document_id: Document ID
            user_id: Optional user ID for access check
            
        Returns:
            File bytes or None if not found/unauthorized
        """
        document = await self.repository.get_by_id(document_id)
        
        if not document or not document.storage_key:
            return None
        
        # Check user access
        if user_id and document.user_id and document.user_id != user_id:
            return None
        
        try:
            content = await self.storage.download(document.storage_key)
            return content
        except StorageError as e:
            logger.error(f"Failed to download file: {e}")
            return None
    
    def _to_response(self, document: Document, download_url: Optional[str] = None) -> DocumentResponse:
        """Convert Document model to response schema."""
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            file_type=SchemaDocumentType(document.file_type.value),
            file_size=document.file_size,
            file_size_mb=document.file_size_mb,
            mime_type=document.mime_type,
            title=document.title,
            description=document.description,
            status=SchemaDocumentStatus(document.status.value),
            error_message=document.error_message,
            chunk_count=document.chunk_count,
            total_tokens=document.total_tokens,
            metadata=document.metadata_ or {},
            storage_key=document.storage_key,
            download_url=download_url,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
            user_id=document.user_id,
        )

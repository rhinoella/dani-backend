"""
Document upload and management routes.

Provides endpoints for:
- File upload (PDF, DOCX, TXT)
- Document listing and retrieval
- Document deletion
- Chunk retrieval
"""

from __future__ import annotations

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user, get_optional_user
from app.database.models import User
from app.database.models.document import DocumentType, DocumentStatus
from app.services.document_service import DocumentService
from app.database.connection import get_async_session, get_db_context
# ... existing imports ...
from app.schemas.document import (
    DocumentResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentStatsResponse,
    DocumentChunksResponse,
    DocumentUpdateRequest,
    DocumentType as SchemaDocumentType,
    DocumentStatus as SchemaDocumentStatus,
    DocumentDownloadUrlResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)


# Maximum upload size: 50MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Allowed content types
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/markdown",
}


def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    """Dependency to get DocumentService instance."""
    return DocumentService(db)


async def process_document_background(document_id: str, file_content: bytes):
    """
    Background task to process a document (extract, chunk, embed).
    Uses get_db_context() for proper session management in background.
    """
    logger.info(f"Starting background processing for document {document_id}")
    try:
        async with get_db_context() as session:
            service = DocumentService(session)
            document = await service.repository.get_by_id(document_id)
            if document:
                await service.process_document(document, file_content)
                logger.info(f"Background processing completed for document {document_id}")
            else:
                logger.error(f"Document {document_id} not found during background processing")
    except Exception as e:
        logger.error(f"Background processing error for {document_id}: {e}", exc_info=True)
        # Ensure document status is updated to FAILED even if the service didn't handle it
        try:
            async with get_db_context() as session:
                from app.repositories.document_repository import DocumentRepository
                repo = DocumentRepository(session)
                await repo.update_status(
                    document_id,
                    DocumentStatus.FAILED,
                    error_message=f"Background processing failed: {str(e)}"
                )
                await session.commit()
                logger.info(f"Updated document {document_id} status to FAILED")
        except Exception as update_error:
            logger.error(f"Failed to update document status to FAILED: {update_error}")




@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="File(s) to upload (PDF, DOCX, or TXT). Can be single or multiple files."),
    titles: Optional[List[str]] = Form(None, description="Optional titles for each document (must match number of files)"),
    descriptions: Optional[List[str]] = Form(None, description="Optional descriptions for each document (must match number of files)"),
    service: DocumentService = Depends(get_document_service),
    current_user: User = Depends(get_current_user),
):
    """
    Upload one or more documents for processing.

    **Single file upload**: Select 1 file
    **Multiple files upload**: Select 2-10 files

    Supported formats:
    - **PDF**: `.pdf` files up to 50MB
    - **DOCX**: `.docx` files up to 25MB
    - **TXT**: `.txt`, `.md` files up to 10MB

    Each document will be:
    1. Validated and stored
    2. Text extracted (Background)
    3. Chunked into segments (Background)
    4. Embedded and stored in vector database (Background)

    Returns:
    - **Single file**: Single document object
    - **Multiple files**: Array of document objects

    Max 10 files per request. Partial success allowed for multiple files.
    """
    # Validate number of files
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files per upload",
        )

    # Validate titles and descriptions match file count if provided
    # If they don't match, ignore them and use defaults
    if titles and len(titles) != len(files):
        logger.warning(f"Titles count ({len(titles)}) doesn't match files count ({len(files)}). Ignoring titles.")
        titles = None

    if descriptions and len(descriptions) != len(files):
        logger.warning(f"Descriptions count ({len(descriptions)}) doesn't match files count ({len(files)}). Ignoring descriptions.")
        descriptions = None

    is_single_file = len(files) == 1
    logger.info(f"Upload request: {len(files)} file(s)")

    results = []
    errors = []

    for idx, file in enumerate(files):
        # Get title and description for this file
        title = titles[idx] if titles else None
        description = descriptions[idx] if descriptions else None

        try:
            # Validate content type
            if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
                ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
                if ext not in [".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".text"]:
                    error_msg = f"Unsupported file type: {file.content_type}. Allowed: PDF, DOCX, TXT"
                    if is_single_file:
                        raise HTTPException(
                            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=error_msg,
                        )
                    errors.append({"filename": file.filename, "error": error_msg})
                    continue

            # Read file content
            try:
                file_content = await file.read()
            except Exception as e:
                logger.error(f"Failed to read file {file.filename}: {e}")
                error_msg = "Failed to read file"
                if is_single_file:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=error_msg,
                    )
                errors.append({"filename": file.filename, "error": error_msg})
                continue

            # Check file size
            if len(file_content) > MAX_UPLOAD_SIZE:
                error_msg = f"File too large: {len(file_content) / (1024*1024):.1f}MB (max: 50MB)"
                if is_single_file:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=error_msg,
                    )
                errors.append({"filename": file.filename, "error": error_msg})
                continue

            # Check for empty file
            if len(file_content) == 0:
                error_msg = "File is empty"
                if is_single_file:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=error_msg,
                    )
                errors.append({"filename": file.filename, "error": error_msg})
                continue

            # Process upload
            result = await service.upload_document(
                filename=file.filename,
                file_content=file_content,
                content_type=file.content_type,
                title=title,
                description=description,
                user_id=current_user.id,
            )

            # Schedule processing in background
            background_tasks.add_task(
                process_document_background,
                result.id,
                file_content
            )

            results.append(result)
            logger.info(f"Upload [{idx+1}/{len(files)}]: {file.filename} -> {result.id}")

        except HTTPException:
            raise  # Re-raise HTTP exceptions for single file uploads
        except ValueError as e:
            logger.warning(f"Document upload validation failed for {file.filename}: {e}")
            error_msg = str(e)
            if is_single_file:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )
            errors.append({"filename": file.filename, "error": error_msg})
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}", exc_info=True)
            error_msg = "Failed to process document"
            if is_single_file:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=error_msg,
                )
            errors.append({"filename": file.filename, "error": str(e)})

    # Log errors for multiple file uploads
    if errors and not is_single_file:
        logger.warning(f"Upload completed with {len(errors)} error(s): {errors}")

    # Check if all uploads failed
    if not results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"All uploads failed. Errors: {errors}",
        )

    # Return single object for single file, array for multiple files
    if is_single_file:
        return results[0]
    else:
        return results


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[SchemaDocumentStatus] = Query(None, description="Filter by status"),
    file_type: Optional[SchemaDocumentType] = Query(None, description="Filter by file type"),
    search: Optional[str] = Query(None, max_length=200, description="Search in filename/title"),
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum documents to return"),
    service: DocumentService = Depends(get_document_service),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    List all documents with filtering and pagination.
    
    If authenticated, returns only the user's documents.
    Otherwise, returns all documents (public view).
    """
    # Convert schema enums to model enums
    model_status = DocumentStatus(status.value) if status else None
    model_type = DocumentType(file_type.value) if file_type else None
    
    return await service.list_documents(
        user_id=current_user.id if current_user else None,
        status=model_status,
        file_type=model_type,
        search=search,
        skip=skip,
        limit=limit,
    )


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats(
    service: DocumentService = Depends(get_document_service),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get document statistics.
    
    Returns counts by status and type, total chunks, tokens, and storage used.
    """
    return await service.get_document_stats(
        user_id=current_user.id if current_user else None,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get a single document by ID.
    
    Returns document metadata including processing status and chunk count.
    """
    document = await service.get_document(
        document_id=document_id,
        user_id=current_user.id if current_user else None,
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    return document


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    document_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum chunks to return"),
    service: DocumentService = Depends(get_document_service),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get chunks for a specific document.
    
    Returns all text chunks stored in the vector database for this document.
    Useful for viewing how the document was split for search.
    """
    result = await service.get_document_chunks(
        document_id=document_id,
        user_id=current_user.id if current_user else None,
        limit=limit,
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    return result


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    update_data: DocumentUpdateRequest,
    service: DocumentService = Depends(get_document_service),
    current_user: User = Depends(get_current_user),
):
    """
    Update document metadata (title, description).
    
    Does not re-process the document.
    """
    # First check if document exists and user has access
    document = await service.get_document(
        document_id=document_id,
        user_id=current_user.id if current_user else None,
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Update the document
    try:
        updated = await service.repository.update(
            document_id,
            title=update_data.title,
            description=update_data.description,
        )
        await service.session.commit()
        
        if updated:
            return service._to_response(updated)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document",
        )
    except Exception as e:
        logger.error(f"Failed to update document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document",
        )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a document and its chunks.
    
    This will:
    1. Remove all chunks from the vector database
    2. Soft-delete the document record
    
    The document can be recovered by an admin if needed.
    """
    result = await service.delete_document(
        document_id=document_id,
        user_id=current_user.id if current_user else None,
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    return result


@router.post("/{document_id}/reprocess", response_model=DocumentUploadResponse)
async def reprocess_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    current_user: User = Depends(get_current_user),
):
    """
    Re-process an existing document.
    
    This is useful if processing failed or if chunking/embedding settings changed.
    Note: Requires the original file to still be accessible.
    """
    # This endpoint would require storing the original file, which we're not doing
    # For now, return a not-implemented error
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Re-processing not yet implemented. Please delete and re-upload the document.",
    )


@router.get("/{document_id}/download-url", response_model=DocumentDownloadUrlResponse)
async def get_download_url(
    document_id: str,
    expires_in: int = Query(3600, ge=60, le=86400, description="URL expiry in seconds (1 min to 24 hours)"),
    inline: bool = Query(False, description="If true, use inline Content-Disposition for in-browser viewing"),
    service: DocumentService = Depends(get_document_service),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get a presigned download URL for the original document file.
    
    The URL will be valid for the specified duration (default 1 hour, max 24 hours).
    Use inline=true for in-browser viewing (PDF preview), inline=false for download.
    """
    # Check document exists and user has access
    document = await service.get_document(
        document_id=document_id,
        user_id=current_user.id if current_user else None,
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Get download URL
    try:
        download_url = await service.get_download_url(
            document_id=document_id,
            user_id=current_user.id if current_user else None,
            expires_in=expires_in,
            inline=inline,
        )
        
        if not download_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found in storage",
            )
        
        return DocumentDownloadUrlResponse(
            id=document_id,
            filename=document.filename,
            download_url=download_url,
            expires_in_seconds=expires_in,
        )
    except Exception as e:
        logger.error(f"Failed to generate download URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL",
        )


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Download the original document file.
    
    Streams the file content directly to the client. Use the `/download-url` endpoint
    for getting a presigned URL instead, which is more efficient for large files.
    """
    # Check document exists and user has access
    document = await service.get_document(
        document_id=document_id,
        user_id=current_user.id if current_user else None,
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    # Download file content
    try:
        result = await service.download_file(
            document_id=document_id,
            user_id=current_user.id if current_user else None,
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found in storage",
            )
        
        content, content_type, filename = result
        
        return Response(
            content=content,
            media_type=content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content)),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download document",
        )

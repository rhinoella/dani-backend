"""
Unified exception handling for DANI Engine.

This module provides:
- Custom exception classes for different error types
- Standardized error response format
- Exception handlers for FastAPI
"""

from __future__ import annotations

from typing import Any, Optional, Dict
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Error Response Schema
# =============================================================================

class ErrorDetail(BaseModel):
    """Standardized error detail."""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    success: bool = False
    error: ErrorDetail
    request_id: Optional[str] = None


# =============================================================================
# Custom Exception Classes
# =============================================================================

class DANIException(Exception):
    """Base exception for DANI Engine."""
    
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        self.field = field
        super().__init__(self.message)
    
    def to_response(self) -> ErrorResponse:
        """Convert exception to standardized error response."""
        return ErrorResponse(
            error=ErrorDetail(
                code=self.code,
                message=self.message,
                field=self.field,
                details=self.details,
            )
        )


# --- Authentication Errors ---

class AuthenticationError(DANIException):
    """Authentication failed."""
    
    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="AUTHENTICATION_REQUIRED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class InvalidTokenError(DANIException):
    """Invalid or expired token."""
    
    def __init__(
        self,
        message: str = "Invalid or expired token",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="INVALID_TOKEN",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthorizationError(DANIException):
    """User not authorized for this action."""
    
    def __init__(
        self,
        message: str = "Not authorized to perform this action",
        resource: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="AUTHORIZATION_DENIED",
            status_code=status.HTTP_403_FORBIDDEN,
            details={"resource": resource} if resource else None,
        )


# --- Resource Errors ---

class NotFoundError(DANIException):
    """Resource not found."""
    
    def __init__(
        self,
        resource: str,
        resource_id: Optional[str] = None,
    ):
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} with ID '{resource_id}' not found"
        
        super().__init__(
            message=message,
            code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "id": resource_id},
        )


class ConflictError(DANIException):
    """Resource conflict (e.g., duplicate)."""
    
    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="RESOURCE_CONFLICT",
            status_code=status.HTTP_409_CONFLICT,
            details={"resource": resource} if resource else None,
        )


# --- Validation Errors ---

class ValidationError(DANIException):
    """Input validation failed."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=422,  # Unprocessable Content
            field=field,
            details=details,
        )


class QueryTooLongError(ValidationError):
    """Query exceeds maximum length."""
    
    def __init__(self, length: int, max_length: int):
        super().__init__(
            message=f"Query too long: {length} characters (maximum: {max_length})",
            field="query",
            details={"length": length, "max_length": max_length},
        )


class InvalidFormatError(ValidationError):
    """Invalid output format requested."""
    
    def __init__(self, format: str, valid_formats: list):
        super().__init__(
            message=f"Invalid output format: {format}",
            field="output_format",
            details={"format": format, "valid_formats": valid_formats},
        )


# --- Rate Limiting Errors ---

class RateLimitError(DANIException):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit_type: str = "requests",
        limit: int = 0,
        remaining: int = 0,
        reset_at: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={
                "limit_type": limit_type,
                "limit": limit,
                "remaining": remaining,
                "reset_at": reset_at,
            },
        )


# --- Service Errors ---

class ServiceError(DANIException):
    """External service error."""
    
    def __init__(
        self,
        service: str,
        message: str = "Service temporarily unavailable",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=f"{service}: {message}",
            code="SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service, **(details or {})},
        )


class VectorStoreError(ServiceError):
    """Qdrant vector store error."""
    
    def __init__(self, message: str = "Vector store temporarily unavailable"):
        super().__init__(service="qdrant", message=message)


class LLMError(ServiceError):
    """LLM service error."""
    
    def __init__(self, message: str = "LLM service temporarily unavailable"):
        super().__init__(service="ollama", message=message)


class DatabaseError(ServiceError):
    """Database service error."""
    
    def __init__(self, message: str = "Database temporarily unavailable"):
        super().__init__(service="database", message=message)


class CacheError(ServiceError):
    """Cache service error."""
    
    def __init__(self, message: str = "Cache temporarily unavailable"):
        super().__init__(service="redis", message=message)


class StorageError(ServiceError):
    """S3/Storage service error."""
    
    def __init__(
        self,
        message: str = "Storage service temporarily unavailable",
        operation: Optional[str] = None,
        key: Optional[str] = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if key:
            details["key"] = key
        super().__init__(service="s3", message=message, details=details or None)


# --- MCP (Model Context Protocol) Errors ---

class MCPError(DANIException):
    """Base exception for MCP client errors."""
    
    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        error_details = details or {}
        if server_name:
            error_details["server_name"] = server_name
        super().__init__(
            message=message,
            code="MCP_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=error_details or None,
        )


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""
    
    def __init__(
        self,
        message: str = "Failed to connect to MCP server",
        server_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            server_name=server_name,
        )
        self.code = "MCP_CONNECTION_ERROR"
        self.status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class MCPTimeoutError(MCPError):
    """MCP operation timed out."""
    
    def __init__(
        self,
        message: str = "MCP operation timed out",
        server_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        details = {"timeout": timeout} if timeout else None
        super().__init__(
            message=message,
            server_name=server_name,
            details=details,
        )
        self.code = "MCP_TIMEOUT"
        self.status_code = status.HTTP_504_GATEWAY_TIMEOUT


class MCPToolError(MCPError):
    """Error executing an MCP tool."""
    
    def __init__(
        self,
        message: str = "MCP tool execution failed",
        server_name: Optional[str] = None,
        tool_name: Optional[str] = None,
    ):
        details = {}
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(
            message=message,
            server_name=server_name,
            details=details or None,
        )
        self.code = "MCP_TOOL_ERROR"


class MCPServerNotFoundError(MCPError):
    """MCP server not registered or not found."""
    
    def __init__(self, server_name: str):
        super().__init__(
            message=f"MCP server '{server_name}' not found",
            server_name=server_name,
        )
        self.code = "MCP_SERVER_NOT_FOUND"
        self.status_code = status.HTTP_404_NOT_FOUND


class MCPToolNotFoundError(MCPError):
    """MCP tool not found on server."""
    
    def __init__(self, tool_name: str, server_name: Optional[str] = None):
        super().__init__(
            message=f"MCP tool '{tool_name}' not found",
            server_name=server_name,
            details={"tool_name": tool_name},
        )
        self.code = "MCP_TOOL_NOT_FOUND"
        self.status_code = status.HTTP_404_NOT_FOUND


# --- Document Errors ---

class DocumentError(DANIException):
    """Document processing error."""
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        details = {}
        if document_id:
            details["document_id"] = document_id
        if filename:
            details["filename"] = filename
        super().__init__(
            message=message,
            code="DOCUMENT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or None,
        )


class DocumentNotFoundError(NotFoundError):
    """Document not found."""
    
    def __init__(self, document_id: str):
        super().__init__(resource="Document", resource_id=document_id)


class UnsupportedFileTypeError(ValidationError):
    """Unsupported file type for document upload."""
    
    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            message=f"Unsupported file type: {file_type}",
            field="file",
            details={"file_type": file_type, "supported_types": supported_types},
        )


class FileTooLargeError(ValidationError):
    """File exceeds maximum size."""
    
    def __init__(self, size_mb: float, max_size_mb: float):
        super().__init__(
            message=f"File too large: {size_mb:.2f}MB (maximum: {max_size_mb}MB)",
            field="file",
            details={"size_mb": size_mb, "max_size_mb": max_size_mb},
        )


# --- Ingestion Errors ---

class IngestionError(DANIException):
    """Ingestion operation failed."""
    
    def __init__(
        self,
        message: str,
        transcript_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="INGESTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"transcript_id": transcript_id} if transcript_id else None,
        )


class TranscriptTooLargeError(ValidationError):
    """Transcript exceeds maximum size."""
    
    def __init__(self, size_mb: float, max_size_mb: float):
        super().__init__(
            message=f"Transcript too large: {size_mb:.2f}MB (maximum: {max_size_mb}MB)",
            field="transcript",
            details={"size_mb": size_mb, "max_size_mb": max_size_mb},
        )


# --- Webhook Errors ---

class WebhookError(DANIException):
    """Webhook processing error."""
    
    def __init__(
        self,
        message: str,
        event_type: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="WEBHOOK_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"event_type": event_type} if event_type else None,
        )


class WebhookSignatureError(DANIException):
    """Invalid webhook signature."""
    
    def __init__(self):
        super().__init__(
            message="Invalid webhook signature",
            code="INVALID_SIGNATURE",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


# --- Conversation Errors ---

class ConversationLimitError(DANIException):
    """Conversation limit exceeded."""
    
    def __init__(self, limit: int, limit_type: str = "conversations"):
        super().__init__(
            message=f"Maximum {limit_type} limit reached: {limit}",
            code="CONVERSATION_LIMIT",
            status_code=status.HTTP_403_FORBIDDEN,
            details={"limit": limit, "limit_type": limit_type},
        )


class MessageTooLongError(ValidationError):
    """Message exceeds maximum length."""
    
    def __init__(self, length: int, max_length: int):
        super().__init__(
            message=f"Message too long: {length} characters (maximum: {max_length})",
            field="content",
            details={"length": length, "max_length": max_length},
        )


# =============================================================================
# Exception Handlers for FastAPI
# =============================================================================

async def dani_exception_handler(request: Request, exc: DANIException) -> JSONResponse:
    """Handle DANIException and return standardized response."""
    logger.warning(
        f"DANIException: {exc.code} - {exc.message}",
        extra={"details": exc.details, "path": request.url.path},
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response().model_dump(),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException and convert to standardized response."""
    # Map common HTTP status codes to error codes
    code_map = {
        400: "BAD_REQUEST",
        401: "AUTHENTICATION_REQUIRED",
        403: "AUTHORIZATION_DENIED",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }
    
    error_code = code_map.get(exc.status_code, "UNKNOWN_ERROR")
    
    response = ErrorResponse(
        error=ErrorDetail(
            code=error_code,
            message=str(exc.detail),
        )
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        exc_info=True,
        extra={"path": request.url.path},
    )
    
    response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred. Please try again later.",
        )
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(),
    )


def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app."""
    app.add_exception_handler(DANIException, dani_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)


# =============================================================================
# Utility Functions
# =============================================================================

def raise_not_found(resource: str, resource_id: Optional[str] = None) -> None:
    """Convenience function to raise NotFoundError."""
    raise NotFoundError(resource=resource, resource_id=resource_id)


def raise_unauthorized(message: str = "Authentication required") -> None:
    """Convenience function to raise AuthenticationError."""
    raise AuthenticationError(message=message)


def raise_forbidden(message: str = "Not authorized", resource: Optional[str] = None) -> None:
    """Convenience function to raise AuthorizationError."""
    raise AuthorizationError(message=message, resource=resource)


def raise_validation_error(message: str, field: Optional[str] = None) -> None:
    """Convenience function to raise ValidationError."""
    raise ValidationError(message=message, field=field)

"""
Tests for unified exception handling.
"""

import pytest
from fastapi import HTTPException, status
from unittest.mock import MagicMock, AsyncMock

from app.core.exceptions import (
    # Base
    DANIException,
    ErrorResponse,
    ErrorDetail,
    # Auth errors
    AuthenticationError,
    InvalidTokenError,
    AuthorizationError,
    # Resource errors
    NotFoundError,
    ConflictError,
    # Validation errors
    ValidationError,
    QueryTooLongError,
    InvalidFormatError,
    # Rate limiting
    RateLimitError,
    # Service errors
    ServiceError,
    VectorStoreError,
    LLMError,
    DatabaseError,
    CacheError,
    # MCP errors
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    MCPToolError,
    MCPServerNotFoundError,
    MCPToolNotFoundError,
    # Ingestion errors
    IngestionError,
    TranscriptTooLargeError,
    # Webhook errors
    WebhookError,
    WebhookSignatureError,
    # Conversation errors
    ConversationLimitError,
    MessageTooLongError,
    # Handlers
    dani_exception_handler,
    http_exception_handler,
    generic_exception_handler,
    # Utilities
    raise_not_found,
    raise_unauthorized,
    raise_forbidden,
    raise_validation_error,
)


class TestErrorResponse:
    """Tests for ErrorResponse model."""
    
    def test_error_response_structure(self):
        """Test error response has correct structure."""
        response = ErrorResponse(
            error=ErrorDetail(
                code="TEST_ERROR",
                message="Test message",
                field="test_field",
                details={"key": "value"},
            ),
            request_id="req-123",
        )
        
        assert response.success is False
        assert response.error.code == "TEST_ERROR"
        assert response.error.message == "Test message"
        assert response.error.field == "test_field"
        assert response.error.details == {"key": "value"}
        assert response.request_id == "req-123"
    
    def test_error_response_minimal(self):
        """Test minimal error response."""
        response = ErrorResponse(
            error=ErrorDetail(code="ERROR", message="Something went wrong")
        )
        
        assert response.success is False
        assert response.error.field is None
        assert response.error.details is None
        assert response.request_id is None


class TestDANIException:
    """Tests for base DANIException."""
    
    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = DANIException(message="Test error")
        
        assert exc.message == "Test error"
        assert exc.code == "INTERNAL_ERROR"
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.details is None
        assert exc.field is None
    
    def test_full_exception(self):
        """Test exception with all fields."""
        exc = DANIException(
            message="Detailed error",
            code="CUSTOM_ERROR",
            status_code=400,
            details={"extra": "info"},
            field="name",
        )
        
        assert exc.message == "Detailed error"
        assert exc.code == "CUSTOM_ERROR"
        assert exc.status_code == 400
        assert exc.details == {"extra": "info"}
        assert exc.field == "name"
    
    def test_to_response(self):
        """Test converting exception to response."""
        exc = DANIException(
            message="Test",
            code="TEST",
            field="test_field",
            details={"key": "value"},
        )
        
        response = exc.to_response()
        
        assert isinstance(response, ErrorResponse)
        assert response.error.code == "TEST"
        assert response.error.message == "Test"
        assert response.error.field == "test_field"
        assert response.error.details == {"key": "value"}


class TestAuthenticationErrors:
    """Tests for authentication error classes."""
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError()
        
        assert exc.message == "Authentication required"
        assert exc.code == "AUTHENTICATION_REQUIRED"
        assert exc.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_authentication_error_custom_message(self):
        """Test AuthenticationError with custom message."""
        exc = AuthenticationError(message="Token expired")
        
        assert exc.message == "Token expired"
    
    def test_invalid_token_error(self):
        """Test InvalidTokenError."""
        exc = InvalidTokenError()
        
        assert exc.message == "Invalid or expired token"
        assert exc.code == "INVALID_TOKEN"
        assert exc.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_authorization_error(self):
        """Test AuthorizationError."""
        exc = AuthorizationError(resource="conversation")
        
        assert exc.message == "Not authorized to perform this action"
        assert exc.code == "AUTHORIZATION_DENIED"
        assert exc.status_code == status.HTTP_403_FORBIDDEN
        assert exc.details == {"resource": "conversation"}


class TestResourceErrors:
    """Tests for resource error classes."""
    
    def test_not_found_error(self):
        """Test NotFoundError."""
        exc = NotFoundError(resource="User")
        
        assert exc.message == "User not found"
        assert exc.code == "RESOURCE_NOT_FOUND"
        assert exc.status_code == status.HTTP_404_NOT_FOUND
    
    def test_not_found_error_with_id(self):
        """Test NotFoundError with resource ID."""
        exc = NotFoundError(resource="Conversation", resource_id="conv-123")
        
        assert exc.message == "Conversation with ID 'conv-123' not found"
        assert exc.details == {"resource": "Conversation", "id": "conv-123"}
    
    def test_conflict_error(self):
        """Test ConflictError."""
        exc = ConflictError(message="Email already exists", resource="User")
        
        assert exc.message == "Email already exists"
        assert exc.code == "RESOURCE_CONFLICT"
        assert exc.status_code == status.HTTP_409_CONFLICT
        assert exc.details == {"resource": "User"}


class TestValidationErrors:
    """Tests for validation error classes."""
    
    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError(message="Invalid input", field="email")
        
        assert exc.message == "Invalid input"
        assert exc.code == "VALIDATION_ERROR"
        assert exc.status_code == 422  # Unprocessable Content
        assert exc.field == "email"
    
    def test_query_too_long_error(self):
        """Test QueryTooLongError."""
        exc = QueryTooLongError(length=5000, max_length=2000)
        
        assert "5000" in exc.message
        assert "2000" in exc.message
        assert exc.field == "query"
        assert exc.details == {"length": 5000, "max_length": 2000}
    
    def test_invalid_format_error(self):
        """Test InvalidFormatError."""
        exc = InvalidFormatError(
            format="invalid",
            valid_formats=["summary", "tasks", "email"],
        )
        
        assert "invalid" in exc.message
        assert exc.field == "output_format"
        assert exc.details["valid_formats"] == ["summary", "tasks", "email"]


class TestRateLimitError:
    """Tests for RateLimitError."""
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(
            limit_type="minute",
            limit=20,
            remaining=0,
            reset_at="2024-12-27T12:00:00Z",
        )
        
        assert exc.code == "RATE_LIMIT_EXCEEDED"
        assert exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert exc.details["limit_type"] == "minute"
        assert exc.details["limit"] == 20
        assert exc.details["remaining"] == 0


class TestServiceErrors:
    """Tests for service error classes."""
    
    def test_service_error(self):
        """Test ServiceError."""
        exc = ServiceError(service="external-api", message="Connection timeout")
        
        assert "external-api" in exc.message
        assert "Connection timeout" in exc.message
        assert exc.code == "SERVICE_ERROR"
        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    
    def test_vector_store_error(self):
        """Test VectorStoreError."""
        exc = VectorStoreError()
        
        assert exc.details["service"] == "qdrant"
    
    def test_llm_error(self):
        """Test LLMError."""
        exc = LLMError(message="Model not loaded")
        
        assert exc.details["service"] == "ollama"
        assert "Model not loaded" in exc.message
    
    def test_database_error(self):
        """Test DatabaseError."""
        exc = DatabaseError()
        
        assert exc.details["service"] == "database"
    
    def test_cache_error(self):
        """Test CacheError."""
        exc = CacheError()
        
        assert exc.details["service"] == "redis"


class TestMCPErrors:
    """Tests for MCP (Model Context Protocol) error classes."""
    
    def test_mcp_error_base(self):
        """Test MCPError base class."""
        exc = MCPError(message="Something went wrong", server_name="nano-banana")
        
        assert exc.message == "Something went wrong"
        assert exc.code == "MCP_ERROR"
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert exc.details == {"server_name": "nano-banana"}
    
    def test_mcp_error_without_server(self):
        """Test MCPError without server name."""
        exc = MCPError(message="Generic MCP error")
        
        assert exc.message == "Generic MCP error"
        assert exc.details is None
    
    def test_mcp_connection_error(self):
        """Test MCPConnectionError."""
        exc = MCPConnectionError(server_name="nano-banana")
        
        assert exc.code == "MCP_CONNECTION_ERROR"
        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "nano-banana" in str(exc.details)
    
    def test_mcp_connection_error_custom_message(self):
        """Test MCPConnectionError with custom message."""
        exc = MCPConnectionError(
            message="Server process exited unexpectedly",
            server_name="test-server"
        )
        
        assert exc.message == "Server process exited unexpectedly"
    
    def test_mcp_timeout_error(self):
        """Test MCPTimeoutError."""
        exc = MCPTimeoutError(server_name="nano-banana", timeout=30.0)
        
        assert exc.code == "MCP_TIMEOUT"
        assert exc.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        assert exc.details["timeout"] == 30.0
    
    def test_mcp_tool_error(self):
        """Test MCPToolError."""
        exc = MCPToolError(
            message="Image generation failed",
            server_name="nano-banana",
            tool_name="generate_image"
        )
        
        assert exc.code == "MCP_TOOL_ERROR"
        assert exc.details["tool_name"] == "generate_image"
        assert exc.details["server_name"] == "nano-banana"
    
    def test_mcp_server_not_found_error(self):
        """Test MCPServerNotFoundError."""
        exc = MCPServerNotFoundError(server_name="nonexistent")
        
        assert exc.code == "MCP_SERVER_NOT_FOUND"
        assert exc.status_code == status.HTTP_404_NOT_FOUND
        assert "nonexistent" in exc.message
    
    def test_mcp_tool_not_found_error(self):
        """Test MCPToolNotFoundError."""
        exc = MCPToolNotFoundError(tool_name="unknown_tool", server_name="nano-banana")
        
        assert exc.code == "MCP_TOOL_NOT_FOUND"
        assert exc.status_code == status.HTTP_404_NOT_FOUND
        assert "unknown_tool" in exc.message
        assert exc.details["tool_name"] == "unknown_tool"


class TestIngestionErrors:
    """Tests for ingestion error classes."""
    
    def test_ingestion_error(self):
        """Test IngestionError."""
        exc = IngestionError(message="Failed to parse", transcript_id="t-123")
        
        assert exc.code == "INGESTION_ERROR"
        assert exc.details == {"transcript_id": "t-123"}
    
    def test_transcript_too_large_error(self):
        """Test TranscriptTooLargeError."""
        exc = TranscriptTooLargeError(size_mb=15.5, max_size_mb=10.0)
        
        assert "15.50MB" in exc.message
        assert "10.0MB" in exc.message


class TestWebhookErrors:
    """Tests for webhook error classes."""
    
    def test_webhook_error(self):
        """Test WebhookError."""
        exc = WebhookError(message="Invalid payload", event_type="transcription.completed")
        
        assert exc.code == "WEBHOOK_ERROR"
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert exc.details["event_type"] == "transcription.completed"
    
    def test_webhook_signature_error(self):
        """Test WebhookSignatureError."""
        exc = WebhookSignatureError()
        
        assert exc.code == "INVALID_SIGNATURE"
        assert exc.status_code == status.HTTP_401_UNAUTHORIZED


class TestConversationErrors:
    """Tests for conversation error classes."""
    
    def test_conversation_limit_error(self):
        """Test ConversationLimitError."""
        exc = ConversationLimitError(limit=100)
        
        assert exc.code == "CONVERSATION_LIMIT"
        assert exc.status_code == status.HTTP_403_FORBIDDEN
        assert exc.details["limit"] == 100
    
    def test_message_too_long_error(self):
        """Test MessageTooLongError."""
        exc = MessageTooLongError(length=5000, max_length=4000)
        
        assert exc.field == "content"
        assert exc.details["length"] == 5000
        assert exc.details["max_length"] == 4000


class TestExceptionHandlers:
    """Tests for exception handlers."""
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/v1/test"
        return request
    
    @pytest.mark.asyncio
    async def test_dani_exception_handler(self, mock_request):
        """Test DANIException handler."""
        exc = NotFoundError(resource="User", resource_id="123")
        
        response = await dani_exception_handler(mock_request, exc)
        
        assert response.status_code == 404
        body = response.body.decode()
        assert "RESOURCE_NOT_FOUND" in body
        assert "User" in body
    
    @pytest.mark.asyncio
    async def test_http_exception_handler(self, mock_request):
        """Test HTTPException handler."""
        exc = HTTPException(status_code=400, detail="Bad request")
        
        response = await http_exception_handler(mock_request, exc)
        
        assert response.status_code == 400
        body = response.body.decode()
        assert "BAD_REQUEST" in body
        assert "Bad request" in body
    
    @pytest.mark.asyncio
    async def test_generic_exception_handler(self, mock_request):
        """Test generic exception handler."""
        exc = RuntimeError("Something went wrong")
        
        response = await generic_exception_handler(mock_request, exc)
        
        assert response.status_code == 500
        body = response.body.decode()
        assert "INTERNAL_ERROR" in body


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_raise_not_found(self):
        """Test raise_not_found utility."""
        with pytest.raises(NotFoundError) as exc_info:
            raise_not_found("Conversation", "conv-123")
        
        assert exc_info.value.details["resource"] == "Conversation"
        assert exc_info.value.details["id"] == "conv-123"
    
    def test_raise_unauthorized(self):
        """Test raise_unauthorized utility."""
        with pytest.raises(AuthenticationError):
            raise_unauthorized("Token required")
    
    def test_raise_forbidden(self):
        """Test raise_forbidden utility."""
        with pytest.raises(AuthorizationError):
            raise_forbidden("Access denied", resource="admin")
    
    def test_raise_validation_error(self):
        """Test raise_validation_error utility."""
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error("Invalid email", field="email")
        
        assert exc_info.value.field == "email"

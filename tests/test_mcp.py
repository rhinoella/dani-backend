"""
Tests for MCP Client Infrastructure.

Tests the MCP client, registry, schemas, and tool wrappers.
"""

import asyncio
import base64
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)
from app.mcp.client import (
    MCPClient,
    extract_all_content,
    extract_image,
    extract_text,
)
from app.mcp.registry import MCPRegistry, get_registry, setup_default_servers
from app.mcp.schemas import (
    CallToolRequest,
    CallToolResponse,
    EditImageRequest,
    GenerateImageRequest,
    GenerateImageResponse,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPCapabilities,
    MCPContent,
    MCPImageContent,
    MCPInitializeResult,
    MCPRegistryStatus,
    MCPServerConfig,
    MCPServerStatus,
    MCPTextContent,
    MCPTool,
    MCPToolInputSchema,
    MCPToolResult,
    MCPToolsListResult,
    MCPTransport,
)
from app.mcp.tools.nano_banana import ImageResult, NanoBananaClient


# =============================================================================
# Schema Tests
# =============================================================================


class TestMCPSchemas:
    """Test MCP schema models."""

    def test_server_config_defaults(self):
        """Test MCPServerConfig with defaults."""
        config = MCPServerConfig(
            name="test-server",
            command="npx",
            args=["test-mcp"],
        )
        assert config.name == "test-server"
        assert config.command == "npx"
        assert config.args == ["test-mcp"]
        assert config.env == {}
        assert config.transport == MCPTransport.STDIO
        assert config.enabled is True

    def test_server_config_full(self):
        """Test MCPServerConfig with all fields."""
        config = MCPServerConfig(
            name="nano-banana",
            command="npx",
            args=["nano-banana-mcp"],
            env={"GEMINI_API_KEY": "test-key"},
            transport=MCPTransport.STDIO,
            description="Image generation server",
            enabled=True,
        )
        assert config.env["GEMINI_API_KEY"] == "test-key"
        assert config.description == "Image generation server"

    def test_jsonrpc_request(self):
        """Test JSON-RPC request serialization."""
        request = JSONRPCRequest(
            id=1,
            method="tools/list",
            params={"foo": "bar"},
        )
        assert request.jsonrpc == "2.0"
        assert request.id == 1
        assert request.method == "tools/list"

        # Test serialization
        data = request.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "tools/list"

    def test_jsonrpc_response_success(self):
        """Test successful JSON-RPC response."""
        response = JSONRPCResponse(
            id=1,
            result={"tools": []},
        )
        assert response.id == 1
        assert response.result == {"tools": []}
        assert response.error is None

    def test_jsonrpc_response_error(self):
        """Test error JSON-RPC response."""
        response = JSONRPCResponse(
            id=1,
            error={"code": -32600, "message": "Invalid request"},
        )
        assert response.error is not None
        assert response.error["code"] == -32600

    def test_mcp_tool(self):
        """Test MCPTool model."""
        tool = MCPTool(
            name="generate_image",
            description="Generate an image from text",
            inputSchema=MCPToolInputSchema(
                type="object",
                properties={"prompt": {"type": "string"}},
                required=["prompt"],
            ),
        )
        assert tool.name == "generate_image"
        assert "prompt" in tool.input_schema.properties

    def test_mcp_tool_result_text(self):
        """Test MCPToolResult with text content."""
        result = MCPToolResult(
            content=[MCPTextContent(text="Generated successfully")],
            isError=False,
        )
        assert not result.is_error
        assert len(result.content) == 1

    def test_mcp_tool_result_image(self):
        """Test MCPToolResult with image content."""
        result = MCPToolResult(
            content=[
                MCPImageContent(
                    data=base64.b64encode(b"fake-image").decode(),
                    mimeType="image/png",
                )
            ],
        )
        assert len(result.content) == 1

    def test_mcp_tool_result_error(self):
        """Test MCPToolResult with error."""
        result = MCPToolResult(
            content=[MCPTextContent(text="API key invalid")],
            isError=True,
        )
        assert result.is_error

    def test_mcp_server_status(self):
        """Test MCPServerStatus model."""
        status = MCPServerStatus(
            name="nano-banana",
            connected=True,
            tools=["generate_image", "edit_image"],
            connected_at=datetime.now(),
        )
        assert status.connected
        assert len(status.tools) == 2

    def test_generate_image_request(self):
        """Test GenerateImageRequest validation."""
        request = GenerateImageRequest(
            prompt="A sunset over mountains",
            server_name="nano-banana",
        )
        assert request.prompt == "A sunset over mountains"
        assert request.server_name == "nano-banana"

    def test_call_tool_request(self):
        """Test CallToolRequest validation."""
        request = CallToolRequest(
            server_name="nano-banana",
            tool_name="generate_image",
            arguments={"prompt": "test"},
        )
        assert request.server_name == "nano-banana"
        assert request.tool_name == "generate_image"


# =============================================================================
# Client Tests
# =============================================================================


class TestMCPClient:
    """Test MCPClient functionality."""

    @pytest.fixture
    def config(self):
        """Create test server config."""
        return MCPServerConfig(
            name="test-server",
            command="echo",
            args=["test"],
            env={"TEST_KEY": "test-value"},
        )

    def test_client_initialization(self, config):
        """Test client initialization."""
        client = MCPClient(config)
        assert not client.connected
        assert client.tools == []
        assert client.tool_names == []

    @pytest.mark.asyncio
    async def test_client_not_connected_error(self, config):
        """Test error when calling tool without connection."""
        client = MCPClient(config)

        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_extract_text_from_result(self):
        """Test extracting text from tool result."""
        result = MCPToolResult(
            content=[MCPTextContent(text="Hello world")],
        )
        text = extract_text(result)
        assert text == "Hello world"

    @pytest.mark.asyncio
    async def test_extract_text_from_dict_result(self):
        """Test extracting text from dict content."""
        result = MCPToolResult(
            content=[{"type": "text", "text": "Hello dict"}],  # type: ignore
        )
        text = extract_text(result)
        assert text == "Hello dict"

    @pytest.mark.asyncio
    async def test_extract_image_from_result(self):
        """Test extracting image from tool result."""
        image_data = base64.b64encode(b"fake-image-data").decode()
        result = MCPToolResult(
            content=[MCPImageContent(data=image_data, mimeType="image/png")],
        )
        extracted = extract_image(result)
        assert extracted is not None
        data, mime = extracted
        assert data == image_data
        assert mime == "image/png"

    @pytest.mark.asyncio
    async def test_extract_image_from_dict_result(self):
        """Test extracting image from dict content."""
        image_data = base64.b64encode(b"fake-image").decode()
        result = MCPToolResult(
            content=[{"type": "image", "data": image_data, "mimeType": "image/jpeg"}],  # type: ignore
        )
        extracted = extract_image(result)
        assert extracted is not None
        data, mime = extracted
        assert mime == "image/jpeg"

    @pytest.mark.asyncio
    async def test_extract_all_content(self):
        """Test extracting all content as dicts."""
        result = MCPToolResult(
            content=[
                MCPTextContent(text="Hello"),
                MCPImageContent(data="abc123", mimeType="image/png"),
            ],
        )
        contents = extract_all_content(result)
        assert len(contents) == 2
        assert contents[0]["type"] == "text"
        assert contents[1]["type"] == "image"


# =============================================================================
# Registry Tests
# =============================================================================


class TestMCPRegistry:
    """Test MCPRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        return MCPRegistry()

    @pytest.fixture
    def sample_config(self):
        """Create sample server config."""
        return MCPServerConfig(
            name="test-server",
            command="npx",
            args=["test-mcp"],
            description="Test MCP server",
        )

    def test_register_server(self, registry, sample_config):
        """Test registering a server."""
        registry.register(sample_config)
        assert "test-server" in registry.server_names
        assert len(registry.server_names) == 1

    def test_unregister_server(self, registry, sample_config):
        """Test unregistering a server."""
        registry.register(sample_config)
        result = registry.unregister("test-server")
        assert result is True
        assert "test-server" not in registry.server_names

    def test_unregister_nonexistent(self, registry):
        """Test unregistering a server that doesn't exist."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_config(self, registry, sample_config):
        """Test getting server configuration."""
        registry.register(sample_config)
        config = registry.get_config("test-server")
        assert config is not None
        assert config.name == "test-server"

    def test_get_config_not_found(self, registry):
        """Test getting config for non-registered server."""
        config = registry.get_config("nonexistent")
        assert config is None

    def test_connected_servers_empty(self, registry, sample_config):
        """Test connected servers when none connected."""
        registry.register(sample_config)
        assert registry.connected_servers == []

    def test_list_all_tools_empty(self, registry):
        """Test listing tools when no servers connected."""
        tools = registry.list_all_tools()
        assert tools == {}

    def test_find_tool_not_found(self, registry):
        """Test finding tool that doesn't exist."""
        result = registry.find_tool("nonexistent_tool")
        assert result is None

    def test_get_status_no_servers(self, registry):
        """Test status with no servers."""
        status = registry.get_status()
        assert status.servers == []
        assert status.total_tools == 0

    def test_get_status_with_registered_server(self, registry, sample_config):
        """Test status with registered but not connected server."""
        registry.register(sample_config)
        status = registry.get_status()
        assert len(status.servers) == 1
        assert status.servers[0].name == "test-server"
        assert not status.servers[0].connected

    @pytest.mark.asyncio
    async def test_connect_unregistered_server(self, registry):
        """Test connecting to unregistered server."""
        with pytest.raises(ValueError, match="not registered"):
            await registry.connect("nonexistent")

    @pytest.mark.asyncio
    async def test_connect_disabled_server(self, registry):
        """Test connecting to disabled server."""
        config = MCPServerConfig(
            name="disabled-server",
            command="npx",
            args=["test"],
            enabled=False,
        )
        registry.register(config)

        with pytest.raises(ValueError, match="disabled"):
            await registry.connect("disabled-server")

    @pytest.mark.asyncio
    async def test_disconnect_no_op(self, registry):
        """Test disconnecting from non-connected server."""
        await registry.disconnect("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, registry, sample_config):
        """Test calling tool on disconnected server with auto_connect=False."""
        registry.register(sample_config)

        with pytest.raises(MCPConnectionError, match="Not connected"):
            await registry.call_tool(
                "test-server",
                "test_tool",
                {},
                auto_connect=False,
            )


# =============================================================================
# Nano Banana Client Tests
# =============================================================================


class TestNanoBananaClient:
    """Test NanoBananaClient wrapper."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = NanoBananaClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert not client.connected

    def test_init_without_api_key_uses_settings(self, monkeypatch):
        """Test initialization from settings."""
        # NanoBananaClient now uses credential_manager which reads from settings
        # We need to patch the credential_manager's get_gemini_api_key method
        from app.mcp.security import credential_manager
        monkeypatch.setattr(credential_manager, "get_gemini_api_key", lambda: "settings-key")
        client = NanoBananaClient()
        assert client.api_key == "settings-key"

    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        """Test connection fails without API key."""
        client = NanoBananaClient(api_key=None)
        client.api_key = None  # Ensure it's None

        with pytest.raises(ValueError, match="API key required"):
            await client.connect()

    def test_image_result_has_image_base64(self):
        """Test ImageResult.has_image with base64 data."""
        result = ImageResult(
            success=True,
            image_base64="abc123",
        )
        assert result.has_image

    def test_image_result_has_image_bytes(self):
        """Test ImageResult.has_image with byte data."""
        result = ImageResult(
            success=True,
            image_data=b"fake-image",
        )
        assert result.has_image

    def test_image_result_no_image(self):
        """Test ImageResult.has_image when no image."""
        result = ImageResult(success=False, error="Failed")
        assert not result.has_image

    def test_image_result_save_no_data(self, tmp_path):
        """Test ImageResult.save fails without data."""
        result = ImageResult(success=False)

        with pytest.raises(ValueError, match="No image data"):
            result.save(tmp_path / "test.png")

    def test_image_result_save_from_base64(self, tmp_path):
        """Test ImageResult.save from base64."""
        image_data = b"fake-image-content"
        result = ImageResult(
            success=True,
            image_base64=base64.b64encode(image_data).decode(),
        )

        path = result.save(tmp_path / "test.png")
        assert (tmp_path / "test.png").exists()
        assert (tmp_path / "test.png").read_bytes() == image_data

    def test_image_result_save_from_bytes(self, tmp_path):
        """Test ImageResult.save from bytes."""
        image_data = b"fake-image-bytes"
        result = ImageResult(
            success=True,
            image_data=image_data,
        )

        path = result.save(tmp_path / "output.png")
        assert (tmp_path / "output.png").read_bytes() == image_data


# =============================================================================
# API Route Tests
# =============================================================================


class TestMCPAPIRoutes:
    """Test MCP API route handlers."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        with patch("app.api.routes.mcp.get_registry") as mock:
            registry = MagicMock(spec=MCPRegistry)
            registry.server_names = ["nano-banana"]
            registry.connected_servers = []
            mock.return_value = registry
            yield registry

    @pytest.mark.asyncio
    async def test_call_tool_request_validation(self):
        """Test CallToolRequest validation."""
        request = CallToolRequest(
            server_name="nano-banana",
            tool_name="generate_image",
            arguments={"prompt": "A test image"},
        )
        assert request.server_name == "nano-banana"
        assert request.tool_name == "generate_image"
        assert request.arguments["prompt"] == "A test image"

    @pytest.mark.asyncio
    async def test_call_tool_response_success(self):
        """Test CallToolResponse for success."""
        response = CallToolResponse(
            success=True,
            content=[{"type": "text", "text": "Success"}],
            execution_time_ms=150.0,
        )
        assert response.success
        assert response.error is None

    @pytest.mark.asyncio
    async def test_call_tool_response_error(self):
        """Test CallToolResponse for error."""
        response = CallToolResponse(
            success=False,
            error="API key invalid",
            execution_time_ms=50.0,
        )
        assert not response.success
        assert response.error == "API key invalid"

    @pytest.mark.asyncio
    async def test_generate_image_request(self):
        """Test GenerateImageRequest model."""
        request = GenerateImageRequest(
            prompt="A beautiful sunset over mountains",
        )
        assert request.prompt == "A beautiful sunset over mountains"
        assert request.server_name == "nano-banana"  # Default

    @pytest.mark.asyncio
    async def test_generate_image_response_success(self):
        """Test GenerateImageResponse for success."""
        response = GenerateImageResponse(
            success=True,
            image_data="base64-encoded-data",
        )
        assert response.success
        assert response.image_data is not None

    @pytest.mark.asyncio
    async def test_generate_image_response_with_url(self):
        """Test GenerateImageResponse with URL."""
        response = GenerateImageResponse(
            success=True,
            image_url="https://example.com/image.png",
        )
        assert response.success
        assert response.image_url is not None

    @pytest.mark.asyncio
    async def test_edit_image_request(self):
        """Test EditImageRequest model."""
        request = EditImageRequest(
            image_path="/path/to/image.png",
            prompt="Add a rainbow",
            reference_images=["/path/to/ref.png"],
        )
        assert request.image_path == "/path/to/image.png"
        assert request.prompt == "Add a rainbow"
        assert len(request.reference_images) == 1


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestMCPIntegration:
    """Integration tests with mocked subprocess."""

    @pytest.fixture
    def mock_process(self):
        """Create mock subprocess."""
        process = AsyncMock()
        process.stdin = AsyncMock()
        process.stdout = AsyncMock()
        process.stderr = AsyncMock()
        process.returncode = None
        return process

    @pytest.mark.asyncio
    async def test_client_connect_flow(self, mock_process):
        """Test full connection flow with mocked process."""
        config = MCPServerConfig(
            name="test",
            command="npx",
            args=["test-mcp"],
        )
        client = MCPClient(config, timeout=5.0)

        # Mock responses
        init_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "1.0"},
            },
        }) + "\n"

        tools_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }) + "\n"

        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                init_response.encode(),
                tools_response.encode(),
            ]
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await client.connect()

        assert client.connected
        assert "test_tool" in client.tool_names

        # Cleanup
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_registry_connect_all(self):
        """Test registry connect_all with mixed results."""
        registry = MCPRegistry()

        # Register multiple servers
        registry.register(MCPServerConfig(
            name="server1",
            command="npx",
            args=["server1-mcp"],
        ))
        registry.register(MCPServerConfig(
            name="disabled",
            command="npx",
            args=["disabled-mcp"],
            enabled=False,
        ))

        # Mock connection to fail for server1
        with patch.object(registry, "connect") as mock_connect:
            mock_connect.side_effect = MCPConnectionError("Connection failed")

            results = await registry.connect_all(ignore_errors=True)

        assert "server1" in results
        assert "disabled" in results
        assert results["disabled"] == "disabled"
        assert "failed" in results["server1"].lower()

    @pytest.mark.asyncio
    async def test_setup_default_servers(self):
        """Test setup_default_servers helper."""
        with patch("app.mcp.registry._registry", None):
            registry = await setup_default_servers(
                gemini_api_key="test-key",
                imgbb_api_key="imgbb-key",
            )

        assert "nano-banana" in registry.server_names
        assert "nano-banana-python" in registry.server_names

        # Python version should be disabled by default
        python_config = registry.get_config("nano-banana-python")
        assert python_config is not None
        assert not python_config.enabled


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Test helper/utility functions."""

    def test_extract_text_empty_result(self):
        """Test extract_text with empty content."""
        result = MCPToolResult(content=[])
        assert extract_text(result) is None

    def test_extract_text_no_text_content(self):
        """Test extract_text with only image content."""
        result = MCPToolResult(
            content=[MCPImageContent(data="abc", mimeType="image/png")]
        )
        assert extract_text(result) is None

    def test_extract_image_empty_result(self):
        """Test extract_image with empty content."""
        result = MCPToolResult(content=[])
        assert extract_image(result) is None

    def test_extract_image_no_image_content(self):
        """Test extract_image with only text content."""
        result = MCPToolResult(
            content=[MCPTextContent(text="Hello")]
        )
        assert extract_image(result) is None

    def test_extract_all_content_mixed(self):
        """Test extract_all_content with mixed content types."""
        result = MCPToolResult(
            content=[
                MCPTextContent(text="Message 1"),
                {"type": "text", "text": "Message 2"},  # type: ignore
                MCPImageContent(data="img1", mimeType="image/png"),
            ],
        )
        contents = extract_all_content(result)
        assert len(contents) == 3
        assert all(isinstance(c, dict) for c in contents)


# =============================================================================
# Security Tests
# =============================================================================


class TestMCPSecurity:
    """Test MCP security module."""

    def test_credential_manager_mask_credential(self):
        """Test credential masking."""
        from app.mcp.security import credential_manager

        assert credential_manager.mask_credential("my-secret-api-key-12345") == "***2345"
        assert credential_manager.mask_credential("abc") == "***"
        assert credential_manager.mask_credential("") == "<empty>"

    def test_credential_manager_fingerprint(self):
        """Test credential fingerprinting."""
        from app.mcp.security import credential_manager

        fp1 = credential_manager.get_credential_fingerprint("key1")
        fp2 = credential_manager.get_credential_fingerprint("key1")
        fp3 = credential_manager.get_credential_fingerprint("key2")

        assert fp1 == fp2  # Same key = same fingerprint
        assert fp1 != fp3  # Different key = different fingerprint
        assert len(fp1) == 8  # 8 char hash

    def test_input_validator_prompt_valid(self):
        """Test valid prompt validation."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_prompt("A beautiful sunset")
        assert is_valid
        assert error is None

    def test_input_validator_prompt_empty(self):
        """Test empty prompt validation."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_prompt("")
        assert not is_valid
        assert "required" in error.lower()

    def test_input_validator_prompt_too_long(self):
        """Test prompt length validation."""
        from app.mcp.security import input_validator

        long_prompt = "x" * 5000
        is_valid, error = input_validator.validate_prompt(long_prompt)
        assert not is_valid
        assert "length" in error.lower()

    def test_input_validator_prompt_blocked_content(self):
        """Test prompt with blocked content."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_prompt("<script>alert(1)</script>")
        assert not is_valid
        assert "blocked" in error.lower()

    def test_input_validator_tool_name_valid(self):
        """Test valid tool name validation."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_tool_name("generate_image")
        assert is_valid
        assert error is None

    def test_input_validator_tool_name_invalid_chars(self):
        """Test tool name with invalid characters."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_tool_name("gen;image")
        assert not is_valid
        assert "invalid" in error.lower()

    def test_input_validator_sanitize_arguments(self):
        """Test argument sanitization."""
        from app.mcp.security import input_validator

        args = {"prompt": "test", "invalid;key": "value"}
        sanitized, warnings = input_validator.sanitize_arguments(args)

        assert "prompt" in sanitized
        assert "invalid;key" not in sanitized
        assert len(warnings) > 0

    def test_command_validator_allowed(self):
        """Test allowed command validation."""
        from app.mcp.security import command_validator

        is_valid, error = command_validator.validate_command("npx")
        assert is_valid
        assert error is None

    def test_command_validator_blocked(self):
        """Test blocked command validation."""
        from app.mcp.security import command_validator

        is_valid, error = command_validator.validate_command("rm")
        assert not is_valid
        assert "not in allowed" in error.lower()

    def test_command_validator_shell_injection(self):
        """Test shell injection prevention."""
        from app.mcp.security import command_validator

        # The command with shell chars gets rejected either via allowlist or blocked char
        is_valid, error = command_validator.validate_command("npx; rm -rf /")
        assert not is_valid
        # May be rejected as "not in allowed list" or "blocked character" depending on parse order
        assert error is not None

    def test_file_path_validation_traversal(self):
        """Test path traversal prevention."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_file_path("../../etc/passwd")
        assert not is_valid
        assert "traversal" in error.lower()

    def test_file_path_validation_system_dirs(self):
        """Test system directory blocking."""
        from app.mcp.security import input_validator

        is_valid, error = input_validator.validate_file_path("/etc/passwd")
        assert not is_valid
        assert "not allowed" in error.lower()

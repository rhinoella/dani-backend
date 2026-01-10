"""
MCP Protocol Schemas.

Defines Pydantic models for the Model Context Protocol (MCP) messages,
following the JSON-RPC 2.0 specification used by MCP.

Reference: https://spec.modelcontextprotocol.io/
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Server Configuration
# =============================================================================


class MCPTransport(str, Enum):
    """Transport mechanism for MCP communication."""

    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection."""

    name: str = Field(..., description="Unique identifier for the server")
    command: str = Field(..., description="Command to start the server (e.g., 'npx', 'uvx')")
    args: list[str] = Field(default_factory=list, description="Arguments for the command")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    transport: MCPTransport = Field(default=MCPTransport.STDIO)
    description: str | None = Field(None, description="Human-readable description")
    enabled: bool = Field(default=True, description="Whether this server is enabled")

    model_config = {"use_enum_values": True}


# =============================================================================
# JSON-RPC 2.0 Base Messages
# =============================================================================


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response message."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


class JSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 notification (no id, no response expected)."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | None = None


# =============================================================================
# MCP Protocol Messages
# =============================================================================


class MCPCapabilities(BaseModel):
    """Server capabilities advertised during initialization."""

    tools: dict[str, Any] | None = None
    resources: dict[str, Any] | None = None
    prompts: dict[str, Any] | None = None
    logging: dict[str, Any] | None = None


class MCPServerInfo(BaseModel):
    """Server information from initialization."""

    name: str
    version: str
    protocol_version: str = Field(alias="protocolVersion", default="2024-11-05")


class MCPInitializeResult(BaseModel):
    """Result of the initialize request."""

    protocol_version: str = Field(alias="protocolVersion", default="2024-11-05")
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)
    server_info: MCPServerInfo | None = Field(alias="serverInfo", default=None)


# =============================================================================
# MCP Tools
# =============================================================================


class MCPToolInputSchema(BaseModel):
    """JSON Schema for tool input parameters."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class MCPTool(BaseModel):
    """Definition of an MCP tool."""

    name: str = Field(..., description="Unique name of the tool")
    description: str | None = Field(None, description="Human-readable description")
    input_schema: MCPToolInputSchema = Field(
        alias="inputSchema", default_factory=MCPToolInputSchema
    )

    model_config = {"populate_by_name": True}


class MCPToolsListResult(BaseModel):
    """Result of tools/list request."""

    tools: list[MCPTool] = Field(default_factory=list)


class MCPToolCall(BaseModel):
    """A call to an MCP tool."""

    server_name: str = Field(..., description="Name of the MCP server")
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPContentType(str, Enum):
    """Types of content in tool results."""

    TEXT = "text"
    IMAGE = "image"
    RESOURCE = "resource"


class MCPTextContent(BaseModel):
    """Text content in a tool result."""

    type: Literal["text"] = "text"
    text: str


class MCPImageContent(BaseModel):
    """Image content in a tool result."""

    type: Literal["image"] = "image"
    data: str = Field(..., description="Base64-encoded image data")
    mime_type: str = Field(alias="mimeType", default="image/png")


class MCPResourceContent(BaseModel):
    """Resource reference in a tool result."""

    type: Literal["resource"] = "resource"
    resource: dict[str, Any]


MCPContent = MCPTextContent | MCPImageContent | MCPResourceContent


class MCPToolResult(BaseModel):
    """Result of a tool call."""

    content: list[MCPContent] = Field(default_factory=list)
    is_error: bool = Field(alias="isError", default=False)

    model_config = {"populate_by_name": True}


# =============================================================================
# API Request/Response Models
# =============================================================================


class MCPServerStatus(BaseModel):
    """Status of an MCP server connection."""

    name: str
    connected: bool
    tools: list[str] = Field(default_factory=list)
    last_error: str | None = None
    connected_at: datetime | None = None


class MCPRegistryStatus(BaseModel):
    """Status of all MCP servers in the registry."""

    servers: list[MCPServerStatus] = Field(default_factory=list)
    total_tools: int = 0


class CallToolRequest(BaseModel):
    """API request to call an MCP tool."""

    server_name: str = Field(..., description="Name of the MCP server")
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict)


class CallToolResponse(BaseModel):
    """API response from calling an MCP tool."""

    success: bool
    content: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    execution_time_ms: float | None = None


# =============================================================================
# Image Generation Specific
# =============================================================================


class GenerateImageRequest(BaseModel):
    """Request to generate an image via MCP."""

    prompt: str = Field(..., description="Text prompt describing the image")
    server_name: str = Field(default="nano-banana", description="MCP server to use")
    reference_images: list[str] | None = Field(
        None, description="Optional paths to reference images"
    )


class GenerateImageResponse(BaseModel):
    """Response from image generation."""

    success: bool
    image_data: str | None = None
    """Base64-encoded image data."""
    image_url: str | None = None
    """URL to hosted image (if available)."""
    file_path: str | None = None
    """Local file path (if saved)."""
    error: str | None = None


class EditImageRequest(BaseModel):
    """Request to edit an existing image via MCP."""

    image_path: str = Field(..., description="Path to the image to edit")
    prompt: str = Field(..., description="Edit instructions")
    server_name: str = Field(default="nano-banana")
    reference_images: list[str] | None = None


class EditImageResponse(BaseModel):
    """Response from image editing."""

    success: bool
    image_data: str | None = None
    image_url: str | None = None
    file_path: str | None = None
    error: str | None = None

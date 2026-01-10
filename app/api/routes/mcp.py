"""
MCP API Routes.

Provides REST API endpoints for:
- Listing registered MCP servers and their tools
- Calling tools on specific servers
- Image generation via Nano Banana

These endpoints enable the frontend to leverage external MCP tools
for features like infographic generation.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.mcp.client import extract_all_content, extract_image, extract_text
from app.mcp.registry import get_registry, setup_default_servers
from app.mcp.schemas import (
    CallToolRequest,
    CallToolResponse,
    EditImageRequest,
    EditImageResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    MCPRegistryStatus,
    MCPServerConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP Tools"])


# =============================================================================
# Schemas
# =============================================================================


class ServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    connected: bool
    tools: list[str]
    description: str | None = None


class ToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str | None = None
    server: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ListToolsResponse(BaseModel):
    """Response listing all available tools."""

    tools: list[ToolInfo]
    total: int


class ConnectServerRequest(BaseModel):
    """Request to connect to an MCP server."""

    server_name: str


class ConnectServerResponse(BaseModel):
    """Response from connecting to a server."""

    success: bool
    server_name: str
    tools: list[str] = Field(default_factory=list)
    error: str | None = None


# =============================================================================
# Lifecycle Management
# =============================================================================


async def ensure_registry_initialized() -> None:
    """Ensure the registry is initialized with default servers."""
    registry = get_registry()

    # Only setup if no servers registered yet
    if not registry.server_names:
        if settings.GEMINI_API_KEY:
            await setup_default_servers(
                gemini_api_key=settings.GEMINI_API_KEY,
                imgbb_api_key=settings.IMGBB_API_KEY or None,
            )
            logger.info("MCP registry initialized with default servers")
        else:
            logger.warning(
                "GEMINI_API_KEY not set - MCP image generation unavailable"
            )


# =============================================================================
# Server Management Endpoints
# =============================================================================


@router.get("/status", response_model=MCPRegistryStatus)
async def get_mcp_status() -> MCPRegistryStatus:
    """
    Get status of all MCP servers.

    Returns connection status and available tools for each server.
    """
    await ensure_registry_initialized()
    registry = get_registry()
    return registry.get_status()


@router.get("/servers", response_model=list[ServerInfo])
async def list_servers() -> list[ServerInfo]:
    """
    List all registered MCP servers.

    Returns basic info about each server including connection status.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    servers = []
    for name in registry.server_names:
        config = registry.get_config(name)
        client = registry.get_client(name)

        servers.append(
            ServerInfo(
                name=name,
                connected=client is not None and client.connected,
                tools=client.tool_names if client and client.connected else [],
                description=config.description if config else None,
            )
        )

    return servers


@router.post("/servers/connect", response_model=ConnectServerResponse)
async def connect_server(
    request: ConnectServerRequest,
) -> ConnectServerResponse:
    """
    Connect to a specific MCP server.

    Spawns the server process and performs initialization.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    try:
        client = await registry.connect(request.server_name)
        return ConnectServerResponse(
            success=True,
            server_name=request.server_name,
            tools=client.tool_names,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to connect to {request.server_name}: {e}")
        return ConnectServerResponse(
            success=False,
            server_name=request.server_name,
            error=str(e),
        )


@router.post("/servers/disconnect/{server_name}")
async def disconnect_server(
    server_name: str,
) -> dict[str, Any]:
    """
    Disconnect from an MCP server.

    Terminates the server process and cleans up resources.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    await registry.disconnect(server_name)
    return {"success": True, "server_name": server_name}


# =============================================================================
# Tool Discovery Endpoints
# =============================================================================


@router.get("/tools", response_model=ListToolsResponse)
async def list_tools() -> ListToolsResponse:
    """
    List all available tools across all connected servers.

    Returns tool metadata including input schemas.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    tools_by_server = registry.list_all_tools()
    all_tools: list[ToolInfo] = []

    for server_name, tools in tools_by_server.items():
        for tool in tools:
            all_tools.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    server=server_name,
                    input_schema=tool.input_schema.model_dump(),
                )
            )

    return ListToolsResponse(tools=all_tools, total=len(all_tools))


@router.get("/tools/{server_name}", response_model=list[ToolInfo])
async def list_server_tools(
    server_name: str,
) -> list[ToolInfo]:
    """
    List tools available on a specific server.

    The server must be connected first.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    client = registry.get_client(server_name)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_name}' not connected",
        )

    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            server=server_name,
            input_schema=tool.input_schema.model_dump(),
        )
        for tool in client.tools
    ]


# =============================================================================
# Tool Execution Endpoints
# =============================================================================


@router.post("/call", response_model=CallToolResponse)
async def call_tool(
    request: CallToolRequest,
) -> CallToolResponse:
    """
    Call a tool on an MCP server.

    Generic endpoint for calling any tool with arbitrary arguments.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    import time

    start = time.time()

    try:
        result = await registry.call_tool(
            server_name=request.server_name,
            tool_name=request.tool_name,
            arguments=request.arguments,
            timeout=settings.MCP_TOOL_TIMEOUT,
        )

        elapsed = (time.time() - start) * 1000

        return CallToolResponse(
            success=not result.is_error,
            content=extract_all_content(result),
            error=extract_text(result) if result.is_error else None,
            execution_time_ms=elapsed,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        elapsed = (time.time() - start) * 1000
        return CallToolResponse(
            success=False,
            error=str(e),
            execution_time_ms=elapsed,
        )


# =============================================================================
# Image Generation Endpoints (Nano Banana)
# =============================================================================


@router.post("/images/generate", response_model=GenerateImageResponse)
async def generate_image(
    request: GenerateImageRequest,
) -> GenerateImageResponse:
    """
    Generate an image using AI.

    Uses Nano Banana MCP server with Google Gemini 2.5 Flash.

    The prompt should describe the image you want to create.
    Results include base64-encoded image data.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    # Check if API key is configured
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image generation not configured. Set GEMINI_API_KEY.",
        )

    try:
        # Connect if needed
        client = registry.get_client(request.server_name)
        if not client:
            await registry.connect(request.server_name)

        # Call generate_image tool
        result = await registry.call_tool(
            server_name=request.server_name,
            tool_name="generate_image",
            arguments={"prompt": request.prompt},
            timeout=settings.MCP_TOOL_TIMEOUT,
        )

        if result.is_error:
            error_msg = extract_text(result) or "Image generation failed"
            return GenerateImageResponse(success=False, error=error_msg)

        # Extract image from result
        image_data = extract_image(result)
        if image_data:
            base64_data, _mime_type = image_data
            return GenerateImageResponse(
                success=True,
                image_data=base64_data,
            )

        # Check for URL or file path in text response
        text = extract_text(result)
        if text:
            if text.startswith("http"):
                return GenerateImageResponse(success=True, image_url=text)
            return GenerateImageResponse(success=True, file_path=text)

        return GenerateImageResponse(
            success=False,
            error="No image data in response",
        )

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return GenerateImageResponse(success=False, error=str(e))


@router.post("/images/edit", response_model=EditImageResponse)
async def edit_image(
    request: EditImageRequest,
) -> EditImageResponse:
    """
    Edit an existing image using AI.

    Uses Nano Banana MCP server with Google Gemini 2.5 Flash.

    Provide the path to an existing image and a prompt describing
    the edits you want to make.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image generation not configured. Set GEMINI_API_KEY.",
        )

    try:
        # Connect if needed
        client = registry.get_client(request.server_name)
        if not client:
            await registry.connect(request.server_name)

        # Build arguments
        args: dict[str, Any] = {
            "imagePath": request.image_path,
            "prompt": request.prompt,
        }
        if request.reference_images:
            args["referenceImages"] = request.reference_images

        # Call edit_image tool
        result = await registry.call_tool(
            server_name=request.server_name,
            tool_name="edit_image",
            arguments=args,
            timeout=settings.MCP_TOOL_TIMEOUT,
        )

        if result.is_error:
            error_msg = extract_text(result) or "Image editing failed"
            return EditImageResponse(success=False, error=error_msg)

        # Extract image
        image_data = extract_image(result)
        if image_data:
            base64_data, _mime_type = image_data
            return EditImageResponse(success=True, image_data=base64_data)

        text = extract_text(result)
        if text:
            if text.startswith("http"):
                return EditImageResponse(success=True, image_url=text)
            return EditImageResponse(success=True, file_path=text)

        return EditImageResponse(
            success=False,
            error="No image data in response",
        )

    except Exception as e:
        logger.error(f"Image editing failed: {e}")
        return EditImageResponse(success=False, error=str(e))


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health")
async def mcp_health() -> dict[str, Any]:
    """
    Check health of MCP subsystem.

    Returns status of all connected servers.
    """
    await ensure_registry_initialized()
    registry = get_registry()

    health = await registry.health_check()

    return {
        "enabled": settings.MCP_ENABLED,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "servers": health,
        "total_connected": sum(1 for v in health.values() if v),
    }

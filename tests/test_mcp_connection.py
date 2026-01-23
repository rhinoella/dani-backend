import asyncio
import pytest
from app.mcp.registry import setup_default_servers

@pytest.mark.asyncio
async def test_mcp_connection():
    print("Setting up MCP servers...")
    registry = await setup_default_servers()
    print(f"Registered servers: {registry.server_names}")

    print("Connecting to servers...")
    results = await registry.connect_all(ignore_errors=True)
    print(f"Connection results: {results}")

    print(f"Connected servers: {registry.connected_servers}")

    # Try to call a tool if connected
    if "nano-banana" in registry.connected_servers:
        print("Testing nano-banana tool call...")
        try:
            result = await registry.call_tool("nano-banana", "generate_image", {"prompt": "test"})
            print(f"Tool call result: {result}")
        except Exception as e:
            print(f"Tool call failed: {e}")
    else:
        print("nano-banana not connected, skipping tool test")
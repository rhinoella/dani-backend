#!/usr/bin/env python3
"""
Test MCP Image Generation
"""

import asyncio
from app.mcp.client import MCPClient
from app.mcp.schemas import MCPServerConfig

async def test_mcp_image_generation():
    print('=== MCP IMAGE GENERATION TEST ===')

    try:
        # Create proper config for MCP client
        config = MCPServerConfig(
            command='npx',
            args=['-y', '@modelcontextprotocol/server-everything', '--port', '3001'],
            env={'NODE_ENV': 'development'}
        )

        client = MCPClient(config)
        print('Connecting to MCP server...')
        await client.connect()

        print('Listing available tools...')
        tools = await client.list_tools()
        print(f'Found {len(tools)} tools')

        # Check if image generation tool exists
        image_tools = [t for t in tools if 'image' in t.get('name', '').lower()]
        if image_tools:
            print(f'Image tools available: {[t.get("name") for t in image_tools]}')

            # Try to generate an image
            print('Generating test image...')
            result = await client.generate_image('A beautiful sunset over mountains with a lake')
            print(f'Result type: {type(result)}')
            print(f'Result: {result}')

            if result and 'content' in result and len(result['content']) > 0:
                content = result['content'][0]
                if 'image_url' in content or 'data' in content:
                    print('✅ IMAGE GENERATION SUCCESSFUL!')
                    if 'image_url' in content:
                        print(f'Image URL: {content["image_url"]}')
                    if 'data' in content:
                        print(f'Image data length: {len(content["data"])}')
                else:
                    print('❌ No image URL or data in response')
                    print(f'Content keys: {list(content.keys()) if isinstance(content, dict) else "not dict"}')
            else:
                print('❌ Empty or invalid response')
        else:
            print('❌ No image generation tools found')
            print('Available tools:')
            for tool in tools[:5]:  # Show first 5
                print(f'  - {tool.get("name", "unnamed")}')

        await client.disconnect()

    except Exception as e:
        print(f'❌ MCP test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_image_generation())
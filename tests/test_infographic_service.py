#!/usr/bin/env python3
"""
Test script for infographic service.
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import pytest
from app.services.infographic_service import InfographicService
from app.database.models.infographic import InfographicStyle

@pytest.mark.asyncio
async def test_infographic():
    print('Testing infographic service...')

    # Create service (will lazy load MCP registry)
    service = InfographicService()

    try:
        result = await service.generate(
            request="Generate an infographic for a test meeting",
            topic="Test Meeting",
            style=InfographicStyle.MODERN,
            width=1024,
            height=768,
            persist=False  # Don't persist to database/S3 for test
        )

        if result and 'id' in result:
            print('✅ Infographic generation successful!')
            print(f'Infographic ID: {result["id"]}')
            print(f'Status: {result.get("status", "unknown")}')
            assert True
        else:
            print('❌ Infographic generation failed')
            print(f'Result: {result}')
            assert False, f"Infographic generation failed, result: {result}"

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        assert False, f"Infographic generation failed with error: {e}"
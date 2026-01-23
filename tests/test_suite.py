#!/usr/bin/env python3
"""
Comprehensive test suite for DANI Engine
"""

import asyncio
import requests
import time
import json
from typing import Dict, List

def test_api_endpoints():
    """Test 1-2: API endpoint availability"""
    print("=== TEST 1-2: API Endpoint Availability ===")

    endpoints = [
        ("http://localhost:8000/", "Root"),
        ("http://localhost:8000/docs", "Docs"),
        ("http://localhost:8000/openapi.json", "OpenAPI Schema"),
        ("http://localhost:8000/api/v1/health", "Health Check"),
    ]

    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=5)
            status = "✅" if response.status_code in [200, 302] else "❌"
            print(f"{status} {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)[:50]}...")

def test_retrieval_system():
    """Test 3: Retrieval system functionality"""
    print("\n=== TEST 3: Retrieval System ===")

    try:
        from app.services.retrieval_service import RetrievalService

        async def run_retrieval_test():
            service = RetrievalService()
            queries = [
                "What did we discuss about marketing strategy?",
                "strategy meetings",
                "digital transformation"
            ]

            for query in queries:
                try:
                    result = await service.search_with_confidence(query, limit=3)
                    confidence = result['confidence']['level']
                    chunk_count = len(result['chunks'])
                    status = "✅" if chunk_count > 0 else "❌"
                    print(f"{status} Query '{query[:30]}...': {chunk_count} chunks, {confidence} confidence")
                except Exception as e:
                    print(f"❌ Query '{query[:30]}...': {str(e)[:50]}...")

        asyncio.run(run_retrieval_test())

    except Exception as e:
        print(f"❌ Retrieval system test failed: {e}")

def test_database_connections():
    """Test 4: Database connectivity"""
    print("\n=== TEST 4: Database Connections ===")

    try:
        from app.vectorstore.qdrant import QdrantStore
        from app.database.connection import get_db

        # Test Qdrant
        store = QdrantStore()
        collections = store.client.get_collections().collections
        print(f"✅ Qdrant: {len(collections)} collections found")

        # Test PostgreSQL
        async def test_db():
            try:
                async for session in get_db():
                    result = await session.execute("SELECT 1")
                    print("✅ PostgreSQL: Connection successful")
                    break
            except Exception as e:
                print(f"❌ PostgreSQL: {str(e)[:50]}...")

        asyncio.run(test_db())

    except Exception as e:
        print(f"❌ Database test failed: {e}")

def test_llm_integration():
    """Test 5: LLM integration"""
    print("\n=== TEST 5: LLM Integration ===")

    try:
        from app.llm.client import OllamaClient

        async def test_llm():
            client = OllamaClient()
            try:
                response = await client.generate("Hello, test message", max_tokens=50)
                if response and len(response.strip()) > 0:
                    print("✅ LLM: Response generated successfully")
                    print(f"   Response length: {len(response)} chars")
                else:
                    print("❌ LLM: Empty response")
            except Exception as e:
                print(f"❌ LLM: {str(e)[:50]}...")

        asyncio.run(test_llm())

    except Exception as e:
        print(f"❌ LLM test failed: {e}")

def test_mcp_functionality():
    """Test 6: MCP functionality"""
    print("\n=== TEST 6: MCP Functionality ===")

    try:
        from app.mcp.client import MCPClient

        async def test_mcp():
            client = MCPClient()
            try:
                await client.connect()
                tools = await client.list_tools()
                print(f"✅ MCP: Connected, {len(tools)} tools available")

                # Test image generation
                print("   Testing image generation...")
                result = await client.generate_image("A simple test image of a cat")
                if result and 'content' in result and len(result['content']) > 0:
                    print("   ✅ Image generation: Success")
                else:
                    print("   ❌ Image generation: Failed")

                await client.disconnect()

            except Exception as e:
                print(f"❌ MCP: {str(e)[:50]}...")

        asyncio.run(test_mcp())

    except Exception as e:
        print(f"❌ MCP test failed: {e}")

def test_error_handling():
    """Test 7: Error handling"""
    print("\n=== TEST 7: Error Handling ===")

    # Test invalid API calls
    try:
        response = requests.get("http://localhost:8000/api/v1/nonexistent", timeout=5)
        if response.status_code in [404, 422]:
            print("✅ Error handling: Proper 404/422 responses")
        else:
            print(f"❌ Error handling: Unexpected status {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")

    # Test malformed queries
    try:
        from app.services.retrieval_service import RetrievalService

        async def test_malformed():
            service = RetrievalService()
            try:
                result = await service.search_with_confidence("", limit=1)
                print("❌ Error handling: Should reject empty query")
            except Exception:
                print("✅ Error handling: Properly rejects empty query")

        asyncio.run(test_malformed())

    except Exception as e:
        print(f"❌ Malformed query test failed: {e}")

def test_performance():
    """Test 8: Performance benchmarks"""
    print("\n=== TEST 8: Performance Benchmarks ===")

    try:
        from app.services.retrieval_service import RetrievalService

        async def benchmark():
            service = RetrievalService()

            # Test retrieval speed
            start_time = time.time()
            result = await service.search_with_confidence("test query", limit=5)
            end_time = time.time()

            duration = end_time - start_time
            if duration < 2.0:  # Should be fast
                print(".2f")
            else:
                print(".2f")
            # Test concurrent requests
            import asyncio
            start_time = time.time()
            tasks = [service.search_with_confidence(f"query {i}", limit=3) for i in range(3)]
            await asyncio.gather(*tasks)
            end_time = time.time()

            concurrent_duration = end_time - start_time
            if concurrent_duration < 5.0:
                print(".2f")
            else:
                print(".2f")
        asyncio.run(benchmark())

    except Exception as e:
        print(f"❌ Performance test failed: {e}")

def test_configuration():
    """Test 9: Configuration validation"""
    print("\n=== TEST 9: Configuration Validation ===")

    try:
        from app.core.config import settings

        # Check required settings
        required_settings = [
            'QDRANT_URL',
            'DATABASE_URL',
            'OLLAMA_BASE_URL',
            'EMBEDDING_MODEL',
            'LLM_MODEL'
        ]

        missing = []
        for setting in required_settings:
            value = getattr(settings, setting, None)
            if not value or str(value).startswith('__MISSING__'):
                missing.append(setting)

        if not missing:
            print("✅ Configuration: All required settings present")
        else:
            print(f"❌ Configuration: Missing settings: {missing}")

        # Check model availability
        print("   Checking model availability...")
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if settings.LLM_MODEL in model_names:
                    print(f"   ✅ LLM model '{settings.LLM_MODEL}' available")
                else:
                    print(f"   ❌ LLM model '{settings.LLM_MODEL}' not found")
            else:
                print("   ❌ Cannot check model availability")
        except Exception as e:
            print(f"   ❌ Model check failed: {str(e)[:50]}...")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def test_edge_cases():
    """Test 10: Edge cases and boundary conditions"""
    print("\n=== TEST 10: Edge Cases ===")

    try:
        from app.services.retrieval_service import RetrievalService
        from app.utils.query_processor import QueryProcessor

        async def test_edges():
            service = RetrievalService()
            processor = QueryProcessor()

            # Test very long query
            long_query = "What " * 100 + "is the meaning of life?"
            try:
                intent = processor.detect_intent(long_query)
                print("✅ Edge case: Handles long queries")
            except Exception:
                print("❌ Edge case: Fails on long queries")

            # Test special characters
            special_query = "What about AI/ML & tech???!!!"
            try:
                result = await service.search_with_confidence(special_query, limit=2)
                print("✅ Edge case: Handles special characters")
            except Exception:
                print("❌ Edge case: Fails on special characters")

            # Test empty result handling
            try:
                # This should handle gracefully even if no results
                result = await service.search_with_confidence("xyz_nonexistent_topic_123", limit=1)
                print("✅ Edge case: Handles queries with no results")
            except Exception as e:
                print(f"❌ Edge case: Crashes on no results: {str(e)[:50]}...")

        asyncio.run(test_edges())

    except Exception as e:
        print(f"❌ Edge case test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting DANI Engine Comprehensive Test Suite")
    print("=" * 60)

    test_api_endpoints()
    test_retrieval_system()
    test_database_connections()
    test_llm_integration()
    test_mcp_functionality()
    test_error_handling()
    test_performance()
    test_configuration()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
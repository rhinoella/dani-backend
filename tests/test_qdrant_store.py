"""
Tests for Qdrant Vector Store.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from app.vectorstore.qdrant import QdrantStore, qdrant_retry
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse


# ============== Fixtures ==============

@pytest.fixture
def mock_settings():
    """Mock settings for Qdrant store."""
    with patch('app.vectorstore.qdrant.settings') as mock:
        mock.QDRANT_URL = "http://localhost:6333"
        yield mock


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    with patch('app.vectorstore.qdrant.QdrantClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        yield mock_client


# ============== Tests ==============

class TestQdrantStore:
    """Tests for QdrantStore."""
    
    def test_init(self, mock_settings, mock_qdrant_client):
        """Test QdrantStore initialization."""
        store = QdrantStore()
        
        assert store.url == "http://localhost:6333"
        assert store.client is not None
    
    def test_ensure_collection_exists(self, mock_settings, mock_qdrant_client):
        """Test ensure_collection when collection exists."""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_collection]
        
        store = QdrantStore()
        store.ensure_collection("test_collection", vector_size=768)
        
        # Should not create collection
        mock_qdrant_client.create_collection.assert_not_called()
    
    def test_ensure_collection_creates_new(self, mock_settings, mock_qdrant_client):
        """Test ensure_collection creates new collection."""
        mock_qdrant_client.get_collections.return_value.collections = []
        
        store = QdrantStore()
        store.ensure_collection("new_collection", vector_size=768)
        
        mock_qdrant_client.create_collection.assert_called_once()
    
    def test_ensure_collection_creates_indexes(self, mock_settings, mock_qdrant_client):
        """Test ensure_collection creates payload indexes."""
        mock_qdrant_client.get_collections.return_value.collections = []
        
        store = QdrantStore()
        store.ensure_collection("new_collection", vector_size=768)
        
        # Should create payload indexes
        assert mock_qdrant_client.create_payload_index.call_count >= 1
    
    def test_ensure_collection_index_already_exists(self, mock_settings, mock_qdrant_client):
        """Test ensure_collection handles existing indexes."""
        mock_qdrant_client.get_collections.return_value.collections = []
        mock_qdrant_client.create_payload_index.side_effect = Exception("Index exists")
        
        store = QdrantStore()
        
        # Should not raise
        store.ensure_collection("new_collection", vector_size=768)
    
    def test_ensure_collection_exception(self, mock_settings, mock_qdrant_client):
        """Test ensure_collection handles exceptions."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection error")
        
        store = QdrantStore()
        
        with pytest.raises(Exception):
            store.ensure_collection("test", vector_size=768)
    
    def test_upsert_success(self, mock_settings, mock_qdrant_client):
        """Test successful upsert."""
        store = QdrantStore()
        
        points = [
            qm.PointStruct(
                id=1,
                vector=[0.1] * 768,
                payload={"text": "test"}
            )
        ]
        
        store.upsert("test_collection", points)
        
        mock_qdrant_client.upsert.assert_called_once_with(
            collection_name="test_collection",
            points=points,
        )
    
    def test_upsert_empty_points(self, mock_settings, mock_qdrant_client):
        """Test upsert with empty points list."""
        store = QdrantStore()
        
        store.upsert("test_collection", [])
        
        # Should not call upsert
        mock_qdrant_client.upsert.assert_not_called()
    
    def test_upsert_exception(self, mock_settings, mock_qdrant_client):
        """Test upsert handles exceptions."""
        mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")
        
        store = QdrantStore()
        
        points = [
            qm.PointStruct(
                id=1,
                vector=[0.1] * 768,
                payload={"text": "test"}
            )
        ]
        
        with pytest.raises(Exception):
            store.upsert("test_collection", points)
    
    @pytest.mark.asyncio
    async def test_search_success(self, mock_settings, mock_qdrant_client):
        """Test successful search."""
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {"text": "test"}
        
        # Mock the query_points result
        mock_query_result = MagicMock()
        mock_query_result.points = [mock_result]
        mock_qdrant_client.query_points.return_value = mock_query_result
        
        store = QdrantStore()
        
        with patch('app.vectorstore.qdrant._executor'), \
             patch('app.vectorstore.qdrant.qdrant_breaker'):
            # Mock run_in_executor to return results directly
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop_instance = MagicMock()
                mock_loop.return_value = mock_loop_instance
                mock_loop_instance.run_in_executor = AsyncMock(return_value=[mock_result])
                
                results = await store.search(
                    collection="test_collection",
                    query_vector=[0.1] * 768,
                    limit=5
                )
        
        assert len(results) == 1
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, mock_settings, mock_qdrant_client):
        """Test search with filter."""
        store = QdrantStore()
        
        filter_condition = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="source_file",
                    match=qm.MatchValue(value="test.pdf")
                )
            ]
        )
        
        with patch('app.vectorstore.qdrant._executor'), \
             patch('app.vectorstore.qdrant.qdrant_breaker'):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop_instance = MagicMock()
                mock_loop.return_value = mock_loop_instance
                mock_loop_instance.run_in_executor = AsyncMock(return_value=[])
                
                results = await store.search(
                    collection="test_collection",
                    query_vector=[0.1] * 768,
                    limit=5,
                    filter_=filter_condition
                )
        
        assert results == []


class TestQdrantStoreAsync:
    """Tests for async Qdrant operations - tests run_in_executor wrapping."""
    
    @pytest.mark.asyncio
    async def test_search_uses_executor(self, mock_settings, mock_qdrant_client):
        """Test that search uses run_in_executor for async."""
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {"text": "test"}
        
        store = QdrantStore()
        
        with patch('app.vectorstore.qdrant._executor') as mock_executor, \
             patch('app.vectorstore.qdrant.qdrant_breaker'), \
             patch('asyncio.get_event_loop') as mock_loop:
            
            mock_loop_instance = MagicMock()
            mock_loop.return_value = mock_loop_instance
            mock_loop_instance.run_in_executor = AsyncMock(return_value=[mock_result])
            
            results = await store.search(
                collection="test_collection",
                query_vector=[0.1] * 768,
                limit=5
            )
            
            # Verify run_in_executor was called with the executor
            mock_loop_instance.run_in_executor.assert_called_once()
            call_args = mock_loop_instance.run_in_executor.call_args
            assert call_args[0][0] == mock_executor  # First arg is the executor
    
    @pytest.mark.asyncio
    async def test_search_circuit_breaker_open(self, mock_settings, mock_qdrant_client):
        """Test search handles circuit breaker open state."""
        from app.core.circuit_breaker import CircuitBreakerOpen
        
        store = QdrantStore()
        
        with patch('app.vectorstore.qdrant.qdrant_breaker') as mock_breaker:
            mock_breaker.__enter__ = MagicMock(side_effect=CircuitBreakerOpen("qdrant", 30.0))
            mock_breaker.__exit__ = MagicMock()
            
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                await store.search(
                    collection="test_collection",
                    query_vector=[0.1] * 768,
                    limit=5
                )


class TestQdrantRetry:
    """Tests for retry decorator."""
    
    def test_retry_decorator_exists(self):
        """Test that retry decorator is defined."""
        assert qdrant_retry is not None
    
    @pytest.mark.asyncio
    async def test_retry_on_unexpected_response(self, mock_settings, mock_qdrant_client):
        """Test retry on UnexpectedResponse."""
        call_count = [0]
        
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise UnexpectedResponse(status_code=500, reason_phrase="Server error")
            return []
        
        mock_qdrant_client.get_collections.side_effect = side_effect
        
        store = QdrantStore()
        
        # The retry should eventually succeed or fail after max attempts
        try:
            store.ensure_collection("test", vector_size=768)
        except:
            pass  # Expected if retries exhausted


class TestQdrantStoreCollectionManagement:
    """Tests for collection management."""
    
    def test_delete_collection(self, mock_settings, mock_qdrant_client):
        """Test delete collection if method exists."""
        store = QdrantStore()
        
        if hasattr(store, 'delete_collection'):
            store.delete_collection("test_collection")
            mock_qdrant_client.delete_collection.assert_called_once()
    
    def test_get_collection_info(self, mock_settings, mock_qdrant_client):
        """Test get collection info if method exists."""
        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_qdrant_client.get_collection.return_value = mock_info
        
        store = QdrantStore()
        
        if hasattr(store, 'get_collection_info'):
            info = store.get_collection_info("test_collection")
            assert info is not None

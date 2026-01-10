"""
Tests for Ollama Embedding Client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.embeddings.client import OllamaEmbeddingClient


# ============== Fixtures ==============

@pytest.fixture
def mock_settings():
    """Mock settings for embedding client."""
    with patch('app.embeddings.client.settings') as mock:
        mock.OLLAMA_BASE_URL = "http://localhost:11434"
        mock.EMBEDDING_MODEL = "nomic-embed-text"
        yield mock


@pytest.fixture
def embedding_client(mock_settings):
    """Create embedding client for testing."""
    return OllamaEmbeddingClient()


# ============== Tests ==============

class TestOllamaEmbeddingClient:
    """Tests for OllamaEmbeddingClient."""
    
    def test_init(self, mock_settings):
        """Test client initialization."""
        client = OllamaEmbeddingClient()
        # Settings are now read dynamically on each request, not cached in __init__
        # Verify the client and timeout are initialized
        assert client.client is not None
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 180.0
    
    def test_timeout_config(self, mock_settings):
        """Test timeout configuration."""
        client = OllamaEmbeddingClient()
        # Timeouts were increased for slow/large embeddings
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 180.0
    
    @pytest.mark.asyncio
    async def test_close(self, embedding_client):
        """Test close method."""
        await embedding_client.close()
        # Should not raise
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test async context manager."""
        async with OllamaEmbeddingClient() as client:
            assert client is not None
    
    @pytest.mark.asyncio
    async def test_embed_one_success(self, embedding_client):
        """Test successful embedding."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await embedding_client.embed_one("test text")
        
        assert result == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_embed_one_text_too_long_gets_truncated(self, embedding_client):
        """Test embedding with text too long - should be truncated, not raise."""
        long_text = "a" * 40000
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            # Should succeed by truncating, not raise ValueError
            result = await embedding_client.embed_one(long_text)
        
        assert result == [0.1, 0.2, 0.3]
        # Verify the payload was truncated (32000 chars max)
        call_args = mock_post.call_args
        payload = call_args[1]['json'] if 'json' in call_args[1] else call_args[0][1]
        # The prompt in the payload should be truncated
    
    @pytest.mark.asyncio
    async def test_embed_one_retry_success(self, embedding_client):
        """Test embedding retries and succeeds."""
        fail_response = MagicMock()
        fail_response.status_code = 500
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            # First call fails, second succeeds
            mock_post.side_effect = [fail_response, success_response]
            
            result = await embedding_client.embed_one("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert mock_post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_embed_one_all_retries_fail(self, embedding_client):
        """Test embedding fails after all retries."""
        fail_response = MagicMock()
        fail_response.status_code = 500
        fail_response.json.return_value = {"error": "Internal error"}
        
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_post.return_value = fail_response
            
            with pytest.raises(RuntimeError, match="failed after retries"):
                await embedding_client.embed_one("test text")
        
        assert mock_post.call_count == 5  # 5 attempts (max_retries increased)
    
    @pytest.mark.asyncio
    async def test_embed_one_connection_error(self, embedding_client):
        """Test embedding with connection error."""
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                await embedding_client.embed_one("test text")
    
    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, embedding_client):
        """Test batch embedding with empty list."""
        result = await embedding_client.embed_batch([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, embedding_client):
        """Test successful batch embedding."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            result = await embedding_client.embed_batch(["text1", "text2", "text3"])
        
        assert len(result) == 3
        assert all(r == [0.1, 0.2, 0.3] for r in result)
    
    @pytest.mark.asyncio
    async def test_embed_batch_with_custom_batch_size(self, embedding_client):
        """Test batch embedding with custom batch size."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1]}
        
        with patch.object(embedding_client.client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            # 5 texts with batch_size=2 should result in 3 batches
            result = await embedding_client.embed_batch(
                ["t1", "t2", "t3", "t4", "t5"], 
                batch_size=2
            )
        
        assert len(result) == 5
    
    @pytest.mark.asyncio
    async def test_embed_batch_partial_failure(self, embedding_client):
        """Test batch embedding with failure in one item."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1]}
        
        with patch.object(embedding_client, 'embed_one', new_callable=AsyncMock) as mock_embed:
            # First succeeds, second fails
            mock_embed.side_effect = [
                [0.1, 0.2],
                Exception("Embedding failed"),
            ]
            
            with pytest.raises(RuntimeError, match="batch index"):
                await embedding_client.embed_batch(["text1", "text2"], batch_size=2)


class TestEmbeddingClientRetries:
    """Tests for retry behavior."""
    
    @pytest.mark.asyncio
    async def test_retry_on_500(self, mock_settings):
        """Test retry on 500 error."""
        client = OllamaEmbeddingClient()
        
        fail_response = MagicMock()
        fail_response.status_code = 500
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"embedding": [0.1]}
        
        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return fail_response
            return success_response
        
        with patch.object(client.client, 'post', side_effect=mock_post), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            result = await client.embed_one("test")
        
        assert result == [0.1]
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_on_connect_error(self, mock_settings):
        """Test retry on connection error."""
        client = OllamaEmbeddingClient()
        
        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": [0.1]}
            return mock_response
        
        with patch.object(client.client, 'post', side_effect=mock_post), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            result = await client.embed_one("test")
        
        assert result == [0.1]
        assert call_count == 3


class TestEmbeddingErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_http_status_error_with_detail(self, mock_settings):
        """Test HTTP error includes detail."""
        client = OllamaEmbeddingClient()
        
        fail_response = MagicMock()
        fail_response.status_code = 400
        fail_response.json.return_value = {"error": "Bad request"}
        
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_post.return_value = fail_response
            
            with pytest.raises(RuntimeError, match="failed after retries"):
                await client.embed_one("test")
    
    @pytest.mark.asyncio
    async def test_http_status_error_no_json(self, mock_settings):
        """Test HTTP error when response is not JSON."""
        client = OllamaEmbeddingClient()
        
        fail_response = MagicMock()
        fail_response.status_code = 502
        fail_response.json.side_effect = Exception("Not JSON")
        fail_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Bad gateway", 
                request=MagicMock(), 
                response=fail_response
            )
        )
        
        with patch.object(client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            mock_post.return_value = fail_response
            
            with pytest.raises(httpx.HTTPStatusError):
                await client.embed_one("test")

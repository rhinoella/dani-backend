"""
Tests for Ollama LLM Client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import json

from app.llm.ollama import OllamaClient
from app.core.circuit_breaker import CircuitBreakerOpen


# ============== Fixtures ==============

@pytest.fixture
def mock_settings():
    """Mock settings for Ollama client."""
    with patch('app.llm.ollama.settings') as mock:
        mock.OLLAMA_BASE_URL = "http://localhost:11434"
        mock.LLM_MODEL = "llama3.2:3b"
        yield mock


@pytest.fixture
def ollama_client(mock_settings):
    """Create Ollama client for testing."""
    return OllamaClient()


# ============== Tests ==============

class TestOllamaClient:
    """Tests for OllamaClient."""
    
    def test_init(self, mock_settings):
        """Test OllamaClient initialization."""
        client = OllamaClient()
        # Settings are now read dynamically on each request, not cached in __init__
        # Verify the client and timeout are initialized
        assert client.client is not None
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 600.0
    
    def test_timeout_config(self, mock_settings):
        """Test timeout configuration."""
        client = OllamaClient()
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 600.0  # 10 minutes for generation
    
    @pytest.mark.asyncio
    async def test_close(self, ollama_client):
        """Test close method."""
        await ollama_client.close()
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test async context manager."""
        async with OllamaClient() as client:
            assert client is not None
    
    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_client):
        """Test successful text generation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "Generated text response"}
        
        with patch.object(ollama_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('app.llm.ollama.ollama_breaker') as mock_breaker:
            mock_post.return_value = mock_response
            mock_breaker.__aenter__ = AsyncMock()
            mock_breaker.__aexit__ = AsyncMock()
            
            result = await ollama_client.generate("Test prompt")
        
        assert result == "Generated text response"
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_empty_response(self, ollama_client):
        """Test generation with empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": ""}
        
        # Create a proper async context manager mock
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            mock_post.return_value = mock_response
            
            # ValueError is caught by generic Exception handler and wrapped in RuntimeError
            with pytest.raises(RuntimeError, match="LLM generation failed.*empty response"):
                await ollama_client.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_circuit_breaker_open(self, ollama_client):
        """Test generation when circuit breaker is open."""
        # Create a mock that raises on __aenter__
        mock_breaker_cm = MagicMock()
        mock_breaker_cm.__aenter__ = AsyncMock(
            side_effect=CircuitBreakerOpen("ollama", 30.0)
        )
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                await ollama_client.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, ollama_client):
        """Test generation timeout."""
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            mock_post.side_effect = httpx.TimeoutException("Read timed out")
            
            with pytest.raises(RuntimeError, match="timed out"):
                await ollama_client.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_http_error(self, ollama_client):
        """Test generation HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            mock_post.side_effect = httpx.HTTPStatusError(
                "Not found",
                request=MagicMock(),
                response=mock_response
            )
            
            with pytest.raises(RuntimeError, match="Ollama API error"):
                await ollama_client.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_connect_error(self, ollama_client):
        """Test generation connection error - retries then fails."""
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm), \
             patch('app.llm.ollama.ollama_retry', lambda f: f):  # Disable retry for test speed
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                await ollama_client.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_unexpected_error(self, ollama_client):
        """Test generation with unexpected error."""
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'post', new_callable=AsyncMock) as mock_post, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            mock_post.side_effect = Exception("Unexpected error")
            
            with pytest.raises(RuntimeError, match="LLM generation failed"):
                await ollama_client.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_stream_success(self, ollama_client):
        """Test successful streaming generation."""
        async def mock_aiter_lines():
            yield '{"response": "Hello"}'
            yield '{"response": " World"}'
            yield '{"response": "!"}'
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__ = AsyncMock()
        
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'stream', return_value=mock_stream_context), \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            
            tokens = []
            async for token in ollama_client.generate_stream("Test prompt"):
                tokens.append(token)
        
        assert tokens == ["Hello", " World", "!"]
    
    @pytest.mark.asyncio
    async def test_generate_stream_json_decode_error(self, ollama_client):
        """Test streaming with invalid JSON."""
        async def mock_aiter_lines():
            yield '{"response": "Hello"}'
            yield 'invalid json'
            yield '{"response": "World"}'
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__ = AsyncMock()
        
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'stream', return_value=mock_stream_context), \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            
            tokens = []
            async for token in ollama_client.generate_stream("Test prompt"):
                tokens.append(token)
        
        # Should skip invalid JSON and continue
        assert tokens == ["Hello", "World"]
    
    @pytest.mark.asyncio
    async def test_generate_stream_circuit_breaker_open(self, ollama_client):
        """Test streaming when circuit breaker is open."""
        mock_breaker_cm = MagicMock()
        mock_breaker_cm.__aenter__ = AsyncMock(
            side_effect=CircuitBreakerOpen("ollama", 30.0)
        )
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                async for _ in ollama_client.generate_stream("Test prompt"):
                    pass
    
    @pytest.mark.asyncio
    async def test_generate_stream_connect_error(self, ollama_client):
        """Test streaming connection error."""
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'stream') as mock_stream, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm), \
             patch('app.llm.ollama.ollama_retry', lambda f: f):  # Disable retry for test speed
            mock_stream.side_effect = httpx.ConnectError("Connection refused")
            
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                async for _ in ollama_client.generate_stream("Test prompt"):
                    pass
    
    @pytest.mark.asyncio
    async def test_generate_stream_unexpected_error(self, ollama_client):
        """Test streaming with unexpected error."""
        mock_breaker_cm = AsyncMock()
        mock_breaker_cm.__aenter__ = AsyncMock(return_value=None)
        mock_breaker_cm.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(ollama_client.client, 'stream') as mock_stream, \
             patch('app.llm.ollama.ollama_breaker', mock_breaker_cm):
            mock_stream.side_effect = Exception("Unexpected error")
            
            with pytest.raises(RuntimeError, match="LLM streaming failed"):
                async for _ in ollama_client.generate_stream("Test prompt"):
                    pass
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}]
        }
        
        with patch.object(ollama_client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await ollama_client.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self, ollama_client):
        """Test health check when model not found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "other-model"}]
        }
        
        with patch.object(ollama_client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            
            result = await ollama_client.health_check()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, ollama_client):
        """Test health check with error."""
        with patch.object(ollama_client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            
            result = await ollama_client.health_check()
        
        assert result is False

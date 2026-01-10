"""
Tests for Health Service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.services.health_service import HealthService, CheckResult


# ============== Fixtures ==============

@pytest.fixture
def mock_settings():
    """Mock settings for health service."""
    with patch('app.services.health_service.settings') as mock:
        mock.OLLAMA_BASE_URL = "http://localhost:11434"
        mock.QDRANT_URL = "http://localhost:6333"
        mock.LLM_MODEL = "llama3.2:3b"
        mock.APP_NAME = "DANI"
        mock.ENV = "test"
        mock.DEBUG = True
        yield mock


# ============== Tests ==============

class TestCheckResult:
    """Tests for CheckResult dataclass."""
    
    def test_check_result_reachable(self):
        """Test CheckResult with reachable=True."""
        result = CheckResult(reachable=True)
        assert result.reachable is True
        assert result.error is None
        assert result.meta is None
    
    def test_check_result_with_error(self):
        """Test CheckResult with error."""
        result = CheckResult(reachable=False, error="Connection refused")
        assert result.reachable is False
        assert result.error == "Connection refused"
    
    def test_check_result_with_meta(self):
        """Test CheckResult with metadata."""
        result = CheckResult(reachable=True, meta={"models_count": 5})
        assert result.meta == {"models_count": 5}


class TestHealthService:
    """Tests for HealthService."""
    
    def test_init(self, mock_settings):
        """Test HealthService initialization."""
        service = HealthService()
        assert service.ollama_base_url == "http://localhost:11434"
        assert service.qdrant_url == "http://localhost:6333"
        assert service.llm_model == "llama3.2:3b"
    
    def test_init_strips_trailing_slash(self, mock_settings):
        """Test that URLs have trailing slashes stripped."""
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434/"
        mock_settings.QDRANT_URL = "http://localhost:6333/"
        
        service = HealthService()
        
        assert service.ollama_base_url == "http://localhost:11434"
        assert service.qdrant_url == "http://localhost:6333"
    
    @pytest.mark.asyncio
    async def test_check_ollama_success(self, mock_settings):
        """Test successful Ollama health check."""
        service = HealthService()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_ollama()
        
        assert result.reachable is True
        assert result.meta == {"models_count": 1}
    
    @pytest.mark.asyncio
    async def test_check_ollama_failure(self, mock_settings):
        """Test Ollama health check failure."""
        service = HealthService()
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_ollama()
        
        assert result.reachable is False
        assert "Connection refused" in result.error
    
    @pytest.mark.asyncio
    async def test_check_ollama_http_error(self, mock_settings):
        """Test Ollama health check with HTTP error."""
        service = HealthService()
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=MagicMock(),
                response=MagicMock(status_code=500)
            )
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_ollama()
        
        assert result.reachable is False
    
    @pytest.mark.asyncio
    async def test_check_ollama_model_available_success(self, mock_settings):
        """Test successful model availability check."""
        service = HealthService()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "nomic-embed-text"}
            ]
        }
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_ollama_model_available()
        
        assert result.reachable is True
        assert result.meta == {"configured": "llama3.2:3b"}
    
    @pytest.mark.asyncio
    async def test_check_ollama_model_not_found(self, mock_settings):
        """Test model not found."""
        service = HealthService()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "other-model"},
            ]
        }
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_ollama_model_available()
        
        assert result.reachable is False
        assert "not found" in result.error
        assert result.meta == {"configured": "llama3.2:3b"}
    
    @pytest.mark.asyncio
    async def test_check_ollama_model_connection_error(self, mock_settings):
        """Test model check with connection error."""
        service = HealthService()
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_ollama_model_available()
        
        assert result.reachable is False
        assert result.meta == {"configured": "llama3.2:3b"}
    
    @pytest.mark.asyncio
    async def test_check_qdrant_success(self, mock_settings):
        """Test successful Qdrant health check."""
        service = HealthService()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"status": "ok"}
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_qdrant()
        
        assert result.reachable is True
        assert result.meta == {"status": "ok"}
    
    @pytest.mark.asyncio
    async def test_check_qdrant_non_json_response(self, mock_settings):
        """Test Qdrant health check with non-JSON response."""
        service = HealthService()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "OK"
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_qdrant()
        
        assert result.reachable is True
        assert result.meta == {"raw": "OK"}
    
    @pytest.mark.asyncio
    async def test_check_qdrant_failure(self, mock_settings):
        """Test Qdrant health check failure."""
        service = HealthService()
        
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            MockClient.return_value.__aenter__.return_value = mock_client
            
            result = await service.check_qdrant()
        
        assert result.reachable is False
        assert "Connection refused" in result.error
    
    @pytest.mark.asyncio
    async def test_full_health_all_ok(self, mock_settings):
        """Test full health check when all services are OK."""
        service = HealthService()
        
        # Mock all checks
        with patch.object(service, 'check_ollama') as mock_ollama, \
             patch.object(service, 'check_qdrant') as mock_qdrant, \
             patch.object(service, 'check_ollama_model_available') as mock_model:
            
            mock_ollama.return_value = CheckResult(reachable=True, meta={"models_count": 2})
            mock_qdrant.return_value = CheckResult(reachable=True, meta={"status": "ok"})
            mock_model.return_value = CheckResult(reachable=True, meta={"configured": "llama3.2:3b"})
            
            result = await service.full_health()
        
        assert result["status"] == "ok"
        assert result["app"]["name"] == "DANI"
    
    @pytest.mark.asyncio
    async def test_full_health_ollama_down(self, mock_settings):
        """Test full health check when Ollama is down."""
        service = HealthService()
        
        with patch.object(service, 'check_ollama') as mock_ollama, \
             patch.object(service, 'check_qdrant') as mock_qdrant, \
             patch.object(service, 'check_ollama_model_available') as mock_model:
            
            mock_ollama.return_value = CheckResult(reachable=False, error="Connection refused")
            mock_qdrant.return_value = CheckResult(reachable=True, meta={"status": "ok"})
            # Model check should be skipped when Ollama is down
            
            result = await service.full_health()
        
        assert result["status"] == "degraded"
    
    @pytest.mark.asyncio
    async def test_full_health_qdrant_down(self, mock_settings):
        """Test full health check when Qdrant is down."""
        service = HealthService()
        
        with patch.object(service, 'check_ollama') as mock_ollama, \
             patch.object(service, 'check_qdrant') as mock_qdrant, \
             patch.object(service, 'check_ollama_model_available') as mock_model:
            
            mock_ollama.return_value = CheckResult(reachable=True, meta={"models_count": 2})
            mock_qdrant.return_value = CheckResult(reachable=False, error="Connection refused")
            mock_model.return_value = CheckResult(reachable=True, meta={"configured": "llama3.2:3b"})
            
            result = await service.full_health()
        
        assert result["status"] == "degraded"
    
    @pytest.mark.asyncio
    async def test_full_health_model_missing(self, mock_settings):
        """Test full health check when model is missing."""
        service = HealthService()
        
        with patch.object(service, 'check_ollama') as mock_ollama, \
             patch.object(service, 'check_qdrant') as mock_qdrant, \
             patch.object(service, 'check_ollama_model_available') as mock_model:
            
            mock_ollama.return_value = CheckResult(reachable=True, meta={"models_count": 2})
            mock_qdrant.return_value = CheckResult(reachable=True, meta={"status": "ok"})
            mock_model.return_value = CheckResult(reachable=False, error="Model not found")
            
            result = await service.full_health()
        
        assert result["status"] == "degraded"

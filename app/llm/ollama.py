from __future__ import annotations

import json
import logging
import httpx
from typing import AsyncIterator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.core.config import settings
from app.core.circuit_breaker import ollama_breaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)

# Retry decorator for transient failures (connection issues, timeouts)
# Does NOT retry on HTTP 4xx errors (bad request, model not found)
ollama_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout, ConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class OllamaClient:
    """
    Ollama text generation client with streaming support.
    Dynamically reads settings on each request to support instant local/cloud switching.
    """

    def __init__(self) -> None:
        # Extended timeout for LLM generation (can take several minutes)
        self.timeout = httpx.Timeout(
            connect=5.0,
            read=600.0,  # 10 minutes for generation
            write=10.0,
            pool=10.0,
        )
        
        # Connection pooling for performance
        # NOTE: Headers are NOT set here - they are set per-request to support dynamic API key changes
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
    
    def _get_headers(self) -> dict:
        """Get current authorization headers based on live settings.
        This is called on each request to support instant local/cloud switching."""
        headers = {}
        if settings.OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"
        return headers
    
    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self.client.aclose()
    
    async def __aenter__(self) -> "OllamaClient":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    @ollama_retry
    async def generate(self, prompt: str, system: str = None, stream: bool = False, options: dict = None) -> str:
        """Generate text with retry logic and circuit breaker for resilience.
        
        Settings are read on each request to support instant local/cloud switching:
        - OLLAMA_BASE_URL: Computed from OLLAMA_ENV setting
        - LLM_MODEL: May vary by deployment
        - OLLAMA_API_KEY: Added to headers if present
        """
        logger.info(f"Sending prompt to Ollama ({len(prompt)} chars)")
        
        # Read settings dynamically on each request (not cached from __init__)
        base_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        model = settings.LLM_MODEL

        # Default options
        request_options = {
            "num_ctx": settings.LLM_NUM_CTX,
            "num_predict": settings.LLM_NUM_PREDICT,
            "temperature": settings.LLM_TEMPERATURE,
            "num_thread": settings.LLM_NUM_THREAD,
        }
        
        # Merge provided options if any
        if options:
            request_options.update(options)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Non-streaming for simplicity
            "keep_alive": "24h",  # Keep model loaded for 24 hours
            "options": request_options
        }
        
        if system:
            payload["system"] = system

        try:
            # Circuit breaker protects against repeated failures
            async with ollama_breaker:
                response = await self.client.post(base_url, json=payload, headers=self._get_headers())
                response.raise_for_status()
                
                data = response.json()
                logger.info("Received response from Ollama")
                
                text = data.get("response", "").strip()
                
                if not text:
                    logger.warning("Empty response from Ollama")
                    raise ValueError("Ollama returned empty response")
                
                return text
        
        except CircuitBreakerOpen as e:
            logger.warning(f"Ollama circuit breaker open: {e}")
            raise RuntimeError(
                "The AI service is temporarily unavailable due to repeated failures. "
                f"Please try again in {e.recovery_time:.0f} seconds."
            ) from e
            
        except httpx.TimeoutException as e:
            logger.error(f"Ollama request timed out after {self.timeout.read}s")
            raise RuntimeError(
                f"LLM generation timed out. The response took longer than {self.timeout.read} seconds. "
                "Try a shorter query or check if Ollama is overloaded."
            ) from e
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(
                f"Ollama API error ({e.response.status_code}): {e.response.text}"
            ) from e
            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}")
            raise RuntimeError(
                f"Cannot connect to Ollama. Is it running at {settings.OLLAMA_BASE_URL}?"
            ) from e
            
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e
        
        # This should never be reached as all code paths either return or raise
        raise RuntimeError("Unexpected code path in generate()")  # pragma: no cover
    
    @ollama_retry
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens with retry logic and circuit breaker for resilience.
        
        Settings are read on each request to support instant local/cloud switching:
        - OLLAMA_BASE_URL: Computed from OLLAMA_ENV setting
        - LLM_MODEL: May vary by deployment
        - OLLAMA_API_KEY: Added to headers if present
        """
        logger.info(f"Sending streaming prompt to Ollama ({len(prompt)} chars)")
        
        # Read settings dynamically on each request (not cached from __init__)
        base_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        model = settings.LLM_MODEL

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "24h",  # Keep model loaded for 24 hours
            "options": {
                "num_ctx": settings.LLM_NUM_CTX,
                "num_predict": settings.LLM_NUM_PREDICT,
                "temperature": settings.LLM_TEMPERATURE,
                "num_thread": settings.LLM_NUM_THREAD,
            }
        }

        try:
            # Check circuit breaker before streaming
            async with ollama_breaker:
                async with self.client.stream("POST", base_url, json=payload, headers=self._get_headers()) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if token := data.get("response"):
                                    yield token
                            except json.JSONDecodeError:
                                continue
        
        except CircuitBreakerOpen as e:
            logger.warning(f"Ollama circuit breaker open during streaming: {e}")
            raise RuntimeError(
                "The AI service is temporarily unavailable due to repeated failures. "
                f"Please try again in {e.recovery_time:.0f} seconds."
            ) from e
                            
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}")
            raise RuntimeError(
                f"Cannot connect to Ollama. Is it running at {settings.OLLAMA_BASE_URL}?"
            ) from e
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise RuntimeError(f"LLM streaming failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            # Read settings dynamically (not cached)
            url = f"{settings.OLLAMA_BASE_URL}/api/tags"
            model = settings.LLM_MODEL
            
            response = await self.client.get(url, headers=self._get_headers())
            response.raise_for_status()
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            if model not in models:
                logger.warning(f"Model {model} not found in Ollama")
                return False
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

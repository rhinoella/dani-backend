from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


@dataclass
class CheckResult:
    reachable: bool
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class HealthService:
    def __init__(self) -> None:
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.qdrant_url = settings.QDRANT_URL.rstrip("/")
        self.llm_model = settings.LLM_MODEL

        # Keep timeouts tight for health checks
        self.timeout = httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=3.0)

    async def check_ollama(self) -> CheckResult:
        """
        Checks whether Ollama is reachable.
        """
        url = f"{self.ollama_base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
                return CheckResult(reachable=True, meta={"models_count": len(data.get("models", []))})
        except Exception as e:
            return CheckResult(reachable=False, error=str(e))

    async def check_ollama_model_available(self) -> CheckResult:
        """
        Checks whether the configured LLM model exists in Ollama.
        """
        url = f"{self.ollama_base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                exists = self.llm_model in models

                if not exists:
                    # Show a helpful hint without dumping huge payloads
                    sample = models[:10]
                    return CheckResult(
                        reachable=False,
                        error=f"Configured model '{self.llm_model}' not found in Ollama. Sample available: {sample}",
                        meta={"configured": self.llm_model},
                    )

                return CheckResult(reachable=True, meta={"configured": self.llm_model})
        except Exception as e:
            return CheckResult(reachable=False, error=str(e), meta={"configured": self.llm_model})

    async def check_qdrant(self) -> CheckResult:
        """
        Checks whether Qdrant is reachable.
        """
        # /healthz is lightweight
        url = f"{self.qdrant_url}/healthz"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(url)
                r.raise_for_status()
                # Qdrant returns JSON for /healthz in most versions
                data = r.json() if "application/json" in r.headers.get("content-type", "") else {"raw": r.text}
                return CheckResult(reachable=True, meta=data)
        except Exception as e:
            return CheckResult(reachable=False, error=str(e))

    async def full_health(self) -> Dict[str, Any]:
        ollama = await self.check_ollama()
        qdrant = await self.check_qdrant()

        # Only check model existence if Ollama is reachable
        model_check = await self.check_ollama_model_available() if ollama.reachable else CheckResult(
            reachable=False, error="Ollama not reachable; model check skipped."
        )

        overall_ok = ollama.reachable and qdrant.reachable and model_check.reachable

        return {
            "status": "ok" if overall_ok else "degraded",
            "app": {
                "name": settings.APP_NAME,
                "env": settings.ENV,
                "debug": settings.DEBUG,
            },
            "llm": {
                "provider": "ollama",
                "base_url": settings.OLLAMA_BASE_URL,
                "model": settings.LLM_MODEL,
                "reachable": ollama.reachable,
                "error": ollama.error,
                "meta": ollama.meta,
                "model_available": model_check.reachable,
                "model_error": model_check.error,
            },
            "vector_db": {
                "provider": "qdrant",
                "url": settings.QDRANT_URL,
                "reachable": qdrant.reachable,
                "error": qdrant.error,
                "meta": qdrant.meta,
            },
        }

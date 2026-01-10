"""
Performance and Integration Tests for Ollama Cloud

Tests cloud Ollama connectivity, response quality, and performance with:
- Various prompt sizes (small to huge)
- RAG integration with real data
- Response time tracking
- Quality scoring
- Comparison with local Ollama

Run with: pytest tests/performance/test_ollama_cloud_performance.py -v --performance
"""

import pytest
import time
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from unittest.mock import patch, AsyncMock

from app.core.config import settings
from app.llm.ollama import OllamaClient
from app.embeddings.client import OllamaEmbeddingClient
from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService


pytestmark = pytest.mark.performance


class PerformanceMetrics:
    """Track performance metrics for responses."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.prompt_length = 0
        self.response_length = 0
        self.tokens_per_second = 0
        self.quality_score = 0.0
        self.error = None
    
    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    def record(self) -> Dict[str, Any]:
        return {
            "test": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "duration_ms": self.duration_ms,
            "tokens_per_second": self.tokens_per_second,
            "quality_score": self.quality_score,
            "error": self.error,
            "success": self.error is None,
        }
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.error is None else "❌ FAIL"
        return (
            f"{status} | {self.test_name}\n"
            f"    Duration: {self.duration_ms:.0f}ms\n"
            f"    Prompt: {self.prompt_length} chars | Response: {self.response_length} chars\n"
            f"    Speed: {self.tokens_per_second:.1f} tokens/sec\n"
            f"    Quality: {self.quality_score:.2f}/1.0"
        )


class TestOllamaCloudConnectivity:
    """Test basic cloud Ollama connectivity."""
    
    @pytest.mark.asyncio
    async def test_cloud_ollama_reachable(self):
        """Test that cloud Ollama API is reachable."""
        metrics = PerformanceMetrics("cloud_ollama_reachable")
        
        try:
            # Only run if cloud is configured
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured. Set OLLAMA_ENV=cloud and OLLAMA_API_KEY")
            
            metrics.start_time = time.time()
            
            client = OllamaClient()
            result = await client.health_check()
            
            metrics.end_time = time.time()
            
            assert result is True, "Cloud Ollama health check failed"
            print(f"\n{metrics}")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_cloud_ollama_model_available(self):
        """Test that configured LLM model is available on cloud."""
        metrics = PerformanceMetrics("cloud_model_available")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            metrics.start_time = time.time()
            
            client = OllamaClient()
            # Health check validates model exists
            result = await client.health_check()
            
            metrics.end_time = time.time()
            assert result is True
            
            print(f"\n{metrics}")
            print(f"   Model: {settings.LLM_MODEL}")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_cloud_ollama_auth_header(self):
        """Test that Authorization header is sent correctly."""
        metrics = PerformanceMetrics("cloud_auth_header")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            client = OllamaClient()
            
            # Verify _get_headers returns auth header
            headers = client._get_headers()
            
            assert "Authorization" in headers, "Authorization header missing"
            assert headers["Authorization"].startswith("Bearer "), "Invalid Authorization format"
            assert settings.OLLAMA_API_KEY in headers["Authorization"]
            
            metrics.end_time = time.time()
            print(f"\n{metrics}")
            print(f"   ✅ Authorization header properly configured")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise


class TestOllamaCloudPrompts:
    """Test cloud Ollama with various prompt sizes."""
    
    @pytest.mark.asyncio
    async def test_small_prompt(self):
        """Test small prompt (~50 chars)."""
        metrics = PerformanceMetrics("small_prompt")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            prompt = "What is machine learning? Answer in 1-2 sentences."
            metrics.prompt_length = len(prompt)
            
            metrics.start_time = time.time()
            
            client = OllamaClient()
            response = await client.generate(prompt)
            
            metrics.end_time = time.time()
            metrics.response_length = len(response)
            metrics.tokens_per_second = metrics.response_length / max(metrics.duration_ms / 1000, 0.1)
            
            # Simple quality check
            metrics.quality_score = min(len(response) / 100, 1.0)
            
            assert response, "Empty response received"
            print(f"\n{metrics}")
            print(f"   Response: {response[:100]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_medium_prompt(self):
        """Test medium prompt (~500 chars) - typical user query with context."""
        metrics = PerformanceMetrics("medium_prompt")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            prompt = """
            Based on our recent Q4 planning meetings, we discussed:
            - Budget allocation for 2025: $5M total, split across 3 teams
            - New hiring: 5 engineers, 2 product managers, 1 designer
            - Technology stack upgrade: Python 3.13, Django 5.0, PostgreSQL 16
            - Cloud migration: Moving from on-premise to AWS
            - Security audit planned for February
            - Customer retention program to be launched March 1st
            
            What should be our top 3 priorities for January based on this?
            """
            metrics.prompt_length = len(prompt)
            
            metrics.start_time = time.time()
            
            client = OllamaClient()
            response = await client.generate(prompt)
            
            metrics.end_time = time.time()
            metrics.response_length = len(response)
            metrics.tokens_per_second = metrics.response_length / max(metrics.duration_ms / 1000, 0.1)
            
            # Quality: check for structured response
            has_bullets = "1." in response or "-" in response
            has_explanation = len(response) > 200
            metrics.quality_score = 0.5 if has_bullets else 0.3
            metrics.quality_score += 0.5 if has_explanation else 0.0
            
            assert response, "Empty response received"
            print(f"\n{metrics}")
            print(f"   Response: {response[:150]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_huge_prompt(self):
        """Test huge prompt (~3000 chars) - like RAG with lots of context."""
        metrics = PerformanceMetrics("huge_prompt")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            # Simulate RAG context with multiple transcript chunks
            context_chunks = [
                "Meeting 1 (Jan 5): Discussed Q1 roadmap, approved budget increase to $2.5M",
                "Meeting 2 (Jan 6): Team standup, 3 engineers on-boarded, productivity up 15%",
                "Meeting 3 (Jan 7): Customer feedback session, 5 feature requests documented",
                "Meeting 4 (Jan 8): Board meeting, approved expansion to 2 new markets",
                "Meeting 5 (Jan 9): Engineering review, identified 12 technical debt items",
            ] * 6  # Repeat to make it huge
            
            rag_context = "\n".join([f"• {chunk}" for chunk in context_chunks])
            
            prompt = f"""
            Context from recent meetings:
            {rag_context}
            
            Based on ALL of this context, provide a comprehensive summary of:
            1. Key decisions made
            2. Action items assigned
            3. Timeline for next steps
            4. Budget implications
            5. Team impact assessment
            
            Be thorough and cite specific meetings when possible.
            """
            
            metrics.prompt_length = len(prompt)
            
            print(f"\n   Sending huge prompt: {metrics.prompt_length} chars")
            metrics.start_time = time.time()
            
            client = OllamaClient()
            response = await client.generate(prompt)
            
            metrics.end_time = time.time()
            metrics.response_length = len(response)
            metrics.tokens_per_second = metrics.response_length / max(metrics.duration_ms / 1000, 0.1)
            
            # Quality: check for comprehensive response
            has_all_sections = all(f"{i}." in response for i in range(1, 6))
            mentions_meetings = "meeting" in response.lower()
            metrics.quality_score = 0.6 if has_all_sections else 0.3
            metrics.quality_score += 0.4 if mentions_meetings else 0.0
            
            assert response, "Empty response received"
            assert len(response) > 500, "Response too short for huge prompt"
            
            print(f"{metrics}")
            print(f"   Response: {response[:200]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming response with medium prompt."""
        metrics = PerformanceMetrics("streaming_response")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            prompt = "List 10 best practices for cloud architecture. Be detailed."
            metrics.prompt_length = len(prompt)
            
            metrics.start_time = time.time()
            
            client = OllamaClient()
            full_response = ""
            token_count = 0
            
            async for token in client.generate_stream(prompt):
                full_response += token
                token_count += 1
            
            metrics.end_time = time.time()
            metrics.response_length = len(full_response)
            metrics.tokens_per_second = token_count / max(metrics.duration_ms / 1000, 0.1)
            
            # Quality check
            has_practices = all(str(i) in full_response for i in range(1, 6))
            metrics.quality_score = 0.7 if has_practices else 0.5
            
            assert full_response, "Empty streaming response"
            assert token_count > 50, "Too few tokens streamed"
            
            print(f"\n{metrics}")
            print(f"   Tokens streamed: {token_count}")
            print(f"   Response: {full_response[:150]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise


class TestOllamaCloudEmbeddings:
    """Test cloud Ollama embedding performance."""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test single embedding generation."""
        metrics = PerformanceMetrics("embedding_generation")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            text = "What are the key decisions from the Q4 planning meetings?"
            metrics.prompt_length = len(text)
            
            metrics.start_time = time.time()
            
            embedder = OllamaEmbeddingClient()
            embedding = await embedder.embed_query(text)
            
            metrics.end_time = time.time()
            metrics.response_length = len(embedding)
            
            assert embedding, "Empty embedding"
            assert len(embedding) == 768, f"Expected 768-dim embedding, got {len(embedding)}"
            
            print(f"\n{metrics}")
            print(f"   Embedding dimension: {len(embedding)}")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_batch_embeddings(self):
        """Test batch embedding generation."""
        metrics = PerformanceMetrics("batch_embeddings")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            texts = [
                "Meeting summary: Q1 planning approved",
                "Action items: Budget review by Friday",
                "Decisions: Hired 3 new engineers",
                "Timeline: Launch marketing campaign March 1st",
                "Team impact: 15% productivity increase",
            ]
            
            metrics.prompt_length = sum(len(t) for t in texts)
            
            metrics.start_time = time.time()
            
            embedder = OllamaEmbeddingClient()
            embeddings = await embedder.embed_documents(texts, batch_size=3)
            
            metrics.end_time = time.time()
            metrics.response_length = len(embeddings)
            
            assert len(embeddings) == len(texts), "Embedding count mismatch"
            assert all(len(e) == 768 for e in embeddings), "Invalid embedding dimensions"
            
            metrics.tokens_per_second = len(texts) / max(metrics.duration_ms / 1000, 0.1)
            
            print(f"\n{metrics}")
            print(f"   Texts embedded: {len(texts)}")
            print(f"   Batch size: 3")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise


class TestOllamaCloudRAGIntegration:
    """Test cloud Ollama with RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_rag_with_cloud_retrieval(self):
        """Test RAG retrieval with cloud Ollama."""
        metrics = PerformanceMetrics("rag_cloud_retrieval")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            query = "What are the top priorities for Q1?"
            metrics.prompt_length = len(query)
            
            metrics.start_time = time.time()
            
            retrieval = RetrievalService()
            results = await retrieval.retrieve(query, top_k=5)
            
            metrics.end_time = time.time()
            metrics.response_length = sum(len(r.get("text", "")) for r in results)
            
            # Quality: number of relevant results retrieved
            metrics.quality_score = min(len(results) / 5, 1.0)
            
            print(f"\n{metrics}")
            print(f"   Results retrieved: {len(results)}")
            if results:
                print(f"   First result: {results[0].get('text', '')[:100]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            # Don't fail on this - might not have data indexed
            if "Qdrant" in str(e) or "connect" in str(e).lower():
                pytest.skip(f"Qdrant not available: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_rag_with_cloud_chat(self):
        """Test full RAG chat pipeline with cloud Ollama."""
        metrics = PerformanceMetrics("rag_cloud_chat")
        
        try:
            if settings.OLLAMA_ENV != "cloud" or not settings.OLLAMA_API_KEY:
                pytest.skip("Cloud Ollama not configured")
            
            query = "Summarize the key decisions from recent meetings"
            metrics.prompt_length = len(query)
            
            metrics.start_time = time.time()
            
            chat = ChatService()
            response = await chat.answer(query, verbose=True)
            
            metrics.end_time = time.time()
            
            answer = response.get("answer", "")
            metrics.response_length = len(answer)
            metrics.tokens_per_second = metrics.response_length / max(metrics.duration_ms / 1000, 0.1)
            
            # Quality: check for coherent answer
            metrics.quality_score = min(len(answer) / 200, 1.0)
            if response.get("confidence", {}).get("retrieval_quality", 0) > 0.5:
                metrics.quality_score = min(metrics.quality_score + 0.2, 1.0)
            
            assert answer, "Empty chat response"
            
            print(f"\n{metrics}")
            print(f"   Answer: {answer[:150]}...")
            print(f"   Sources: {len(response.get('sources', []))} documents")
            print(f"   Confidence: {response.get('confidence', {})}")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            if "Qdrant" in str(e) or "connect" in str(e).lower():
                pytest.skip(f"Qdrant not available: {e}")
            raise


class TestOllamaCloudComparison:
    """Compare cloud vs local Ollama performance."""
    
    @pytest.mark.asyncio
    async def test_cloud_vs_local_latency(self):
        """Compare latency between cloud and local."""
        results = {
            "cloud": {"duration_ms": 0, "error": None},
            "local": {"duration_ms": 0, "error": None},
        }
        
        prompt = "Explain cloud vs local computing in 2 sentences."
        
        # Test cloud
        if settings.OLLAMA_ENV == "cloud" and settings.OLLAMA_API_KEY:
            try:
                metrics = PerformanceMetrics("cloud_latency")
                metrics.start_time = time.time()
                
                client = OllamaClient()
                response = await client.generate(prompt)
                
                metrics.end_time = time.time()
                results["cloud"]["duration_ms"] = metrics.duration_ms
                
                print(f"\n☁️  Cloud Response: {metrics.duration_ms:.0f}ms")
                
            except Exception as e:
                results["cloud"]["error"] = str(e)
                print(f"\n❌ Cloud Error: {e}")
        else:
            print("\n⏭️  Cloud Ollama not configured")
        
        # Test local (if different from cloud)
        if settings.OLLAMA_ENV != "cloud":
            try:
                metrics = PerformanceMetrics("local_latency")
                metrics.start_time = time.time()
                
                client = OllamaClient()
                response = await client.generate(prompt)
                
                metrics.end_time = time.time()
                results["local"]["duration_ms"] = metrics.duration_ms
                
                print(f"🖥️  Local Response: {metrics.duration_ms:.0f}ms")
                
            except Exception as e:
                results["local"]["error"] = str(e)
                print(f"\n❌ Local Error: {e}")
        
        # Comparison
        if results["cloud"]["duration_ms"] > 0 and results["local"]["duration_ms"] > 0:
            ratio = results["cloud"]["duration_ms"] / results["local"]["duration_ms"]
            print(f"\n📊 Cloud is {ratio:.1f}x {'faster' if ratio < 1 else 'slower'} than local")


@pytest.fixture
def performance_report(tmp_path):
    """Fixture to generate performance report."""
    report_file = tmp_path / "performance_report.json"
    metrics_list = []
    
    yield metrics_list
    
    # Write report
    if metrics_list:
        with open(report_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "ollama_env": settings.OLLAMA_ENV,
                "ollama_base_url": settings.OLLAMA_BASE_URL,
                "model": settings.LLM_MODEL,
                "metrics": metrics_list,
            }, f, indent=2)
        print(f"\n📊 Performance report: {report_file}")

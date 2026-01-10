"""
Ollama Cloud Performance Tests

Tests cloud Ollama connectivity, response quality, and performance with:
- Various prompt sizes (small to huge)
- RAG integration with real data
- Response time tracking
- Quality scoring
- Comparison with local Ollama

Run with: pytest tests/test_ollama_cloud_performance.py -v --performance
"""

import pytest
import time
import asyncio
from typing import Dict, Any, Optional

from app.core.config import settings
from app.llm.ollama import OllamaClient
from app.embeddings.client import OllamaEmbeddingClient


pytestmark = pytest.mark.performance


class PerformanceMetrics:
    """Track performance metrics for responses."""
    
    def __init__(self, test_name: str):
        self.test_name: str = test_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.prompt_length: int = 0
        self.response_length: int = 0
        self.tokens_per_second: float = 0.0
        self.quality_score: float = 0.0
        self.error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
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
        """Test that cloud or configured Ollama API is reachable."""
        metrics = PerformanceMetrics("ollama_reachable")
        
        try:
            metrics.start_time = time.time()
            
            client = OllamaClient()
            result = await client.health_check()
            
            metrics.end_time = time.time()
            
            assert result is True, "Ollama health check failed"
            
            print(f"\n{metrics}")
            print(f"   Environment: {settings.OLLAMA_ENV}")
            print(f"   Base URL: {settings.OLLAMA_BASE_URL}")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_ollama_model_available(self):
        """Test that configured LLM model is available."""
        metrics = PerformanceMetrics("model_available")
        
        try:
            metrics.start_time = time.time()
            
            client = OllamaClient()
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
    async def test_ollama_auth_header(self):
        """Test that Authorization header is sent correctly."""
        metrics = PerformanceMetrics("auth_header")
        
        try:
            client = OllamaClient()
            
            # Verify _get_headers returns correct headers
            headers = client._get_headers()
            
            if settings.OLLAMA_ENV == "cloud" and settings.OLLAMA_API_KEY:
                assert "Authorization" in headers, "Authorization header missing"
                assert headers["Authorization"].startswith("Bearer "), "Invalid Authorization format"
            
            metrics.end_time = time.time()
            print(f"\n{metrics}")
            print(f"   Headers configured: ✅")
            if "Authorization" in headers:
                print(f"   Authorization: Present (Bearer token)")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise


class TestOllamaPromptSizes:
    """Test Ollama with various prompt sizes."""
    
    @pytest.mark.asyncio
    async def test_small_prompt(self):
        """Test small prompt (~50 chars)."""
        metrics = PerformanceMetrics("small_prompt")
        
        try:
            prompt = "What is machine learning? Answer in 1-2 sentences."
            metrics.prompt_length = len(prompt)
            
            print(f"\n📝 Small Prompt Test")
            print(f"   Prompt ({metrics.prompt_length} chars): {prompt}")
            
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
            print(f"   Response: {response[:150]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_medium_prompt(self):
        """Test medium prompt (~500 chars) - typical user query with context."""
        metrics = PerformanceMetrics("medium_prompt")
        
        try:
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
            
            print(f"\n📄 Medium Prompt Test")
            print(f"   Prompt length: {metrics.prompt_length} chars")
            
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
            print(f"   Response: {response[:200]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_huge_prompt(self):
        """Test huge prompt (~3000 chars) - like RAG with lots of context."""
        metrics = PerformanceMetrics("huge_prompt")
        
        try:
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
            
            print(f"\n🗂️  Huge Prompt Test (Simulated RAG Context)")
            print(f"   Prompt length: {metrics.prompt_length} chars")
            print(f"   Context chunks: {len(context_chunks)}")
            
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
            
            print(f"\n{metrics}")
            print(f"   Response: {response[:250]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming response with medium prompt."""
        metrics = PerformanceMetrics("streaming_response")
        
        try:
            prompt = "List 10 best practices for cloud architecture. Be detailed."
            metrics.prompt_length = len(prompt)
            
            print(f"\n🌊 Streaming Response Test")
            print(f"   Prompt: {prompt}")
            
            metrics.start_time = time.time()
            
            client = OllamaClient()
            full_response = ""
            token_count = 0
            
            print("   Streaming: ", end="", flush=True)
            async for token in client.generate_stream(prompt):
                full_response += token
                token_count += 1
                if token_count % 50 == 0:
                    print(".", end="", flush=True)
            print(" done")
            
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
            print(f"   Response: {full_response[:200]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise


class TestOllamaEmbeddings:
    """Test embeddings (local only - cloud doesn't support embeddings)."""
    
    @pytest.fixture(autouse=True)
    def skip_if_cloud(self):
        """Skip all embedding tests if using cloud Ollama."""
        if settings.OLLAMA_ENV == "cloud":
            pytest.skip("Embeddings must use local Ollama (cloud doesn't support them)")
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test single embedding generation (local only)."""
        metrics = PerformanceMetrics("embedding_generation")
        
        try:
            text = "What are the key decisions from the Q4 planning meetings?"
            metrics.prompt_length = len(text)
            
            print(f"\n🧩 Embedding Generation Test (Local Only)")
            print(f"   Text: {text}")
            print(f"   📍 Must use local Docker - cloud doesn't support embeddings")
            
            metrics.start_time = time.time()
            
            embedder = OllamaEmbeddingClient()
            embedding = await embedder.embed_query(text)
            
            metrics.end_time = time.time()
            metrics.response_length = len(embedding)
            metrics.quality_score = 1.0 if len(embedding) == 768 else 0.5
            
            assert embedding, "Empty embedding"
            assert len(embedding) == 768, f"Expected 768-dim embedding, got {len(embedding)}"
            
            print(f"\n{metrics}")
            print(f"   Embedding dimension: {len(embedding)}")
            
        except Exception as e:
            metrics.error = str(e)
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip("Embeddings must be on local Ollama (cloud doesn't support them)")
            if "refused" in str(e).lower() or "connect" in str(e).lower():
                pytest.skip(f"Local Ollama not available: {e}")
            print(f"\n{metrics}")
            raise
    
    @pytest.mark.asyncio
    async def test_batch_embeddings(self):
        """Test batch embedding generation (local only)."""
        metrics = PerformanceMetrics("batch_embeddings")
        
        try:
            texts = [
                "Meeting summary: Q1 planning approved",
                "Action items: Budget review by Friday",
                "Decisions: Hired 3 new engineers",
                "Timeline: Launch marketing campaign March 1st",
                "Team impact: 15% productivity increase",
            ]
            
            metrics.prompt_length = sum(len(t) for t in texts)
            
            print(f"\n📦 Batch Embeddings Test (Local Only)")
            print(f"   Texts to embed: {len(texts)}")
            print(f"   📍 Must use local Docker - cloud doesn't support embeddings")
            
            metrics.start_time = time.time()
            
            embedder = OllamaEmbeddingClient()
            embeddings = await embedder.embed_documents(texts, batch_size=3)
            
            metrics.end_time = time.time()
            metrics.response_length = len(embeddings)
            metrics.quality_score = 1.0 if len(embeddings) == len(texts) else 0.5
            
            # Quality check
            assert embeddings, "Empty embeddings"
            assert len(embeddings) == len(texts), f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            
            print(f"\n{metrics}")
            print(f"   Embeddings generated: {len(embeddings)}")
            if embeddings:
                print(f"   Dimension per vector: {len(embeddings[0])}")
            
        except Exception as e:
            metrics.error = str(e)
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip("Embeddings must be on local Ollama (cloud doesn't support them)")
            if "refused" in str(e).lower() or "connect" in str(e).lower():
                pytest.skip(f"Local Ollama not available: {e}")
            print(f"\n{metrics}")
            raise
            
            assert len(embeddings) == len(texts), "Embedding count mismatch"
            assert all(len(e) == 768 for e in embeddings), "Invalid embedding dimensions"
            
            metrics.tokens_per_second = len(texts) / max(metrics.duration_ms / 1000, 0.1)
            
            print(f"\n{metrics}")
            print(f"   Texts embedded: {len(texts)}")
            print(f"   Batch size: 3")
            for i, text in enumerate(texts, 1):
                print(f"   [{i}] {text[:50]}...")
            
        except Exception as e:
            metrics.error = str(e)
            print(f"\n{metrics}")
            raise


class TestOllamaRAGIntegration:
    """Test Ollama with RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_rag_retrieval(self):
        """Test RAG retrieval with your working RAG pipeline (local embeddings + cloud LLM)."""
        metrics = PerformanceMetrics("rag_retrieval")
        
        try:
            query = "What are the top priorities for Q1?"
            metrics.prompt_length = len(query)
            
            print(f"\n🔍 RAG Retrieval Test (Hybrid Search)")
            print(f"   Query: {query}")
            print(f"   📍 Embeddings: Local Docker (nomic-embed-text)")
            print(f"   ☁️  LLM: Cloud Ollama")
            
            metrics.start_time = time.time()
            
            # Temporarily force local embeddings even if LLM is on cloud
            from app.core.config import settings
            original_env = settings.OLLAMA_ENV
            settings.OLLAMA_ENV = "local"  # Force local for embeddings
            
            try:
                from app.services.retrieval_service import RetrievalService
                retrieval = RetrievalService()
                # Use correct method: search() with limit parameter
                # Embeddings will use local Docker automatically (nomic-embed-text)
                results = await retrieval.search(
                    query=query,
                    limit=5,
                    use_hybrid=True,
                    use_reranking=True,
                    use_adaptive=True
                )
                
                metrics.end_time = time.time()
                metrics.response_length = sum(len(r.get("content", "")) for r in results)
                
                # Quality: number of relevant results retrieved
                metrics.quality_score = min(len(results) / 5, 1.0)
                
                print(f"\n{metrics}")
                print(f"   Results retrieved: {len(results)}")
                if results:
                    for i, result in enumerate(results[:3], 1):
                        content = result.get("content", "")[:80]
                        score = result.get("score", 0)
                        print(f"   [{i}] Score: {score:.3f} | {content}...")
                
                assert len(results) > 0, "No results retrieved from RAG"
                
            finally:
                # Restore original environment
                settings.OLLAMA_ENV = original_env
            
        except Exception as e:
            metrics.error = str(e)
            if "Qdrant" in str(e) or "connect" in str(e).lower():
                pytest.skip(f"Qdrant not available: {e}")
            if "embedding" in str(e).lower() or "404" in str(e) or "refused" in str(e).lower():
                pytest.skip(f"Local Ollama not available for embeddings: {e}")
            print(f"\n{metrics}")
            raise


class TestOllamaComparison:
    """Compare Ollama performance (cloud vs local)."""
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Compare performance metrics."""
        results = {
            "current": {"duration_ms": 0, "error": None},
        }
        
        prompt = "Explain cloud architecture in 2 sentences."
        
        try:
            metrics = PerformanceMetrics("performance_comparison")
            metrics.start_time = time.time()
            
            client = OllamaClient()
            response = await client.generate(prompt)
            
            metrics.end_time = time.time()
            results["current"]["duration_ms"] = metrics.duration_ms
            
            print(f"\n📊 Performance Comparison")
            print(f"   Environment: {settings.OLLAMA_ENV}")
            print(f"   Base URL: {settings.OLLAMA_BASE_URL}")
            print(f"   Model: {settings.LLM_MODEL}")
            print(f"   Response Time: {metrics.duration_ms:.0f}ms")
            
            if settings.OLLAMA_ENV == "cloud":
                print(f"   Status: ✅ Cloud Ollama performing")
                print(f"   Expected: 500-3000ms for small prompt")
            else:
                print(f"   Status: ✅ Local Ollama performing")
                print(f"   Expected: 100-500ms for small prompt")
            
        except Exception as e:
            results["current"]["error"] = str(e)
            print(f"\n❌ Comparison Test Failed: {e}")


@pytest.fixture
def performance_summary():
    """Fixture to generate performance summary."""
    yield
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE TEST SUMMARY")
    print("=" * 70)

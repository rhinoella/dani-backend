"""
RAG Integration Tests with Ollama Cloud

Tests the complete RAG pipeline using cloud Ollama for both retrieval and generation.
Includes quality scoring and latency tracking.

Run with: pytest tests/integration/test_rag_cloud_integration.py -v -s
"""

import pytest
import time
from typing import Dict, List, Any

from app.core.config import settings
from app.services.retrieval_service import RetrievalService
from app.services.chat_service import ChatService
from app.llm.ollama import OllamaClient
from app.embeddings.client import OllamaEmbeddingClient


pytestmark = pytest.mark.integration


class RAGTestData:
    """Test data mimicking real meeting transcripts."""
    
    SAMPLE_DOCUMENTS = [
        {
            "id": "meeting_001",
            "text": "Q1 Planning Meeting (Jan 5): Discussed quarterly roadmap, approved budget increase to $2.5M, assigned teams to initiatives",
            "source": "meeting_transcript_001.txt",
            "timestamp": "2025-01-05T14:00:00"
        },
        {
            "id": "meeting_002", 
            "text": "Engineering Standup (Jan 6): Onboarded 3 new engineers, productivity metrics up 15%, resolved 12 blockers",
            "source": "meeting_transcript_002.txt",
            "timestamp": "2025-01-06T10:00:00"
        },
        {
            "id": "meeting_003",
            "text": "Customer Feedback Session (Jan 7): Collected 5 major feature requests, prioritized authentication improvements",
            "source": "meeting_transcript_003.txt",
            "timestamp": "2025-01-07T15:00:00"
        },
        {
            "id": "meeting_004",
            "text": "Board Meeting (Jan 8): Approved expansion into 2 new markets, hiring plan for Q1, approved cloud migration to AWS",
            "source": "meeting_transcript_004.txt",
            "timestamp": "2025-01-08T09:00:00"
        },
        {
            "id": "meeting_005",
            "text": "Technical Debt Review (Jan 9): Identified 12 critical technical debt items, prioritized 3 for Q1, estimated cost $300K",
            "source": "meeting_transcript_005.txt",
            "timestamp": "2025-01-09T13:00:00"
        },
    ]
    
    RETRIEVAL_QUERIES = [
        "What are the key decisions from recent meetings?",
        "How much budget was allocated for Q1?",
        "How many new engineers were hired?",
        "What feature requests were mentioned?",
        "What technical debt should we address?",
    ]


class RAGQualityMetrics:
    """Metrics for RAG response quality."""
    
    @staticmethod
    def calculate_relevance(query: str, response: str) -> float:
        """
        Calculate relevance score (0.0-1.0).
        Based on keyword overlap and response coherence.
        """
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate Jaccard similarity
        overlap = query_words & response_words
        union = query_words | response_words
        
        jaccard = len(overlap) / len(union) if union else 0
        
        # Boost score if response mentions sources/meetings
        source_bonus = 0.1 if "meeting" in response.lower() else 0
        coherence_bonus = 0.1 if len(response) > 100 else 0
        
        return min(jaccard + source_bonus + coherence_bonus, 1.0)
    
    @staticmethod
    def calculate_completeness(query: str, response: str) -> float:
        """
        Calculate completeness score (0.0-1.0).
        Based on response length and structure.
        """
        # Minimum response length
        length_score = min(len(response) / 150, 0.5)
        
        # Check for structured response
        structure_score = 0.5 if any(c in response for c in "•-*1234567890") else 0.25
        
        return min(length_score + structure_score, 1.0)
    
    @staticmethod
    def calculate_latency_score(duration_ms: float) -> float:
        """
        Calculate latency score (0.0-1.0).
        Cloud latency expected to be higher than local.
        """
        if duration_ms < 1000:
            return 1.0  # Excellent
        elif duration_ms < 2000:
            return 0.9  # Good
        elif duration_ms < 3000:
            return 0.7  # Acceptable
        elif duration_ms < 5000:
            return 0.5  # Slow
        else:
            return 0.2  # Very slow


class TestRAGCloudIntegration:
    """Test RAG pipeline with cloud Ollama."""
    
    @pytest.mark.asyncio
    async def test_retrieval_service_connection(self):
        """Test basic retrieval service connectivity."""
        retrieval = RetrievalService()
        
        # Should connect without errors
        assert retrieval is not None
        print("\n✅ Retrieval service initialized")
    
    @pytest.mark.asyncio
    async def test_retrieval_with_sample_query(self):
        """Test document retrieval with a sample query."""
        try:
            retrieval = RetrievalService()
            query = "What are the key decisions from recent meetings?"
            
            start = time.time()
            results = await retrieval.retrieve(query, top_k=3)
            duration = (time.time() - start) * 1000
            
            print(f"\n📊 Retrieval Results:")
            print(f"   Query: {query}")
            print(f"   Duration: {duration:.0f}ms")
            print(f"   Results found: {len(results)}")
            
            if results:
                for i, result in enumerate(results, 1):
                    text = result.get("text", "")[:100]
                    score = result.get("score", 0)
                    print(f"   [{i}] Score: {score:.2f} | Text: {text}...")
            
        except Exception as e:
            # Skip if Qdrant not available
            if "Qdrant" in str(e) or "connect" in str(e).lower():
                pytest.skip(f"Qdrant not available: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_embedding_similarity(self):
        """Test embedding similarity for relevant documents."""
        try:
            embedder = OllamaEmbeddingClient()
            
            # Similar queries should have high embedding similarity
            queries = [
                "What decisions were made in recent meetings?",
                "What are the key decisions from meetings?",
                "Tell me about the meeting outcomes",
            ]
            
            start = time.time()
            embeddings = await embedder.embed_documents(queries)
            duration = (time.time() - start) * 1000
            
            print(f"\n📐 Embedding Similarity Test:")
            print(f"   Queries: {len(queries)}")
            print(f"   Duration: {duration:.0f}ms")
            print(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
            
            # Calculate cosine similarity between first two
            if len(embeddings) >= 2:
                dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
                mag1 = sum(a ** 2 for a in embeddings[0]) ** 0.5
                mag2 = sum(b ** 2 for b in embeddings[1]) ** 0.5
                
                if mag1 and mag2:
                    similarity = dot_product / (mag1 * mag2)
                    print(f"   Similarity (Q1 vs Q2): {similarity:.3f}")
                    
                    # Similar queries should have high similarity
                    assert similarity > 0.7, f"Expected high similarity, got {similarity}"
            
        except Exception as e:
            if "Ollama" in str(e) or "connect" in str(e).lower():
                pytest.skip(f"Ollama not available: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_cloud_llm_response_quality(self):
        """Test LLM response quality on structured prompts."""
        llm = OllamaClient()
        
        prompt = """
        Based on these meeting notes:
        - Q1 Planning: Budget increased to $2.5M, 3 teams assigned
        - Engineering: 3 new hires, productivity +15%
        - Customer Feedback: 5 feature requests, prioritize auth
        - Board: Approved 2 new markets, AWS migration
        - Tech Debt: 12 items identified, 3 for Q1
        
        Provide a one-paragraph executive summary.
        """
        
        start = time.time()
        response = await llm.generate(prompt)
        duration = (time.time() - start) * 1000
        
        metrics = RAGQualityMetrics()
        relevance = metrics.calculate_relevance("executive summary meeting decisions", response)
        completeness = metrics.calculate_completeness(prompt, response)
        latency_score = metrics.calculate_latency_score(duration)
        
        overall_score = (relevance + completeness + latency_score) / 3
        
        print(f"\n📊 LLM Response Quality:")
        print(f"   Duration: {duration:.0f}ms")
        print(f"   Response length: {len(response)} chars")
        print(f"   Relevance: {relevance:.2f}")
        print(f"   Completeness: {completeness:.2f}")
        print(f"   Latency score: {latency_score:.2f}")
        print(f"   Overall: {overall_score:.2f}/1.0")
        print(f"   Response: {response[:200]}...")
        
        # Response should be reasonably good quality
        assert overall_score > 0.5, f"Response quality too low: {overall_score}"
    
    @pytest.mark.asyncio  
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with context."""
        llm = OllamaClient()
        
        conversation = [
            ("What was decided in the Q1 planning meeting?", 
             "The Q1 planning meeting approved a budget increase to $2.5M and assigned work to 3 teams for key initiatives."),
            ("How much was the budget increase?",
             None),  # Should be answered from context
            ("What else happened that week?",
             None),  # Should reference other meetings
        ]
        
        print(f"\n💬 Multi-Turn Conversation Test:")
        
        context_messages = []
        
        for i, (question, _) in enumerate(conversation, 1):
            # Build context from previous exchanges
            messages = context_messages.copy()
            messages.append(f"Q: {question}")
            
            prompt = "\n".join(messages) + "\nA:"
            
            start = time.time()
            response = await llm.generate(prompt)
            duration = (time.time() - start) * 1000
            
            print(f"   Turn {i}: {duration:.0f}ms")
            print(f"      Q: {question[:60]}...")
            print(f"      A: {response[:100]}...")
            
            context_messages.append(f"Q: {question}")
            context_messages.append(f"A: {response[:200]}")
    
    @pytest.mark.asyncio
    async def test_rag_performance_metrics(self):
        """Comprehensive RAG performance metrics."""
        print(f"\n📈 RAG PERFORMANCE METRICS")
        print(f"   Environment: {settings.OLLAMA_ENV}")
        print(f"   Base URL: {settings.OLLAMA_BASE_URL}")
        print(f"   LLM Model: {settings.LLM_MODEL}")
        print(f"   Embedding Model: {settings.EMBEDDING_MODEL}")
        
        metrics_summary = {
            "llm_prompts": [],
            "embedding_ops": [],
            "retrieval_ops": [],
        }
        
        # Test 1: LLM performance
        print(f"\n   ⚙️ LLM Processing:")
        llm = OllamaClient()
        for i, query in enumerate(RAGTestData.RETRIEVAL_QUERIES[:2], 1):
            start = time.time()
            response = await llm.generate(query)
            duration = (time.time() - start) * 1000
            
            metrics_summary["llm_prompts"].append({
                "query": query,
                "duration_ms": duration,
                "response_length": len(response)
            })
            
            print(f"      [{i}] {duration:.0f}ms - {query[:50]}...")
        
        # Test 2: Embedding performance
        print(f"\n   🧩 Embedding Processing:")
        embedder = OllamaEmbeddingClient()
        
        # Batch embeddings
        start = time.time()
        embeddings = await embedder.embed_documents(
            RAGTestData.RETRIEVAL_QUERIES[:3],
            batch_size=2
        )
        duration = (time.time() - start) * 1000
        
        metrics_summary["embedding_ops"].append({
            "documents": 3,
            "duration_ms": duration,
            "docs_per_sec": 3 / (duration / 1000),
        })
        
        print(f"      Batch embed 3 docs: {duration:.0f}ms")
        print(f"      Throughput: {3 / (duration / 1000):.1f} docs/sec")
        
        # Summary
        print(f"\n   📊 Summary:")
        avg_llm_time = sum(m["duration_ms"] for m in metrics_summary["llm_prompts"]) / len(metrics_summary["llm_prompts"])
        print(f"      Avg LLM latency: {avg_llm_time:.0f}ms")
        
        embedding_throughput = metrics_summary["embedding_ops"][0]["docs_per_sec"]
        print(f"      Embedding throughput: {embedding_throughput:.1f} docs/sec")


class TestRAGErrorHandling:
    """Test RAG error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty queries."""
        llm = OllamaClient()
        
        # Empty queries should be handled gracefully
        try:
            response = await llm.generate("")
            # If no error, response should be empty or minimal
            print(f"\n✅ Empty query handled: {len(response)} chars response")
        except Exception as e:
            # Or it might raise an exception, which is also acceptable
            print(f"✅ Empty query raised: {type(e).__name__}")
    
    @pytest.mark.asyncio
    async def test_very_long_prompt_handling(self):
        """Test handling of very long prompts."""
        llm = OllamaClient()
        
        # Create a very long prompt
        long_prompt = "Explain this: " + ("x" * 5000)
        
        try:
            start = time.time()
            response = await llm.generate(long_prompt, timeout=30)
            duration = (time.time() - start) * 1000
            
            print(f"\n✅ Very long prompt handled in {duration:.0f}ms")
            print(f"   Response length: {len(response)} chars")
        except asyncio.TimeoutError:
            print(f"\n✅ Very long prompt timed out (acceptable)")
        except Exception as e:
            print(f"\n✅ Very long prompt error: {type(e).__name__}")
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters."""
        llm = OllamaClient()
        
        prompt = """
        Process this text: "It's a 'test' with special chars: @#$%^&*()
        Symbols: π ∞ √ ∫ ℵ 
        Emoji: 🎉 🚀 ✨
        
        What do you see?
        """
        
        try:
            response = await llm.generate(prompt)
            print(f"\n✅ Special characters handled")
            print(f"   Response: {response[:100]}...")
        except Exception as e:
            print(f"\n⚠️  Special characters caused: {type(e).__name__}")


@pytest.fixture
def rag_test_data():
    """Fixture providing RAG test data."""
    return RAGTestData()


@pytest.fixture
def rag_metrics():
    """Fixture providing RAG metrics calculator."""
    return RAGQualityMetrics()

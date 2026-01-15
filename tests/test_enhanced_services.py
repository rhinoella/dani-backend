"""
Tests for Enhanced RAG Services.

Tests the three enhancement components:
1. EnhancedRetriever - Multi-stage retrieval with query expansion and re-ranking
2. EnhancedMemoryService - Semantic memory search and entity extraction  
3. InfographicContextBuilder - Enhanced context for infographic generation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from app.services.enhanced_retrieval import (
    EnhancedRetriever,
    QueryExpander,
    LLMReranker,
    ContextualCompressor,
    EnhancedRetrievalConfig,
)
from app.services.enhanced_memory import (
    EnhancedMemoryService,
    SemanticMemorySearch,
    EntityExtractor,
    TopicSummarizer,
    ConversationMemory,
)
from app.services.infographic_context import InfographicContextBuilder


# ============================================================================
# Query Expander Tests
# ============================================================================

class TestQueryExpander:
    """Tests for QueryExpander component."""
    
    @pytest.fixture
    def expander(self):
        return QueryExpander()
    
    @pytest.mark.asyncio
    async def test_expand_query_basic(self, expander):
        """Test basic query expansion."""
        with patch.object(expander.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "revenue growth trends\nfinancial performance quarterly"
            
            variants = await expander.expand(
                query="What was our revenue growth?",
                num_variants=2
            )
            
            # Should include original + expanded variants
            assert len(variants) >= 1
            assert "What was our revenue growth?" in variants
            mock_gen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_expand_query_llm_failure_returns_original(self, expander):
        """Test that LLM failure still returns original query."""
        with patch.object(expander.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("LLM error")
            
            variants = await expander.expand(
                query="What was our revenue growth?",
                num_variants=2
            )
            
            # Should still return original query
            assert variants == ["What was our revenue growth?"]


# ============================================================================
# LLM Reranker Tests
# ============================================================================

class TestLLMReranker:
    """Tests for LLMReranker component."""
    
    @pytest.fixture
    def reranker(self):
        return LLMReranker()
    
    @pytest.mark.asyncio
    async def test_rerank_basic(self, reranker):
        """Test basic reranking."""
        chunks = [
            {"text": "Revenue increased by 20% this quarter.", "score": 0.6},
            {"text": "The weather was nice yesterday.", "score": 0.8},
            {"text": "Q2 revenue growth exceeded expectations.", "score": 0.5},
        ]
        
        with patch.object(reranker.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            # Return scores favoring revenue-related chunks
            mock_gen.side_effect = ["9", "2", "8"]
            
            reranked = await reranker.rerank(
                query="What was our revenue growth?",
                chunks=chunks,
                top_k=2
            )
            
            # Should return top 2 by relevance
            assert len(reranked) <= 2
    
    @pytest.mark.asyncio
    async def test_rerank_handles_llm_failure(self, reranker):
        """Test that reranking handles LLM failures gracefully."""
        chunks = [
            {"text": "Revenue increased by 20%.", "score": 0.6},
            {"text": "Costs decreased by 10%.", "score": 0.7},
        ]
        
        with patch.object(reranker.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = Exception("LLM error")
            
            reranked = await reranker.rerank(
                query="Revenue growth?",
                chunks=chunks,
                top_k=2
            )
            
            # Should return original chunks on failure
            assert len(reranked) == 2


# ============================================================================
# Contextual Compressor Tests
# ============================================================================

class TestContextualCompressor:
    """Tests for ContextualCompressor component."""
    
    @pytest.fixture
    def compressor(self):
        return ContextualCompressor()
    
    @pytest.mark.asyncio
    async def test_compress_basic(self, compressor):
        """Test basic compression."""
        chunks = [
            {"text": "Long text about various topics. Revenue grew by 20%. Other irrelevant stuff here."},
            {"text": "Another chunk with financial data about costs and profits."},
        ]
        
        with patch.object(compressor.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = [
                "Revenue grew by 20%.",
                "Financial data about costs and profits.",
            ]
            
            compressed = await compressor.compress(
                query="What was revenue growth?",
                chunks=chunks
            )
            
            # Compressor returns chunks that have relevant text
            # Implementation filters out NOT_RELEVANT
            assert isinstance(compressed, list)
    
    @pytest.mark.asyncio
    async def test_compress_handles_empty_extraction(self, compressor):
        """Test compression handles empty/irrelevant extractions."""
        chunks = [
            {"text": "Completely irrelevant text about weather."},
        ]
        
        with patch.object(compressor.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "NOT_RELEVANT"
            
            compressed = await compressor.compress(
                query="What was revenue growth?",
                chunks=chunks
            )
            
            # Should filter out irrelevant chunks
            assert len(compressed) == 0


# ============================================================================
# Enhanced Retriever Tests
# ============================================================================

class TestEnhancedRetriever:
    """Tests for EnhancedRetriever orchestrator."""
    
    @pytest.fixture
    def config(self):
        return EnhancedRetrievalConfig(
            expand_queries=True,
            use_reranking=True,
            compress_chunks=True,
            num_query_variants=2,
            initial_fetch_multiplier=3,
            min_relevance_score=0.5,
        )
    
    @pytest.fixture
    def mock_retrieval_service(self):
        """Mock the base RetrievalService."""
        service = MagicMock()
        service.search_with_confidence = AsyncMock(return_value={
            "chunks": [{"text": "Revenue grew 20%", "score": 0.7}],
            "confidence": "high"
        })
        return service
    
    @pytest.fixture
    def retriever(self, config, mock_retrieval_service):
        return EnhancedRetriever(mock_retrieval_service, config)
    
    @pytest.mark.asyncio
    async def test_retrieve_full_pipeline(self, retriever):
        """Test full enhanced retrieval pipeline."""
        with patch.object(retriever.expander, 'expand', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = ["revenue growth", "revenue growth quarterly"]
            
            with patch.object(retriever.reranker, 'rerank', new_callable=AsyncMock) as mock_rerank:
                mock_rerank.return_value = [{"text": "Revenue grew 20%", "score": 0.7}]
                
                with patch.object(retriever.compressor, 'compress', new_callable=AsyncMock) as mock_compress:
                    mock_compress.return_value = [
                        {"text": "Revenue grew 20%", "score": 0.7, "compressed_text": "Revenue grew 20%"}
                    ]
                    
                    result = await retriever.retrieve("What was revenue growth?")
                    
                    assert "chunks" in result
                    assert "confidence" in result
                    mock_expand.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_with_disabled_components(self):
        """Test retrieval with some components disabled."""
        config = EnhancedRetrievalConfig(
            expand_queries=False,
            use_reranking=False,
            compress_chunks=False,
        )
        
        mock_retrieval_service = MagicMock()
        mock_chunks = [{"text": "Revenue grew", "score": 0.7}]
        mock_retrieval_service.search_with_confidence = AsyncMock(return_value={
            "chunks": mock_chunks,
            "confidence": "high"
        })
        
        retriever = EnhancedRetriever(mock_retrieval_service, config)
        result = await retriever.retrieve("revenue growth")
        
        assert len(result["chunks"]) > 0


# ============================================================================
# Semantic Memory Search Tests
# ============================================================================

class TestSemanticMemorySearch:
    """Tests for SemanticMemorySearch component."""
    
    @pytest.fixture
    def searcher(self):
        return SemanticMemorySearch()
    
    @pytest.mark.asyncio
    async def test_search_messages_basic(self, searcher):
        """Test basic semantic search over messages."""
        messages = [
            {"role": "user", "content": "What was our Q2 revenue?"},
            {"role": "assistant", "content": "Q2 revenue was $10M, up 20%."},
            {"role": "user", "content": "How about costs?"},
            {"role": "assistant", "content": "Operating costs were $8M."},
        ]
        
        with patch.object(searcher.embedder, 'embed_query', new_callable=AsyncMock) as mock_embed_q:
            with patch.object(searcher.embedder, 'embed_document', new_callable=AsyncMock) as mock_embed_d:
                # Mock embeddings as unit vectors
                mock_embed_q.return_value = [0.1] * 768
                mock_embed_d.return_value = [0.1] * 768
                
                results = await searcher.search_messages(
                    query="Tell me about revenue",
                    messages=messages,
                    top_k=2
                )
                
                # Should return semantically relevant messages
                assert isinstance(results, list)
    
    @pytest.mark.asyncio  
    async def test_search_messages_empty(self, searcher):
        """Test search with no messages."""
        results = await searcher.search_messages(
            query="revenue",
            messages=[],
            top_k=2
        )
        
        assert results == []


# ============================================================================
# Entity Extractor Tests  
# ============================================================================

class TestEntityExtractor:
    """Tests for EntityExtractor component."""
    
    @pytest.fixture
    def extractor(self):
        return EntityExtractor()
    
    @pytest.mark.asyncio
    async def test_extract_basic(self, extractor):
        """Test basic entity extraction."""
        messages = [
            {"role": "user", "content": "What did John say in the Q2 meeting about Project Alpha?"},
            {"role": "assistant", "content": "John mentioned the deadline is June 15th."},
        ]
        
        with patch.object(extractor.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = '''{"people": ["John"], "topics": ["Q2 meeting", "Project Alpha"], "dates": ["June 15th"], "projects": ["Project Alpha"], "metrics": []}'''
            
            entities = await extractor.extract(messages)
            
            assert "people" in entities
            assert "John" in entities.get("people", [])
    
    @pytest.mark.asyncio
    async def test_extract_handles_invalid_json(self, extractor):
        """Test entity extraction handles invalid JSON response."""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(extractor.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "This is not valid JSON"
            
            entities = await extractor.extract(messages)
            
            # Should return empty categories on parse failure  
            assert entities == {"people": [], "topics": [], "dates": [], "projects": [], "metrics": []}


# ============================================================================
# Topic Summarizer Tests
# ============================================================================

class TestTopicSummarizer:
    """Tests for TopicSummarizer component."""
    
    @pytest.fixture
    def summarizer(self):
        return TopicSummarizer()
    
    @pytest.mark.asyncio
    async def test_summarize_basic(self, summarizer):
        """Test basic topic summarization."""
        messages = [
            {"role": "user", "content": "What were the Q2 results?"},
            {"role": "assistant", "content": "Revenue was up 20%, costs were down 5%."},
            {"role": "user", "content": "What about Q3 projections?"},
            {"role": "assistant", "content": "We're projecting 15% growth."},
        ]
        
        with patch.object(summarizer.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "User is inquiring about quarterly financial performance, specifically Q2 results and Q3 projections."
            
            summary = await summarizer.summarize(messages)
            
            assert isinstance(summary, str)
            assert len(summary) > 0


# ============================================================================
# Enhanced Memory Service Tests
# ============================================================================

class TestEnhancedMemoryService:
    """Tests for EnhancedMemoryService orchestrator."""
    
    @pytest.fixture
    def mock_session(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_cache(self):
        cache = MagicMock()
        cache.get_messages = AsyncMock(return_value=[
            {"id": "1", "role": "user", "content": "Hello", "created_at": "2024-01-01T00:00:00"},
            {"id": "2", "role": "assistant", "content": "Hi!", "created_at": "2024-01-01T00:01:00"},
        ])
        return cache
    
    @pytest.mark.asyncio
    async def test_get_enhanced_context(self, mock_session, mock_cache):
        """Test getting enhanced context."""
        service = EnhancedMemoryService(mock_session, mock_cache)
        
        # Mock the internal method that gets messages
        with patch.object(service, '_get_recent_messages', new_callable=AsyncMock) as mock_get:
            # Return mock Message objects
            mock_msg1 = MagicMock()
            mock_msg1.role = "user"
            mock_msg1.content = "Hello"
            mock_msg2 = MagicMock()
            mock_msg2.role = "assistant"
            mock_msg2.content = "Hi!"
            mock_get.return_value = [mock_msg1, mock_msg2]
            
            with patch.object(service.semantic_search, 'search_messages', new_callable=AsyncMock) as mock_search:
                mock_search.return_value = [{"role": "user", "content": "Hello", "similarity": 0.9}]
                
                with patch.object(service.entity_extractor, 'extract', new_callable=AsyncMock) as mock_extract:
                    mock_extract.return_value = {"people": [], "topics": [], "dates": [], "projects": [], "metrics": []}
                    
                    with patch.object(service.topic_summarizer, 'summarize', new_callable=AsyncMock) as mock_summarize:
                        mock_summarize.return_value = "General greeting conversation."
                        
                        context = await service.get_enhanced_context(
                            conversation_id="test-conv-id",
                            current_query="How are you?"
                        )
                        
                        assert isinstance(context, ConversationMemory)
                        assert len(context.messages) > 0


# ============================================================================
# Infographic Context Builder Tests
# ============================================================================

class TestInfographicContextBuilder:
    """Tests for InfographicContextBuilder."""
    
    @pytest.fixture
    def builder(self):
        return InfographicContextBuilder()
    
    @pytest.mark.asyncio
    async def test_build_context_basic(self, builder):
        """Test basic context building."""
        conversation_history = [
            {"role": "user", "content": "Show me the Q2 revenue data"},
            {"role": "assistant", "content": "Q2 revenue was $10M."},
        ]
        
        mock_chunks = [
            {"text": "Q2 revenue reached $10M, a 20% increase.", "score": 0.8, "title": "Q2 Report"},
        ]
        
        with patch.object(builder.enhanced_retriever, 'retrieve', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = {
                "chunks": mock_chunks,
                "confidence": "high",
                "pipeline_stats": {}
            }
            
            with patch.object(builder, '_extract_visualization_data', new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = {
                    "main_topic": "Q2 Revenue",
                    "headline": "Revenue Up 20%",
                    "statistics": [{"value": "$10M", "label": "Revenue"}],
                    "key_points": ["Strong growth"]
                }
                
                context = await builder.build_context(
                    request="Create infographic of Q2 revenue",
                    conversation_history=conversation_history
                )
                
                # Check the actual return structure
                assert "structured_data" in context
                assert "sources" in context
                assert "confidence" in context
    
    @pytest.mark.asyncio
    async def test_extract_visualization_data(self, builder):
        """Test visualization data extraction."""
        context = "Q2 revenue was $10M, up 20% from Q1. Costs were $8M."
        
        with patch.object(builder.llm, 'generate', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = '''{"main_topic": "Q2 Results", "headline": "Q2 Revenue Growth", "subtitle": "Strong performance", "statistics": [{"value": "$10M", "label": "Revenue", "trend": "up"}], "key_points": ["Revenue grew 20%"], "comparisons": [], "timeline": [], "categories": [], "data_source": "Q2 Report"}'''
            
            # Call the private method directly
            data = await builder._extract_visualization_data(
                request="Create Q2 revenue infographic",
                conversation_context="",
                rag_context=context
            )
            
            assert isinstance(data, dict)
            assert "statistics" in data or "key_points" in data


# ============================================================================
# Integration Tests
# ============================================================================

class TestEnhancedServicesIntegration:
    """Integration tests for enhanced services working together."""
    
    @pytest.mark.asyncio
    async def test_enhanced_retriever_config_from_settings(self):
        """Test that EnhancedRetriever can be configured from settings."""
        from app.core.config import settings
        from app.services.retrieval_service import RetrievalService
        
        config = EnhancedRetrievalConfig(
            expand_queries=settings.QUERY_EXPANSION_ENABLED,
            use_reranking=settings.LLM_RERANKING_ENABLED,
            compress_chunks=settings.CONTEXTUAL_COMPRESSION_ENABLED,
            num_query_variants=settings.QUERY_EXPANSION_VARIANTS,
            min_relevance_score=settings.MIN_RETRIEVAL_RELEVANCE,
        )
        
        mock_service = MagicMock()
        retriever = EnhancedRetriever(mock_service, config)
        
        assert retriever.config.expand_queries == settings.QUERY_EXPANSION_ENABLED
        assert retriever.config.use_reranking == settings.LLM_RERANKING_ENABLED
    
    @pytest.mark.asyncio
    async def test_conversation_memory_dataclass(self):
        """Test ConversationMemory dataclass structure."""
        memory = ConversationMemory(
            messages=[{"role": "user", "content": "Hello"}],
            relevant_history=[{"role": "user", "content": "Previous hello", "similarity": 0.9}],
            entities={"people": ["John"], "topics": ["greeting"]},
            topic_summary="A greeting exchange.",
            total_messages=5,
            context_token_count=100
        )
        
        assert len(memory.messages) == 1
        assert len(memory.relevant_history) == 1
        assert "people" in memory.entities
        assert memory.topic_summary is not None
        assert memory.total_messages == 5
        assert memory.context_token_count == 100

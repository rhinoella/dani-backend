"""
Tests for RAG Analytics Service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import uuid

from app.services.rag_analytics_service import RAGAnalyticsService, log_rag_interaction
from app.database.models import RAGLog


# ============== Fixtures ==============

@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.fixture
def sample_rag_log():
    """Create a sample RAG log entry."""
    log = MagicMock(spec=RAGLog)
    log.id = uuid.uuid4()
    log.user_id = uuid.uuid4()
    log.conversation_id = uuid.uuid4()
    log.query = "What was discussed in the meeting?"
    log.query_length = 35
    log.query_intent = "information_retrieval"
    log.chunks_retrieved = 5
    log.chunks_used = 3
    log.confidence_score = 0.85
    log.confidence_level = "high"
    log.total_latency_ms = 1500.0
    log.retrieval_latency_ms = 200.0
    log.generation_latency_ms = 1200.0
    log.cache_hit = False
    log.success = True
    log.error_type = None
    log.error_message = None
    log.user_rating = None
    log.user_feedback = None
    log.feedback_at = None
    log.created_at = datetime.utcnow()
    return log


# ============== RAGAnalyticsService Tests ==============

class TestRAGAnalyticsService:
    """Tests for RAGAnalyticsService."""
    
    def test_init_with_session(self, mock_session):
        """Test initialization with session."""
        service = RAGAnalyticsService(session=mock_session)
        assert service.session == mock_session
    
    def test_init_without_session(self):
        """Test initialization without session."""
        service = RAGAnalyticsService()
        assert service.session is None
    
    @pytest.mark.asyncio
    async def test_log_interaction_no_session(self):
        """Test log_interaction returns None when no session."""
        service = RAGAnalyticsService(session=None)
        result = await service.log_interaction(query="test query")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_log_interaction_success(self, mock_session, sample_rag_log):
        """Test successful RAG interaction logging."""
        mock_session.refresh = AsyncMock(side_effect=lambda x: setattr(x, 'id', sample_rag_log.id))
        
        service = RAGAnalyticsService(session=mock_session)
        
        with patch('app.services.rag_analytics_service.RAGLog') as MockRAGLog:
            mock_log_instance = MagicMock()
            mock_log_instance.id = sample_rag_log.id
            MockRAGLog.return_value = mock_log_instance
            
            result = await service.log_interaction(
                query="What was discussed?",
                user_id=sample_rag_log.user_id,
                chunks_retrieved=5,
                chunks_used=3,
                confidence_score=0.85,
                total_latency_ms=1500.0,
                success=True,
            )
        
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_interaction_with_all_params(self, mock_session):
        """Test logging with all parameters."""
        service = RAGAnalyticsService(session=mock_session)
        
        with patch('app.services.rag_analytics_service.RAGLog') as MockRAGLog:
            mock_log_instance = MagicMock()
            mock_log_instance.id = uuid.uuid4()
            MockRAGLog.return_value = mock_log_instance
            
            result = await service.log_interaction(
                query="Test query",
                user_id=uuid.uuid4(),
                conversation_id=uuid.uuid4(),
                query_intent="question",
                query_entities={"entities": ["meeting"]},
                chunks_retrieved=10,
                chunks_used=5,
                retrieval_scores=[0.9, 0.85, 0.8],
                sources=[{"file": "doc.pdf"}],
                answer_length=200,
                output_format="text",
                confidence_score=0.9,
                confidence_level="high",
                confidence_reason="Good context",
                embedding_latency_ms=50.0,
                retrieval_latency_ms=100.0,
                generation_latency_ms=1000.0,
                total_latency_ms=1150.0,
                cache_hit=True,
                cache_type="semantic",
                success=True,
                error_type=None,
                error_message=None,
                model_used="llama3.2:3b",
                embedding_model="nomic-embed-text",
                metadata={"extra": "data"},
            )
        
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_interaction_exception_rollback(self, mock_session):
        """Test that exceptions trigger rollback."""
        mock_session.commit.side_effect = Exception("DB error")
        
        service = RAGAnalyticsService(session=mock_session)
        
        with patch('app.services.rag_analytics_service.RAGLog') as MockRAGLog:
            MockRAGLog.return_value = MagicMock()
            
            result = await service.log_interaction(query="test query")
        
        assert result is None
        mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_feedback_no_session(self):
        """Test add_feedback returns False when no session."""
        service = RAGAnalyticsService(session=None)
        result = await service.add_feedback(uuid.uuid4(), rating=5)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_add_feedback_success(self, mock_session, sample_rag_log):
        """Test successful feedback addition."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_rag_log
        mock_session.execute.return_value = mock_result
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.add_feedback(
            log_id=sample_rag_log.id,
            rating=5,
            feedback="Very helpful!"
        )
        
        assert result is True
        assert sample_rag_log.user_rating == 5
        assert sample_rag_log.user_feedback == "Very helpful!"
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_feedback_log_not_found(self, mock_session):
        """Test feedback when log not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.add_feedback(uuid.uuid4(), rating=5)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_add_feedback_exception(self, mock_session, sample_rag_log):
        """Test feedback exception handling."""
        mock_session.execute.side_effect = Exception("DB error")
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.add_feedback(sample_rag_log.id, rating=5)
        
        assert result is False
        mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_stats_no_session(self):
        """Test get_stats returns empty dict when no session."""
        service = RAGAnalyticsService(session=None)
        result = await service.get_stats()
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_stats_success(self, mock_session):
        """Test successful stats retrieval."""
        # Mock query results
        mock_results = [
            MagicMock(scalar=MagicMock(return_value=100)),  # total queries
            MagicMock(scalar=MagicMock(return_value=95)),   # successful
            MagicMock(scalar=MagicMock(return_value=30)),   # cache hits
            MagicMock(one=MagicMock(return_value=(1500.0, 200.0, 1000.0, 0.85))),  # latencies
            MagicMock(all=MagicMock(return_value=[("high", 80), ("medium", 15)])),  # confidence dist
            MagicMock(all=MagicMock(return_value=[("question", 60), ("summary", 40)])),  # intent dist
            MagicMock(scalar=MagicMock(return_value=4.5)),  # avg rating
        ]
        mock_session.execute.side_effect = mock_results
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_stats(hours=24)
        
        assert result["period_hours"] == 24
        assert result["total_queries"] == 100
        assert result["successful_queries"] == 95
        assert result["success_rate"] == 0.95
        assert result["cache_hits"] == 30
        assert result["cache_hit_rate"] == 0.3
    
    @pytest.mark.asyncio
    async def test_get_stats_with_user_filter(self, mock_session):
        """Test stats with user filter."""
        user_id = uuid.uuid4()
        
        mock_results = [
            MagicMock(scalar=MagicMock(return_value=50)),
            MagicMock(scalar=MagicMock(return_value=48)),
            MagicMock(scalar=MagicMock(return_value=15)),
            MagicMock(one=MagicMock(return_value=(1200.0, 150.0, 900.0, 0.88))),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(scalar=MagicMock(return_value=None)),
        ]
        mock_session.execute.side_effect = mock_results
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_stats(hours=24, user_id=user_id)
        
        assert result["total_queries"] == 50
    
    @pytest.mark.asyncio
    async def test_get_stats_zero_queries(self, mock_session):
        """Test stats with zero queries."""
        mock_results = [
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(scalar=MagicMock(return_value=0)),
            MagicMock(one=MagicMock(return_value=(None, None, None, None))),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(scalar=MagicMock(return_value=None)),
        ]
        mock_session.execute.side_effect = mock_results
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_stats()
        
        assert result["success_rate"] == 0
        assert result["cache_hit_rate"] == 0
    
    @pytest.mark.asyncio
    async def test_get_stats_exception(self, mock_session):
        """Test stats exception handling."""
        mock_session.execute.side_effect = Exception("DB error")
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_stats()
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_recent_errors_no_session(self):
        """Test get_recent_errors returns empty list when no session."""
        service = RAGAnalyticsService(session=None)
        result = await service.get_recent_errors()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_recent_errors_success(self, mock_session):
        """Test successful error retrieval."""
        error_log = MagicMock()
        error_log.id = uuid.uuid4()
        error_log.query = "Failed query"
        error_log.error_type = "generation_error"
        error_log.error_message = "Model timeout"
        error_log.created_at = datetime.utcnow()
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [error_log]
        mock_session.execute.return_value = mock_result
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_recent_errors(limit=10)
        
        assert len(result) == 1
        assert result[0]["error_type"] == "generation_error"
    
    @pytest.mark.asyncio
    async def test_get_recent_errors_exception(self, mock_session):
        """Test error retrieval exception handling."""
        mock_session.execute.side_effect = Exception("DB error")
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_recent_errors()
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_slow_queries_no_session(self):
        """Test get_slow_queries returns empty list when no session."""
        service = RAGAnalyticsService(session=None)
        result = await service.get_slow_queries()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_slow_queries_success(self, mock_session):
        """Test successful slow query retrieval."""
        slow_log = MagicMock()
        slow_log.id = uuid.uuid4()
        slow_log.query = "Complex query that took long"
        slow_log.total_latency_ms = 8000.0
        slow_log.retrieval_latency_ms = 500.0
        slow_log.generation_latency_ms = 7000.0
        slow_log.chunks_retrieved = 20
        slow_log.cache_hit = False
        slow_log.created_at = datetime.utcnow()
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [slow_log]
        mock_session.execute.return_value = mock_result
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_slow_queries(threshold_ms=5000, limit=10)
        
        assert len(result) == 1
        assert result[0]["total_latency_ms"] == 8000.0
    
    @pytest.mark.asyncio
    async def test_get_slow_queries_exception(self, mock_session):
        """Test slow queries exception handling."""
        mock_session.execute.side_effect = Exception("DB error")
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_slow_queries()
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_low_confidence_queries_no_session(self):
        """Test get_low_confidence_queries returns empty list when no session."""
        service = RAGAnalyticsService(session=None)
        result = await service.get_low_confidence_queries()
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_low_confidence_queries_success(self, mock_session):
        """Test successful low confidence query retrieval."""
        low_conf_log = MagicMock()
        low_conf_log.id = uuid.uuid4()
        low_conf_log.query = "Vague question"
        low_conf_log.confidence_score = 0.3
        low_conf_log.confidence_level = "low"
        low_conf_log.confidence_reason = "Insufficient context"
        low_conf_log.chunks_retrieved = 2
        low_conf_log.query_intent = "unclear"
        low_conf_log.created_at = datetime.utcnow()
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [low_conf_log]
        mock_session.execute.return_value = mock_result
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_low_confidence_queries(threshold=0.5, limit=10)
        
        assert len(result) == 1
        assert result[0]["confidence_score"] == 0.3
    
    @pytest.mark.asyncio
    async def test_get_low_confidence_queries_exception(self, mock_session):
        """Test low confidence exception handling."""
        mock_session.execute.side_effect = Exception("DB error")
        
        service = RAGAnalyticsService(session=mock_session)
        
        result = await service.get_low_confidence_queries()
        
        assert result == []


class TestLogRagInteractionHelper:
    """Tests for the log_rag_interaction helper function."""
    
    @pytest.mark.asyncio
    async def test_log_rag_interaction_helper(self, mock_session):
        """Test the convenience function."""
        with patch('app.services.rag_analytics_service.RAGLog') as MockRAGLog:
            mock_log = MagicMock()
            mock_log.id = uuid.uuid4()
            MockRAGLog.return_value = mock_log
            
            result = await log_rag_interaction(
                session=mock_session,
                query="Test query",
                success=True
            )
    
    @pytest.mark.asyncio
    async def test_log_rag_interaction_helper_no_session(self):
        """Test helper with no session."""
        result = await log_rag_interaction(session=None, query="test")
        assert result is None

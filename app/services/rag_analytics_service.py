"""
RAG Analytics Service for logging and analyzing RAG interactions.

Provides:
- Automatic logging of RAG pipeline executions
- Query performance metrics
- Quality analytics
- User feedback collection
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from app.database.models import RAGLog
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGAnalyticsService:
    """Service for RAG interaction logging and analytics."""
    
    def __init__(self, session: Optional[AsyncSession] = None):
        self.session = session
    
    async def log_interaction(
        self,
        query: str,
        *,
        user_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
        query_intent: Optional[str] = None,
        query_entities: Optional[Dict] = None,
        chunks_retrieved: int = 0,
        chunks_used: int = 0,
        retrieval_scores: Optional[List[float]] = None,
        sources: Optional[List[Dict]] = None,
        answer_length: Optional[int] = None,
        output_format: Optional[str] = None,
        confidence_score: Optional[float] = None,
        confidence_level: Optional[str] = None,
        confidence_reason: Optional[str] = None,
        embedding_latency_ms: Optional[float] = None,
        retrieval_latency_ms: Optional[float] = None,
        generation_latency_ms: Optional[float] = None,
        total_latency_ms: Optional[float] = None,
        cache_hit: bool = False,
        cache_type: Optional[str] = None,
        success: bool = True,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        model_used: Optional[str] = None,
        embedding_model: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[RAGLog]:
        """
        Log a RAG interaction to the database.
        
        This is designed to be non-blocking and fail-safe - errors
        in logging should never affect the main RAG pipeline.
        """
        if self.session is None:
            logger.debug("No session provided, skipping RAG log")
            return None
        
        try:
            rag_log = RAGLog(
                user_id=user_id,
                conversation_id=conversation_id,
                query=query[:2000],
                query_length=len(query),
                query_intent=query_intent,
                query_entities=query_entities,
                chunks_retrieved=chunks_retrieved,
                chunks_used=chunks_used,
                retrieval_scores={"scores": retrieval_scores} if retrieval_scores else None,
                sources={"sources": sources} if sources else None,
                answer_length=answer_length,
                output_format=output_format,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                confidence_reason=confidence_reason,
                embedding_latency_ms=embedding_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                cache_hit=cache_hit,
                cache_type=cache_type,
                success=success,
                error_type=error_type,
                error_message=error_message,
                model_used=model_used or settings.LLM_MODEL,
                embedding_model=embedding_model or settings.EMBEDDING_MODEL,
                metadata_=metadata or {},
            )
            
            self.session.add(rag_log)
            await self.session.commit()
            await self.session.refresh(rag_log)
            
            logger.debug(f"Logged RAG interaction: {rag_log.id}")
            return rag_log
            
        except Exception as e:
            logger.warning(f"Failed to log RAG interaction: {e}")
            # Don't raise - logging failures shouldn't break RAG
            await self.session.rollback()
            return None
    
    async def add_feedback(
        self,
        log_id: UUID,
        rating: int,
        feedback: Optional[str] = None,
    ) -> bool:
        """Add user feedback to a RAG log entry."""
        if self.session is None:
            return False
        
        try:
            result = await self.session.execute(
                select(RAGLog).where(RAGLog.id == log_id)
            )
            rag_log = result.scalar_one_or_none()
            
            if not rag_log:
                return False
            
            rag_log.user_rating = rating
            rag_log.user_feedback = feedback
            rag_log.feedback_at = datetime.utcnow()
            
            await self.session.commit()
            return True
            
        except Exception as e:
            logger.warning(f"Failed to add feedback: {e}")
            await self.session.rollback()
            return False
    
    async def get_stats(
        self,
        hours: int = 24,
        user_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """Get RAG pipeline statistics for the specified time period."""
        if self.session is None:
            return {}
        
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Base filter
            filters = [RAGLog.created_at >= since]
            if user_id:
                filters.append(RAGLog.user_id == user_id)
            
            # Total queries
            total_result = await self.session.execute(
                select(func.count(RAGLog.id)).where(and_(*filters))
            )
            total_queries = total_result.scalar() or 0
            
            # Success rate
            success_result = await self.session.execute(
                select(func.count(RAGLog.id)).where(
                    and_(*filters, RAGLog.success == True)
                )
            )
            successful = success_result.scalar() or 0
            
            # Cache hit rate
            cache_result = await self.session.execute(
                select(func.count(RAGLog.id)).where(
                    and_(*filters, RAGLog.cache_hit == True)
                )
            )
            cache_hits = cache_result.scalar() or 0
            
            # Average latencies
            latency_result = await self.session.execute(
                select(
                    func.avg(RAGLog.total_latency_ms),
                    func.avg(RAGLog.retrieval_latency_ms),
                    func.avg(RAGLog.generation_latency_ms),
                    func.avg(RAGLog.confidence_score),
                ).where(and_(*filters, RAGLog.success == True))
            )
            latencies = latency_result.one()
            
            # Confidence level distribution
            confidence_result = await self.session.execute(
                select(
                    RAGLog.confidence_level,
                    func.count(RAGLog.id)
                ).where(and_(*filters)).group_by(RAGLog.confidence_level)
            )
            confidence_dist = {
                level: count for level, count in confidence_result.all() if level
            }
            
            # Query intent distribution
            intent_result = await self.session.execute(
                select(
                    RAGLog.query_intent,
                    func.count(RAGLog.id)
                ).where(and_(*filters)).group_by(RAGLog.query_intent)
            )
            intent_dist = {
                intent: count for intent, count in intent_result.all() if intent
            }
            
            # Average rating (if feedback exists)
            rating_result = await self.session.execute(
                select(func.avg(RAGLog.user_rating)).where(
                    and_(*filters, RAGLog.user_rating.isnot(None))
                )
            )
            avg_rating = rating_result.scalar()
            
            return {
                "period_hours": hours,
                "total_queries": total_queries,
                "successful_queries": successful,
                "success_rate": round(successful / total_queries, 3) if total_queries > 0 else 0,
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hits / total_queries, 3) if total_queries > 0 else 0,
                "avg_total_latency_ms": round(latencies[0] or 0, 2),
                "avg_retrieval_latency_ms": round(latencies[1] or 0, 2),
                "avg_generation_latency_ms": round(latencies[2] or 0, 2),
                "avg_confidence_score": round(latencies[3] or 0, 3),
                "confidence_distribution": confidence_dist,
                "intent_distribution": intent_dist,
                "avg_user_rating": round(avg_rating, 2) if avg_rating else None,
            }
            
        except Exception as e:
            logger.warning(f"Failed to get RAG stats: {e}")
            return {}
    
    async def get_recent_errors(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent failed RAG interactions for debugging."""
        if self.session is None:
            return []
        
        try:
            result = await self.session.execute(
                select(RAGLog)
                .where(RAGLog.success == False)
                .order_by(desc(RAGLog.created_at))
                .limit(limit)
            )
            logs = result.scalars().all()
            
            return [
                {
                    "id": str(log.id),
                    "query": log.query[:100],
                    "error_type": log.error_type,
                    "error_message": log.error_message,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get recent errors: {e}")
            return []
    
    async def get_slow_queries(
        self,
        threshold_ms: float = 5000,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get slowest RAG queries for performance analysis."""
        if self.session is None:
            return []
        
        try:
            result = await self.session.execute(
                select(RAGLog)
                .where(
                    and_(
                        RAGLog.success == True,
                        RAGLog.total_latency_ms > threshold_ms,
                    )
                )
                .order_by(desc(RAGLog.total_latency_ms))
                .limit(limit)
            )
            logs = result.scalars().all()
            
            return [
                {
                    "id": str(log.id),
                    "query": log.query[:100],
                    "total_latency_ms": log.total_latency_ms,
                    "retrieval_latency_ms": log.retrieval_latency_ms,
                    "generation_latency_ms": log.generation_latency_ms,
                    "chunks_retrieved": log.chunks_retrieved,
                    "cache_hit": log.cache_hit,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get slow queries: {e}")
            return []
    
    async def get_low_confidence_queries(
        self,
        threshold: float = 0.5,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get low confidence queries for quality analysis."""
        if self.session is None:
            return []
        
        try:
            result = await self.session.execute(
                select(RAGLog)
                .where(
                    and_(
                        RAGLog.success == True,
                        RAGLog.confidence_score < threshold,
                    )
                )
                .order_by(RAGLog.confidence_score)
                .limit(limit)
            )
            logs = result.scalars().all()
            
            return [
                {
                    "id": str(log.id),
                    "query": log.query[:100],
                    "confidence_score": log.confidence_score,
                    "confidence_level": log.confidence_level,
                    "confidence_reason": log.confidence_reason,
                    "chunks_retrieved": log.chunks_retrieved,
                    "query_intent": log.query_intent,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get low confidence queries: {e}")
            return []


# Helper function for easy logging from services
async def log_rag_interaction(
    session: Optional[AsyncSession],
    **kwargs,
) -> Optional[RAGLog]:
    """Convenience function to log a RAG interaction."""
    service = RAGAnalyticsService(session)
    return await service.log_interaction(**kwargs)

from __future__ import annotations

import logging
from typing import Dict, Any, List, AsyncIterator, Optional
import json
import time

from qdrant_client.http import models as qm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.metrics import metrics
from app.services.retrieval_service import RetrievalService
from app.services.query_rewriter import QueryRewriter, get_query_rewriter
from app.services.agent_service import get_agent_service, Intent
from app.services.enhanced_retrieval import EnhancedRetriever, EnhancedRetrievalConfig
from app.llm.ollama import OllamaClient
from app.llm.prompt_builder import PromptBuilder
from app.llm.output_validator import OutputValidator
from app.persona.templates import validate_output_format
from app.cache.semantic_cache import ResponseCache
from app.cache.conversation_cache import ConversationCache
from app.utils.query_processor import ConfidenceScorer
from app.schemas.retrieval import MetadataFilter

logger = logging.getLogger(__name__)


class ChatService:
    """
    Enhanced RAG chat service with:
    - Semantic response caching
    - Confidence scoring
    - Source attribution
    - Structured output support
    - Automatic conversation history loading
    - Smart context window management
    - Conversation summarization
    """

    def __init__(self, use_enhanced_retrieval: bool = False) -> None:
        self.retrieval = RetrievalService()
        self.llm = OllamaClient()
        self.prompt_builder = PromptBuilder()
        self.validator = OutputValidator()
        self.confidence_scorer = ConfidenceScorer()
        self.query_rewriter = get_query_rewriter()
        
        # Enhanced retrieval for higher accuracy (optional)
        self.use_enhanced_retrieval = use_enhanced_retrieval
        if use_enhanced_retrieval:
            self.enhanced_retriever = EnhancedRetriever(
                self.retrieval,
                EnhancedRetrievalConfig(
                    expand_queries=True,
                    num_query_variants=2,
                    use_reranking=True,
                    compress_chunks=True,
                    min_relevance_score=0.60,
                )
            )
        
        # Response cache for full LLM responses
        self._response_cache = ResponseCache(
            similarity_threshold=0.92,
            max_size=200,
            ttl_seconds=1800,  # 30 minutes
        )
        
        # Conversation cache for history
        self._conversation_cache: Optional[ConversationCache] = None
        
        # Context window limits
        self.max_history_messages = 10  # Keep last 10 messages
        self.max_history_tokens = 2000  # Max tokens for history
        self.summarize_threshold = 20  # Summarize after 20 messages
    
    def set_conversation_cache(self, cache: ConversationCache) -> None:
        """Set conversation cache for history loading."""
        self._conversation_cache = cache
    
    async def _load_conversation_history(
        self,
        conversation_id: str,
        session: AsyncSession
    ) -> List[Dict[str, str]]:
        """
        Load conversation history from database with caching.
        
        Args:
            conversation_id: ID of conversation to load
            session: Database session
            
        Returns:
            List of {"role": "user"|"assistant", "content": str}
        """
        # Try cache first
        if self._conversation_cache:
            cached = await self._conversation_cache.get_messages(conversation_id)
            if cached:
                logger.info(f"Loaded {len(cached)} messages from cache for conversation {conversation_id}")
                return cached
        
        # Load from database
        from app.repositories.message_repository import MessageRepository
        msg_repo = MessageRepository(session)
        
        # Get recent messages (limit to avoid huge history)
        messages = await msg_repo.get_by_conversation(
            conversation_id,
            limit=self.max_history_messages * 2,  # Get more to allow for compression
            ascending=False  # Most recent first
        )
        
        # Convert to conversation history format (oldest first for context)
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in reversed(messages)
        ]
        
        logger.info(f"Loaded {len(history)} messages from DB for conversation {conversation_id}")
        
        # Cache for future requests
        if self._conversation_cache:
            await self._conversation_cache.set_messages(conversation_id, history)
        
        return history
    
    def _compress_conversation_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Compress conversation history to fit within token budget.
        Uses sliding window approach: keeps most recent messages.
        
        Args:
            history: Full conversation history
            max_tokens: Maximum tokens to use (defaults to self.max_history_tokens)
            
        Returns:
            Compressed history within token budget
        """
        if not history:
            return []
        
        max_tokens = max_tokens or self.max_history_tokens
        compressed = []
        total_tokens = 0
        
        # Work backwards (most recent first)
        for msg in reversed(history):
            # Rough token estimate: 1 token ≈ 4 characters
            msg_tokens = len(msg["content"]) // 4
            
            if total_tokens + msg_tokens > max_tokens:
                break
            
            compressed.insert(0, msg)
            total_tokens += msg_tokens
        
        if len(compressed) < len(history):
            logger.info(f"Compressed history from {len(history)} to {len(compressed)} messages ({total_tokens} tokens)")
        
        return compressed
    
    async def _summarize_old_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Summarize older messages into a brief summary.
        Used when conversation exceeds threshold.
        
        Args:
            messages: Messages to summarize (typically messages 11-20)
            
        Returns:
            Brief summary string
        """
        if not messages or len(messages) < 3:
            return ""
        
        # Format messages for summarization
        message_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])
        
        summary_prompt = f"""Summarize this conversation excerpt concisely in 2-3 sentences:

{message_text}

Summary:"""
        
        try:
            summary = await self.llm.generate(summary_prompt, stream=False)
            logger.info(f"Summarized {len(messages)} messages into {len(summary)} chars")
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to summarize messages: {e}")
            return ""
    
    async def _prepare_conversation_context(
        self,
        conversation_id: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
        session: Optional[AsyncSession]
    ) -> List[Dict[str, str]]:
        """
        Prepare conversation context with auto-loading, caching, and compression.
        
        Args:
            conversation_id: Optional conversation ID to auto-load history
            conversation_history: Manually provided history (takes precedence)
            session: Database session for loading history
            
        Returns:
            Prepared conversation history within token budget
        """
        # If history provided manually, use it
        if conversation_history:
            return self._compress_conversation_history(conversation_history)
        
        # Auto-load from conversation_id if provided
        if conversation_id and session:
            history = await self._load_conversation_history(conversation_id, session)
            
            # If conversation is very long, summarize older messages
            if len(history) > self.summarize_threshold:
                logger.info(f"Conversation has {len(history)} messages, applying summarization")
                
                # Keep last 10 messages as-is
                recent_messages = history[-self.max_history_messages:]
                
                # Summarize older messages (11-20)
                old_messages = history[-(self.summarize_threshold):-self.max_history_messages]
                summary = await self._summarize_old_messages(old_messages)
                
                # Create context with summary + recent messages
                if summary:
                    return [{"role": "system", "content": f"Previous conversation summary: {summary}"}] + recent_messages
                else:
                    return recent_messages
            
            # Otherwise just compress to fit token budget
            return self._compress_conversation_history(history)
        
        # No history available
        return []

    async def answer(
        self, 
        query: str, 
        verbose: bool = False,
        output_format: Optional[str] = None,
        use_cache: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        doc_type: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a RAG-grounded answer.
        
        Args:
            query: User's question
            verbose: Include debug info in response
            output_format: Optional output format template
            use_cache: Use response cache
            conversation_history: List of {"role": "user"|"assistant", "content": str}
            conversation_id: ID of conversation to auto-load history from
            session: Database session for loading conversation history
            doc_type: Filter by document type (meeting, email, document, note, all)
            document_ids: Optional list of document IDs to restrict search to
        """
        start_time = time.time()
        debug: Dict[str, Any] = {}
        timings: Dict[str, float] = {}
        
        # Prepare conversation context (auto-load, cache, compress)
        conversation_history = await self._prepare_conversation_context(
            conversation_id, conversation_history, session
        )
        
        if conversation_history:
            debug["conversation_messages_used"] = len(conversation_history)
        
        # Query length validation
        if len(query) > settings.MAX_QUERY_LENGTH:
            logger.warning(f"Query too long: {len(query)} chars (max {settings.MAX_QUERY_LENGTH})")
            return {
                "error": f"Query too long. Maximum {settings.MAX_QUERY_LENGTH} characters allowed.",
                "query_length": len(query),
            }
        
        # Validate output format if provided
        if output_format and not validate_output_format(output_format):
            return {
                "error": f"Invalid output format: {output_format}",
                "valid_formats": ["summary", "decisions", "tasks", "insights", "email", "whatsapp", "slides", "infographic"],
            }

        # 0️⃣ Query rewriting for follow-up queries
        retrieval_query = query
        rewrite_info = None
        if conversation_history:
            rewrite_start = time.time()
            rewrite_result = await self.query_rewriter.rewrite_if_needed(
                query=query,
                conversation_history=conversation_history,
            )
            rewrite_info = rewrite_result
            if rewrite_result["was_rewritten"]:
                retrieval_query = rewrite_result["rewritten_query"]
                logger.info(f"Query rewritten: '{query}' → '{retrieval_query}'")
            timings["query_rewrite_ms"] = round((time.time() - rewrite_start) * 1000, 2)

        # 1️⃣ Retrieve with confidence scoring (using potentially rewritten query)
        retrieval_start = time.time()
        try:
            # Build metadata filter if doc_type specified or document_ids present
            metadata_filter = None
            if (doc_type and doc_type != "all") or document_ids:
                metadata_filter = MetadataFilter(
                    doc_type=doc_type if doc_type != "all" else None,
                    document_ids=document_ids
                )
            
            retrieval_result = await self.retrieval.search_with_confidence(
                query=retrieval_query, 
                limit=5,
                metadata_filter=metadata_filter,
            )
            chunks = retrieval_result["chunks"]
            confidence = retrieval_result["confidence"]
            query_analysis = retrieval_result["query_analysis"]
            disclaimer = retrieval_result["disclaimer"]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "error": "Failed to search knowledge base. Please try again.",
                "detail": str(e) if verbose else None,
            }
        timings["retrieval_ms"] = round((time.time() - retrieval_start) * 1000, 2)
        
        # Record retrieval metrics
        metrics.record_retrieval(
            latency_ms=timings["retrieval_ms"],
            chunks_found=len(chunks),
        )
        
        debug["retrieved_chunks"] = len(chunks)
        debug["meetings"] = list({c["title"] for c in chunks if c.get("title")})
        debug["query_intent"] = query_analysis["intent"]
        debug["confidence"] = confidence
        if rewrite_info and rewrite_info["was_rewritten"]:
            debug["query_rewrite"] = rewrite_info

        if not chunks:
            # Record e2e for no-result response
            metrics.record_e2e_request(latency_ms=(time.time() - start_time) * 1000)
            return {
                "answer": "I don't have a record of that discussion.",
                "sources": [],
                "output_format": output_format,
                "confidence": confidence,
                **({"debug": debug} if verbose else {}),
            }

        # 2️⃣ Check response cache (if enabled)
        cache_hit = False
        if use_cache:
            cache_start = time.time()
            query_vector = await self.retrieval._get_embedding(query)
            
            async def generate_response():
                return await self._generate_response(query, chunks, output_format, verbose, conversation_history=conversation_history)
            
            response, cache_hit = await self._response_cache.get_or_generate(
                query=query,
                query_vector=query_vector,
                generator_fn=generate_response,
            )
            timings["cache_check_ms"] = round((time.time() - cache_start) * 1000, 2)
            
            if cache_hit:
                logger.info(f"Response cache HIT for query: {query[:50]}...")
                timings["total_ms"] = round((time.time() - start_time) * 1000, 2)
                response["timings"] = timings
                response["cache_hit"] = True
                # Record cache hit metrics
                metrics.record_cache_access(hit=True, cache_type="response")
                metrics.record_e2e_request(latency_ms=timings["total_ms"])
                return response
        
        # Record cache miss
        metrics.record_cache_access(hit=False, cache_type="response")
        
        # 3️⃣ Generate new response
        response = await self._generate_response(query, chunks, output_format, verbose, timings, conversation_history=conversation_history)
        
        # Add confidence and timing info
        response["confidence"] = confidence
        response["query_analysis"] = query_analysis
        if disclaimer:
            response["disclaimer"] = disclaimer
        
        timings["total_ms"] = round((time.time() - start_time) * 1000, 2)
        response["timings"] = timings
        response["cache_hit"] = False
        
        # Record e2e metrics
        metrics.record_e2e_request(
            latency_ms=timings["total_ms"],
            error="error" in response,
        )
        
        if verbose:
            response["debug"] = debug

        return response
    
    async def _generate_response(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        output_format: Optional[str],
        verbose: bool,
        timings: Optional[Dict[str, float]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate LLM response from chunks."""
        if timings is None:
            timings = {}
        
        # Build prompt with output format instructions and conversation history
        prompt_start = time.time()
        prompt = self.prompt_builder.build_chat_prompt(
            query, 
            chunks, 
            output_format=output_format,
            conversation_history=conversation_history,
        )
        timings["prompt_build_ms"] = round((time.time() - prompt_start) * 1000, 2)
        
        # Log prompt size for debugging
        prompt_tokens_estimate = len(prompt) // 4  # Rough estimate: 1 token ≈ 4 chars
        logger.info(f"Prompt size: {len(prompt)} chars (~{prompt_tokens_estimate} tokens)")

        # Generate
        generation_start = time.time()
        try:
            raw = await self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Record LLM error
            metrics.record_llm_request(
                latency_ms=(time.time() - generation_start) * 1000,
                tokens_in=prompt_tokens_estimate,
                error=True,
            )
            return {
                "error": "Failed to generate response. Please try again.",
                "detail": str(e) if verbose else None,
            }
        timings["generation_ms"] = round((time.time() - generation_start) * 1000, 2)
        
        # Log generation performance
        gen_time_sec = timings["generation_ms"] / 1000
        response_tokens_estimate = len(raw) // 4
        tokens_per_sec = response_tokens_estimate / gen_time_sec if gen_time_sec > 0 else 0
        logger.info(f"LLM generation: {gen_time_sec:.1f}s, ~{response_tokens_estimate} tokens, ~{tokens_per_sec:.1f} tok/s")
        
        # Record LLM metrics
        metrics.record_llm_request(
            latency_ms=timings["generation_ms"],
            tokens_in=prompt_tokens_estimate,
            tokens_out=response_tokens_estimate,
        )

        # Validate output for quality and safety
        validated = self.validator.validate(raw)
        answer = validated["answer"]
        
        # Log validation warnings if any
        if validated.get("warnings"):
            logger.info(f"Validation warnings: {validated['warnings']}")

        # Build sources with enhanced attribution
        sources = self._build_sources(chunks)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "output_format": output_format,
            "prompt_size": len(prompt),  # For debugging slow responses
        }
    
    def _build_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build source attribution with deduplication and normalized scores."""
        sources = []
        seen_chunks = set()
        
        # Normalize scores for display
        raw_scores = [c.get("score", 0) for c in chunks]
        max_score = max(raw_scores) if raw_scores else 1
        min_score = min(raw_scores) if raw_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1
        
        for c in chunks:
            chunk_text = c.get("text", "")[:1000]
            
            if chunk_text in seen_chunks:
                continue
            seen_chunks.add(chunk_text)
            
            # Normalize score: top results get ~85-95%, lower results get ~50-70%
            raw_score = c.get("score", 0)
            if score_range > 0 and max_score > 0:
                normalized_score = 0.50 + (0.45 * (raw_score - min_score) / score_range)
            else:
                normalized_score = min(0.90, max(0.50, raw_score * 3))
            
            sources.append({
                "title": c.get("title"),
                "date": c.get("date"),
                "transcript_id": c.get("transcript_id"),
                "speakers": c.get("speakers", []),
                "text_preview": chunk_text,
                "relevance_score": round(normalized_score, 3),
                "search_source": c.get("search_source", "vector"),
            })
        
        return sources
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            "response_cache": self._response_cache.get_stats(),
            "embedding_cache": self.retrieval.get_cache_stats(),
        }
    
    async def _execute_tool_stream(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_query: str,
    ) -> AsyncIterator[str]:
        """
        Execute a tool and stream progress/results.
        
        Yields JSON events:
        - tool_progress: {"type": "tool_progress", "tool": "...", "status": "...", "message": "..."}
        - tool_result: {"type": "tool_result", "tool": "...", "status": "complete", "data": {...}}
        - tool_error: {"type": "tool_error", "tool": "...", "error": "..."}
        """
        import time
        start_time = time.time()
        
        try:
            if tool_name == "infographic_generator":
                # Import infographic service
                from app.services.infographic_service import InfographicService, InfographicStyle
                
                yield json.dumps({
                    "type": "tool_progress",
                    "tool": tool_name,
                    "status": "processing",
                    "message": "Generating infographic...",
                })
                
                service = InfographicService()
                
                # Map style string to enum
                style_str = tool_args.get("style", "modern")
                try:
                    style = InfographicStyle(style_str)
                except ValueError:
                    style = InfographicStyle.MODERN
                
                # Generate infographic (without DB persistence for now)
                result = await service.generate(
                    request=tool_args.get("request", user_query),
                    topic=tool_args.get("topic"),
                    style=style,
                    doc_type=tool_args.get("doc_type"),
                    persist=False,  # Don't persist to DB in chat context
                )
                
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                
                if "error" in result:
                    yield json.dumps({
                        "type": "tool_error",
                        "tool": tool_name,
                        "error": result["error"],
                    })
                else:
                    yield json.dumps({
                        "type": "tool_result",
                        "tool": tool_name,
                        "status": "complete",
                        "data": {
                            "structured_data": result.get("structured_data"),
                            "image": result.get("image"),
                            "sources": result.get("sources", []),
                            "timing_ms": elapsed_ms,
                        },
                    })
                    
            elif tool_name == "content_writer":
                # Import ghostwriter service
                from app.services.ghostwriter_service import GhostwriterService
                
                yield json.dumps({
                    "type": "tool_progress",
                    "tool": tool_name,
                    "status": "processing",
                    "message": f"Writing {tool_args.get('content_type', 'content')}...",
                })
                
                service = GhostwriterService()
                
                result = await service.generate(
                    content_type=tool_args.get("content_type", "email"),
                    request=tool_args.get("request", user_query),
                    topic=tool_args.get("topic"),
                    tone=tool_args.get("tone"),
                    doc_type=tool_args.get("doc_type"),
                )
                
                elapsed_ms = round((time.time() - start_time) * 1000, 2)
                
                if "error" in result:
                    yield json.dumps({
                        "type": "tool_error",
                        "tool": tool_name,
                        "error": result["error"],
                    })
                else:
                    yield json.dumps({
                        "type": "tool_result",
                        "tool": tool_name,
                        "status": "complete",
                        "data": {
                            "content": result.get("content"),
                            "content_type": result.get("content_type"),
                            "sources": result.get("sources", []),
                            "timing_ms": elapsed_ms,
                        },
                    })
            else:
                yield json.dumps({
                    "type": "tool_error",
                    "tool": tool_name,
                    "error": f"Unknown tool: {tool_name}",
                })
                
        except Exception as e:
            logger.error(f"[TOOL] Error executing {tool_name}: {e}")
            yield json.dumps({
                "type": "tool_error",
                "tool": tool_name,
                "error": str(e),
            })

    async def _get_full_document_content(self, document_id: str) -> Optional[str]:
        """Fetch full document content from vector store for inclusion in chat context."""
        try:
            from app.vectorstore.qdrant import QdrantStore
            store = QdrantStore()
            
            # Query for all chunks belonging to this document
            results = store.client.scroll(
                collection_name=settings.QDRANT_COLLECTION_DOCUMENTS,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1000,  # Get all chunks
                with_payload=True,
                with_vectors=False,
            )
            
            points = results[0] if results else []
            
            if not points:
                logger.warning(f"No chunks found for document {document_id}")
                return None
            
            # Reconstruct full document from chunks in order
            chunks_by_index = {}
            for point in points:
                payload = point.payload or {}
                chunk_index = payload.get("chunk_index", 0)
                text = payload.get("text", "")
                chunks_by_index[chunk_index] = text
            
            # Sort by index and reconstruct
            sorted_indices = sorted(chunks_by_index.keys())
            full_text = "\n".join([chunks_by_index[idx] for idx in sorted_indices])
            
            logger.info(f"Reconstructed document {document_id}: {len(full_text)} chars from {len(chunks_by_index)} chunks")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to get full document content for {document_id}: {e}")
            return None

    async def answer_stream(
        self, 
        query: str,
        output_format: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[str] = None,
        session: Optional[AsyncSession] = None,
        doc_type: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream answer tokens as they're generated for faster perceived response.
        Yields JSON chunks that can be assembled on the client side.
        
        Args:
            query: User's current question
            output_format: Optional output format template
            conversation_history: List of {"role": "user"|"assistant", "content": str} for context
            conversation_id: ID of conversation to auto-load history from
            session: Database session for loading conversation history
            doc_type: Filter by document type (meeting, email, document, note, all)
            document_ids: Optional list of document IDs to restrict search to
        """
        import time
        stream_start_time = time.time()
        
        # Prepare conversation context (auto-load, cache, compress)
        conversation_history = await self._prepare_conversation_context(
            conversation_id, conversation_history, session
        )
        
        logger.info(f"[STREAM] === Starting streaming response ===")
        logger.info(f"[STREAM] Query: {query[:100]}...")
        logger.info(f"[STREAM] Conversation history: {len(conversation_history) if conversation_history else 0} messages")
        if document_ids:
            logger.info(f"[STREAM] Restricting search to {len(document_ids)} documents")
        
        # 🤖 AGENT ROUTING: Check if this request needs a tool
        agent = get_agent_service()
        try:
            tool_decision = await agent.decide(query)
            logger.info(f"[STREAM] Agent decision: {tool_decision.to_dict()}")
            
            if tool_decision.is_tool_use():
                # This is a tool request - stream tool events instead of normal RAG
                logger.info(f"[STREAM] Routing to tool: {tool_decision.tool_name}")
                
                # Emit tool_call event to notify frontend
                yield json.dumps({
                    "type": "tool_call",
                    "tool": tool_decision.tool_name,
                    "status": "starting",
                    "args": tool_decision.tool_args,
                    "confidence": tool_decision.confidence,
                })
                
                # Execute the tool and stream results
                async for event in self._execute_tool_stream(
                    tool_name=tool_decision.tool_name,
                    tool_args=tool_decision.tool_args,
                    user_query=query,
                ):
                    yield event
                
                # Tool execution complete - return early
                return
                
        except Exception as e:
            # Agent routing failed - fall back to normal RAG
            logger.warning(f"[STREAM] Agent routing failed, falling back to RAG: {e}")
            # Continue with normal RAG flow below
        
        # 0️⃣ Query rewriting for follow-up queries
        retrieval_query = query
        if conversation_history:
            rewrite_start = time.time()
            rewrite_result = await self.query_rewriter.rewrite_if_needed(
                query=query,
                conversation_history=conversation_history,
            )
            if rewrite_result["was_rewritten"]:
                retrieval_query = rewrite_result["rewritten_query"]
                logger.info(f"[STREAM] Query rewritten: '{query}' → '{retrieval_query}'")
                # Send rewrite info to client
                yield json.dumps({
                    "type": "rewrite",
                    "original": query,
                    "rewritten": retrieval_query,
                })
            rewrite_time_ms = round((time.time() - rewrite_start) * 1000, 2)
            logger.info(f"[STREAM] Query rewrite check completed in {rewrite_time_ms}ms")
        
        # 1️⃣ Retrieve chunks with confidence scoring (using potentially rewritten query)
        retrieval_start = time.time()
        
        # Build metadata filter if doc_type specified or document_ids present
        metadata_filter = None
        if (doc_type and doc_type != "all") or document_ids:
            metadata_filter = MetadataFilter(
                doc_type=doc_type if doc_type != "all" else None,
                document_ids=document_ids
            )
        
        retrieval_result = await self.retrieval.search_with_confidence(
            query=retrieval_query, 
            limit=5,
            metadata_filter=metadata_filter,
        )
        chunks = retrieval_result["chunks"]
        confidence = retrieval_result["confidence"]
        disclaimer = retrieval_result["disclaimer"]
        retrieval_time_ms = round((time.time() - retrieval_start) * 1000, 2)
        
        logger.info(f"[STREAM] Retrieval completed in {retrieval_time_ms}ms, found {len(chunks)} chunks")
        logger.info(f"[STREAM] Confidence: {confidence}")
        
        # 🔄 If specific documents were uploaded, inject their full content as context
        if document_ids:
            logger.info(f"[STREAM] Injecting full content for {len(document_ids)} uploaded documents")
            for doc_id in document_ids:
                full_content = await self._get_full_document_content(doc_id)
                if full_content:
                    # Add as a high-priority chunk at the beginning
                    full_doc_chunk = {
                        "text": full_content[:5000],  # Cap at 5000 chars to avoid overwhelming context
                        "title": f"Uploaded Document (Full)",
                        "score": 1.0,  # Highest priority
                        "source": "document",
                        "document_id": doc_id,
                    }
                    chunks.insert(0, full_doc_chunk)  # Insert at beginning
                    logger.info(f"[STREAM] Injected full document content ({len(full_content)} chars)")
                else:
                    logger.warning(f"[STREAM] Could not retrieve full content for document {doc_id}")
        
        if not chunks:
            logger.info(f"[STREAM] No chunks found, returning default response")
            yield json.dumps({"type": "answer", "content": "I don't have a record of that discussion."})
            return
        
        # Log chunk details
        for i, chunk in enumerate(chunks):
            logger.info(f"[STREAM] Chunk {i+1}: title='{chunk.get('title', 'N/A')}', "
                       f"score={chunk.get('score', 0):.3f}, "
                       f"text_preview='{chunk.get('text', '')[:80]}...'")
        
        # 2️⃣ Send sources first (immediate feedback)
        sources = []
        seen_chunks = set()
        
        # Normalize scores for display: map raw scores to 0-100% scale
        # Raw cosine similarity for conversational text is often low (0.05-0.30)
        # We scale to make top result ~90% and lowest ~50% for retrieved results
        raw_scores = [c.get("score", 0) for c in chunks]
        max_score = max(raw_scores) if raw_scores else 1
        min_score = min(raw_scores) if raw_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1
        
        for c in chunks:
            chunk_text = c.get("text", "")[:1000]
            if chunk_text in seen_chunks:
                continue
            seen_chunks.add(chunk_text)
            
            # Normalize score: top results get ~85-95%, lower results get ~50-70%
            raw_score = c.get("score", 0)
            if score_range > 0 and max_score > 0:
                # Scale to 50-95% range based on relative position
                normalized_score = 0.50 + (0.45 * (raw_score - min_score) / score_range)
            else:
                # If all scores are same, use a default based on raw score
                normalized_score = min(0.90, max(0.50, raw_score * 3))
            
            source = {
                "title": c.get("title"),
                "date": c.get("date"),
                "transcript_id": c.get("transcript_id"),
                "speakers": c.get("speakers", []),
                "text_preview": chunk_text,
                "relevance_score": round(normalized_score, 3),  # Normalized score for display
                "raw_score": round(raw_score, 4),  # Keep raw score for debugging
            }
            sources.append(source)
            logger.debug(f"[STREAM] Source: title='{source['title']}', date='{source['date']}', "
                        f"transcript_id='{source['transcript_id']}', speakers={source['speakers']}, "
                        f"relevance_score={source['relevance_score']} (raw={source['raw_score']})")
        
        logger.info(f"[STREAM] Sending {len(sources)} sources to client")
        if sources:
            logger.info(f"[STREAM] First source sample: {sources[0]}")
        yield json.dumps({"type": "sources", "content": sources})
        
        # 3️⃣ Build prompt with output format instructions AND conversation history
        prompt_start = time.time()
        prompt = self.prompt_builder.build_chat_prompt(
            query, 
            chunks, 
            output_format=output_format,
            conversation_history=conversation_history,
        )
        prompt_time_ms = round((time.time() - prompt_start) * 1000, 2)
        logger.info(f"[STREAM] Prompt built in {prompt_time_ms}ms, length={len(prompt)} chars")
        
        # 4️⃣ Stream LLM response
        generation_start = time.time()
        token_count = 0
        async for token in self.llm.generate_stream(prompt):
            token_count += 1
            # Aggressive markdown stripping to ensure plain text
            clean_token = token.replace('*', '').replace('#', '').replace('`', '')
            if clean_token:
                yield json.dumps({"type": "token", "content": clean_token})
        
        generation_time_ms = round((time.time() - generation_start) * 1000, 2)
        total_time_ms = round((time.time() - stream_start_time) * 1000, 2)
        
        # Send timing metadata at the end
        timing_data = {
            "retrieval_ms": retrieval_time_ms,
            "prompt_ms": prompt_time_ms,
            "generation_ms": generation_time_ms,
            "total_ms": total_time_ms,
            "chunks_used": len(chunks),
            "tokens_generated": token_count,
        }
        yield json.dumps({"type": "timing", "content": timing_data})
        
        # Send confidence data
        confidence_data = {
            "level": confidence["level"],
            "avg_score": confidence["metrics"].get("avg_similarity", 0),
            "top_score": confidence["metrics"].get("top_similarity", 0),
            "chunk_count": confidence["metrics"].get("chunk_count", 0),
            "should_fallback": confidence["level"] in ["low", "very_low"],
        }
        yield json.dumps({"type": "confidence", "content": confidence_data, "disclaimer": disclaimer})
        
        logger.info(f"[STREAM] === Stream complete ===")
        logger.info(f"[STREAM] Timing: retrieval={retrieval_time_ms}ms, generation={generation_time_ms}ms, total={total_time_ms}ms")
        logger.info(f"[STREAM] Stats: chunks_used={len(chunks)}, tokens_generated={token_count}")

    async def trace(self, query: str) -> Dict[str, Any]:
        """
        Agent-style trace (non-streaming).
        """
        trace_steps: List[Dict[str, Any]] = []

        trace_steps.append({"step": "query_received", "query": query})

        chunks = await self.retrieval.search(query=query, limit=6)
        trace_steps.append(
            {
                "step": "retrieval",
                "chunks_found": len(chunks),
                "meetings": [c["title"] for c in chunks],
            }
        )

        prompt = self.prompt_builder.build_chat_prompt(query, chunks)
        trace_steps.append(
            {
                "step": "prompt_built",
                "prompt_preview": prompt[:500] + "...",
            }
        )

        raw = await self.llm.generate(prompt)
        trace_steps.append(
            {
                "step": "llm_generated",
                "raw_preview": raw[:500] + "...",
            }
        )

        validated = self.validator.validate(raw)
        trace_steps.append({"step": "validated"})

        return {
            "query": query,
            "answer": validated["answer"],
            "trace": trace_steps,
        }

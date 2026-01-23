from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.memory_service import MemoryService
from app.services.enhanced_memory import EnhancedMemoryService
from app.services.rag_analytics_service import RAGAnalyticsService
from app.api.deps import (
    get_db, 
    get_optional_user, 
    get_current_user,
    get_conversation_cache,
    get_user_conversations_cache,
    check_rate_limit
)
from app.database.models import User
from app.cache.conversation_cache import ConversationCache, UserConversationsCache

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Global service instance (for non-authenticated requests)
service = ChatService(use_enhanced_retrieval=True)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    verbose: bool = False
    stream: bool = True  # Streaming by default for faster perceived response
    output_format: str | None = Field(
        None,
        description="Output format: summary, decisions, tasks, insights, email, whatsapp, slides, infographic"
    )
    # Conversation memory fields
    conversation_id: str | None = Field(None, description="ID of conversation to continue")
    include_history: bool = Field(True, description="Include conversation history in context")
    # Document type filter
    doc_type: str | None = Field(
        None,
        description="Filter sources by type: meeting, email, document, note, or all"
    )
    # Explicit document context (from uploads)
    document_ids: list[str] | None = Field(None, description="List of specific document IDs to include in context")
    # Metadata for attached documents
    attachments: list[dict] | None = Field(None, description="Metadata for attached documents")


async def get_conversation_service(
    db: AsyncSession = Depends(get_db),
    conv_cache: Optional[ConversationCache] = Depends(get_conversation_cache),
    user_conv_cache: Optional[UserConversationsCache] = Depends(get_user_conversations_cache)
) -> ConversationService:
    """Get conversation service with dependencies."""
    return ConversationService(db, conv_cache, user_conv_cache)


async def get_memory_service(
    db: AsyncSession = Depends(get_db),
    conv_cache: Optional[ConversationCache] = Depends(get_conversation_cache)
) -> MemoryService:
    """Get memory service with dependencies."""
    return MemoryService(db, conv_cache)


def get_chat_service() -> ChatService:
    """Get chat service instance."""
    return service


@router.post("")
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
    conv_cache: Optional[ConversationCache] = Depends(get_conversation_cache),
    user_conv_cache: Optional[UserConversationsCache] = Depends(get_user_conversations_cache),
):
    """
    Answer a user query using RAG. 
    
    - If authenticated with conversation_id: continues existing conversation with memory
    - If authenticated without conversation_id: creates new conversation
    - If not authenticated: works without conversation storage
    
    Set stream=true for faster perceived response.
    """
    # Inject conversation cache into chat service for auto-loading
    if conv_cache:
        service.set_conversation_cache(conv_cache)
    
    logger.info(f"[CHAT] === New Chat Request ===")
    logger.info(f"[CHAT] Query: {req.query[:100]}...")
    logger.info(f"[CHAT] Stream: {req.stream}, User: {current_user.id if current_user else 'anonymous'}")
    logger.info(f"[CHAT] Incoming conversation_id: {req.conversation_id}")
    
    conversation_id = req.conversation_id
    message_id = None
    is_new_conversation = False
    
    # Handle conversation memory for authenticated users
    if current_user:
        conv_service = ConversationService(db, conv_cache, user_conv_cache)
        memory_service = MemoryService(db, conv_cache)
        
        # Create or get conversation
        if not conversation_id:
            # Create new conversation
            is_new_conversation = True
            conversation = await conv_service.create_conversation(
                user_id=current_user.id,
                title=None,  # Will be auto-generated from first message
            )
            conversation_id = conversation.id
            logger.info(f"[CHAT] Created NEW conversation: {conversation_id}")
        else:
            # Verify conversation ownership
            logger.info(f"[CHAT] Looking up EXISTING conversation: {conversation_id}")
            conversation = await conv_service.get_conversation(conversation_id, current_user.id)
            if not conversation:
                logger.warning(f"[CHAT] Conversation NOT FOUND: {conversation_id}")
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found"
                )
            logger.info(f"[CHAT] Found existing conversation: id={conversation.id}, title={conversation.title}, message_count={conversation.message_count}")
        
        # Prepare metadata for user message (including attachments)
        user_msg_metadata = {}
        if req.attachments:
            user_msg_metadata["attachments"] = req.attachments
            
        # Add user message to conversation
        logger.info(f"[CHAT] Adding user message to conversation: {conversation_id}")
        user_message = await conv_service.add_message(
            conversation_id=conversation_id,
            user_id=current_user.id,
            role="user",
            content=req.query,
            metadata=user_msg_metadata
        )
        logger.info(f"[CHAT] User message added: id={user_message.id}")
        
        # Persist attachments to conversation if present
        if req.attachments:
            await conv_service.add_attachments_to_conversation(
                conversation_id=conversation_id,
                attachments=req.attachments
            )
        
        # Handle document_ids persistence in conversation
        # 1. Get existing IDs
        existing_doc_ids = await conv_service.get_conversation_document_ids(conversation_id)
        
        # 2. Merge with new IDs if present
        if req.document_ids:
            # Store new document_ids in conversation metadata
            await conv_service.add_document_ids_to_conversation(
                conversation_id=conversation_id,
                document_ids=req.document_ids
            )
            # Merge for current request context (prevent race condition by not re-reading)
            effective_document_ids = list(set(existing_doc_ids + req.document_ids))
            logger.info(f"[CHAT] Stored {len(req.document_ids)} document_ids in conversation metadata. Effective: {effective_document_ids}")
        else:
            effective_document_ids = existing_doc_ids
            logger.info(f"[CHAT] No new document_ids. Existing effective: {effective_document_ids}")
            
        if effective_document_ids:
            logger.info(f"[CHAT] Using {len(effective_document_ids)} document_ids from conversation context: {effective_document_ids}")
    else:
        effective_document_ids = req.document_ids or []
        logger.info(f"[CHAT] Anonymous user - no conversation storage")
    
    try:
        if req.stream:
            logger.info(f"[CHAT] Starting streaming response for conversation: {conversation_id}")
            
            # Load conversation history for context if authenticated and requested
            conversation_history = []
            enhanced_context = None
            if current_user and conversation_id and req.include_history:
                try:
                    logger.info(f"[CHAT] Loading conversation history for streaming context")
                    
                    # Use enhanced memory if enabled
                    if settings.SEMANTIC_MEMORY_SEARCH_ENABLED:
                        logger.info(f"[CHAT] Using enhanced semantic memory search")
                        enhanced_memory = EnhancedMemoryService(db, conv_cache)
                        enhanced_context = await enhanced_memory.get_enhanced_context(
                            conversation_id=conversation_id,
                            current_query=req.query,
                        )
                        conversation_history = enhanced_context.messages
                        
                        # Log enhanced context info
                        if enhanced_context.relevant_history:
                            logger.info(f"[CHAT] Found {len(enhanced_context.relevant_history)} semantically relevant past messages")
                        if enhanced_context.entities:
                            logger.info(f"[CHAT] Extracted entities: {list(enhanced_context.entities.keys())}")
                        if enhanced_context.topic_summary:
                            logger.info(f"[CHAT] Topic summary available for context")
                    else:
                        # Fallback to standard memory service
                        memory_service = MemoryService(db, conv_cache)
                        conversation_history = await memory_service.get_context_for_chat(conversation_id)
                    
                    logger.info(f"[CHAT] Loaded {len(conversation_history)} history messages for streaming")
                except Exception as e:
                    logger.warning(f"[CHAT] Failed to load conversation history: {e}")
                    conversation_history = []
            
            # Streaming response - tokens arrive as they're generated
            async def generate():
                # Send conversation_id first if authenticated
                if current_user and conversation_id:
                    import json
                    logger.info(f"[CHAT] Sending meta with conversation_id: {conversation_id}")
                    meta_data = {'type': 'meta', 'conversation_id': conversation_id}
                    # Include the user message ID so frontend can update its temp ID
                    if user_message:
                        meta_data['user_message_id'] = user_message.id
                    yield f"data: {json.dumps(meta_data)}\n\n"
                
                full_response = ""
                sources = []
                timing_data = {}
                tool_result = None
                tool_name = None
                
                async for chunk in service.answer_stream(
                    req.query, 
                    output_format=req.output_format,
                    conversation_history=conversation_history,
                    conversation_id=conversation_id if current_user else None,
                    session=db if current_user else None,
                    user_id=str(current_user.id) if current_user else None,
                    doc_type=req.doc_type,
                    document_ids=effective_document_ids if effective_document_ids else None,
                ):
                    print(f"DEBUG: Got chunk: {chunk[:50]}...", flush=True)
                    yield f"data: {chunk}\n\n"
                    
                    # Collect response for storage
                    import json
                    chunk_data = json.loads(chunk)
                    if chunk_data.get("type") == "token":
                        full_response += chunk_data.get("content", "")
                    elif chunk_data.get("type") == "sources":
                        sources = chunk_data.get("content", [])
                    elif chunk_data.get("type") == "timing":
                        timing_data = chunk_data.get("content", {})
                    elif chunk_data.get("type") == "tool_result":
                        tool_result = chunk_data.get("data")
                        tool_name = chunk_data.get("tool")
                        if tool_result and "sources" in tool_result:
                            sources = tool_result.get("sources", [])
                    elif chunk_data.get("type") == "tool_call":
                        tool_name = chunk_data.get("tool")
                
                # Log timing summary
                if timing_data:
                    logger.info(f"[CHAT] Response timing: retrieval={timing_data.get('retrieval_ms')}ms, "
                               f"generation={timing_data.get('generation_ms')}ms, "
                               f"total={timing_data.get('total_ms')}ms, "
                               f"chunks={timing_data.get('chunks_used')}, "
                               f"tokens={timing_data.get('tokens_generated')}")
                
                # Store assistant response for authenticated users with FULL sources
                if current_user and conversation_id and (full_response or tool_result):
                    try:
                        logger.info(f"[CHAT] Storing assistant response to conversation: {conversation_id}")
                        logger.info(f"[CHAT] Storing {len(sources)} sources with the message")
                        conv_service = ConversationService(db, conv_cache, user_conv_cache)
                        
                        # Store complete source information for later retrieval
                        sources_to_store = [
                            {
                                "title": s.get("title"),
                                "date": s.get("date"),
                                "transcript_id": s.get("transcript_id"),
                                "speakers": s.get("speakers", []),
                                "text": s.get("text") or s.get("text_preview"),
                                "text_preview": s.get("text_preview") or s.get("text"),
                                "relevance_score": s.get("relevance_score"),
                                "meeting_category": s.get("meeting_category"),
                                "category_confidence": s.get("category_confidence"),
                            }
                            for s in sources
                        ]
                        
                        # Ensure content is not empty for DB constraint
                        content_to_store = full_response
                        if not content_to_store and tool_result:
                            if tool_name == "infographic_generator":
                                headline = tool_result.get("structured_data", {}).get("headline", "")
                                content_to_store = f"Generated infographic: {headline}" if headline else "Generated infographic."
                            elif tool_name == "content_writer":
                                content_to_store = "Generated content."
                            else:
                                content_to_store = "Tool execution complete."

                        assistant_msg = await conv_service.add_message(
                            conversation_id=conversation_id,
                            user_id=current_user.id,
                            role="assistant",
                            content=content_to_store,
                            sources=sources_to_store,
                            metadata={
                                "timing": timing_data,
                                "chunks_used": len(sources),
                                "tool_result": tool_result,
                                "tool_name": tool_name,
                            }
                        )
                        logger.info(f"[CHAT] Assistant message stored: id={assistant_msg.id}, "
                                   f"response_length={len(full_response)}, sources_count={len(sources_to_store)}")
                    except Exception as e:
                        logger.error(f"[CHAT] Failed to store assistant message: {e}")
                
                # Log RAG analytics for streaming requests
                try:
                    rag_analytics = RAGAnalyticsService(db)
                    await rag_analytics.log_interaction(
                        query=req.query,
                        user_id=current_user.id if current_user else None,
                        conversation_id=conversation_id,
                        sources=sources if sources else None,
                        answer_length=len(full_response),
                        output_format=req.output_format,
                        chunks_used=timing_data.get('chunks_used', 0),
                        retrieval_latency_ms=timing_data.get('retrieval_ms'),
                        generation_latency_ms=timing_data.get('generation_ms'),
                        total_latency_ms=timing_data.get('total_ms'),
                        success=True,
                    )
                    logger.info(f"[CHAT] RAG analytics logged for conversation: {conversation_id}")
                except Exception as e:
                    logger.warning(f"[CHAT] Failed to log RAG analytics (stream): {e}")
                
                logger.info(f"[CHAT] === Chat Request Complete === conversation_id={conversation_id}")
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            logger.info(f"[CHAT] Starting non-streaming response for conversation: {conversation_id}")
            # Standard response - wait for complete answer
            
            # Get conversation history for context if authenticated
            history_context = []
            if current_user and conversation_id and req.include_history:
                logger.info(f"[CHAT] Loading conversation history for context")
                
                # Use enhanced memory if enabled
                if settings.SEMANTIC_MEMORY_SEARCH_ENABLED:
                    logger.info(f"[CHAT] Using enhanced semantic memory search for non-streaming")
                    enhanced_memory = EnhancedMemoryService(db, conv_cache)
                    enhanced_context = await enhanced_memory.get_enhanced_context(
                        conversation_id=conversation_id,
                        current_query=req.query,
                    )
                    history_context = enhanced_context.messages
                else:
                    memory_service = MemoryService(db, conv_cache)
                    history_context = await memory_service.get_context_for_chat(conversation_id)
                    
                logger.info(f"[CHAT] Loaded {len(history_context)} history items")
            
            result = await service.answer(
                query=req.query,
                verbose=req.verbose,
                output_format=req.output_format,
                conversation_history=history_context,
                conversation_id=conversation_id if current_user else None,
                session=db if current_user else None,
                doc_type=req.doc_type,
                document_ids=effective_document_ids if effective_document_ids else None,
            )
            
            # Check if service returned an error
            if "error" in result:
                logger.error(f"[CHAT] Service error: {result['error']}")
                raise HTTPException(status_code=400, detail=result["error"])
            
            # Store assistant response for authenticated users
            if current_user and conversation_id:
                try:
                    conv_service = ConversationService(db, conv_cache, user_conv_cache)
                    assistant_message = await conv_service.add_message(
                        conversation_id=conversation_id,
                        user_id=current_user.id,
                        role="assistant",
                        content=result.get("answer", ""),
                        sources=result.get("sources"),
                        confidence_score=result.get("confidence"),
                    )
                    message_id = assistant_message.id if assistant_message else None
                except Exception as e:
                    logger.error(f"Failed to store assistant message: {e}")
            
            # Log RAG analytics
            try:
                rag_analytics = RAGAnalyticsService(db)
                timings = result.get("timings", {})
                await rag_analytics.log_interaction(
                    query=req.query,
                    user_id=current_user.id if current_user else None,
                    conversation_id=conversation_id,
                    query_intent=result.get("query_analysis", {}).get("intent") if result.get("query_analysis") else None,
                    chunks_retrieved=result.get("debug", {}).get("retrieved_chunks", 0) if result.get("debug") else 0,
                    sources=result.get("sources"),
                    answer_length=len(result.get("answer", "")),
                    output_format=req.output_format,
                    confidence_score=result.get("confidence", {}).get("score") if isinstance(result.get("confidence"), dict) else None,
                    confidence_level=result.get("confidence", {}).get("level") if isinstance(result.get("confidence"), dict) else None,
                    retrieval_latency_ms=timings.get("retrieval_ms"),
                    generation_latency_ms=timings.get("generation_ms"),
                    total_latency_ms=timings.get("total_ms"),
                    cache_hit=result.get("cache_hit", False),
                    success=True,
                )
            except Exception as e:
                logger.warning(f"Failed to log RAG analytics: {e}")
            
            # Add conversation tracking to response
            result["conversation_id"] = conversation_id
            result["message_id"] = message_id
            
            return result
            
    except RuntimeError as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Chat service error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


@router.post("/{conversation_id}/messages/{message_id}/edit", response_class=StreamingResponse)
async def edit_message_and_regenerate(
    conversation_id: str,
    message_id: str,
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    conv_service: ConversationService = Depends(get_conversation_service),
    chat_service: ChatService = Depends(get_chat_service),
    _: None = Depends(check_rate_limit)
):
    """
    Edit a user message and regenerate the response.
    This will:
    1. Update the message content
    2. Soft-delete all subsequent messages (rewind history)
    3. Stream a new response based on the updated context
    """
    logger.info(f"[CHAT] Edit request for message {message_id} in conversation {conversation_id}")
    
    # 1. Edit message and rewind history
    updated_message = await conv_service.edit_message(
        conversation_id=conversation_id,
        message_id=message_id,
        user_id=current_user.id,
        new_content=req.query
    )
    
    if not updated_message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Message not found or access denied"
        )

    # 2. Get updated conversation history for context
    # Use existing helper to get history
    history = await conv_service.msg_repo.get_context_messages(conversation_id, limit=10)
    messages, _ = history
    
    # Format for RAG context (exclude the current message we just edited, it's the query)
    conversation_history = [
        {"role": m.role, "content": m.content} 
        for m in messages 
        if m.id != message_id 
    ]
    
    # 3. Stream new response
    # Re-use document IDs from conversation if available
    doc_ids = await conv_service.get_conversation_document_ids(conversation_id)
    
    # If request includes NEW doc IDs, merge them (though usually edit uses existing context)
    if req.document_ids:
        doc_ids = list(set(doc_ids + req.document_ids))
    
    # Setup generator for streaming response with full RAG + History
    async def generate_edit_stream():
        # Send conversation_id first
        import json
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id, 'is_edit': True})}\n\n"
        
        full_response = ""
        sources = []
        timing_data = {}
        tool_result = None
        tool_name = None
        
        # Stream response
        async for chunk in chat_service.answer_stream(
            query=req.query,
            output_format=req.output_format,
            conversation_history=conversation_history,
            doc_type=req.doc_type,
            document_ids=doc_ids if doc_ids else None
        ):
            yield f"data: {chunk}\n\n"
            
            # Collect data for storage
            chunk_data = json.loads(chunk)
            if chunk_data.get("type") == "token":
                full_response += chunk_data.get("content", "")
            elif chunk_data.get("type") == "sources":
                sources = chunk_data.get("content", [])
            elif chunk_data.get("type") == "timing":
                timing_data = chunk_data.get("content", {})
            elif chunk_data.get("type") == "tool_result":
                tool_result = chunk_data.get("data")
                tool_name = chunk_data.get("tool")
                if tool_result and "sources" in tool_result:
                    sources = tool_result.get("sources", [])
            elif chunk_data.get("type") == "tool_call":
                tool_name = chunk_data.get("tool")

        # Store assistant response
        if full_response or tool_result:
            try:
                # Prepare sources for storage
                sources_to_store = [
                    {
                        "title": s.get("title"),
                        "date": s.get("date"),
                        "transcript_id": s.get("transcript_id"),
                        "speakers": s.get("speakers", []),
                        "text": s.get("text") or s.get("text_preview"),
                        "text_preview": s.get("text_preview") or s.get("text"),
                        "relevance_score": s.get("relevance_score"),
                        "meeting_category": s.get("meeting_category"),
                        "category_confidence": s.get("category_confidence"),
                    }
                    for s in sources
                ]
                
                content_to_store = full_response
                if not content_to_store and tool_result:
                     content_to_store = "Tool execution complete."

                await conv_service.add_message(
                    conversation_id=conversation_id,
                    user_id=current_user.id,
                    role="assistant",
                    content=content_to_store,
                    sources=sources_to_store,
                    metadata={
                        "timing": timing_data,
                        "chunks_used": len(sources),
                        "tool_result": tool_result,
                        "tool_name": tool_name,
                        "is_regenerated": True
                    }
                )
            except Exception as e:
                logger.error(f"[CHAT] Failed to store regenerated message: {e}")

    return StreamingResponse(
        generate_edit_stream(),
        media_type="text/event-stream"
    )


@router.post("/trace")
async def chat_trace(req: ChatRequest):
    """Get detailed trace of chat processing including chunks and prompt."""
    try:
        return await service.trace(query=req.query)
    except RuntimeError as e:
        logger.error(f"Chat trace failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Chat service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat trace: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


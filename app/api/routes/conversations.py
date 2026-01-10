"""
Conversation routes.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from app.api.deps import (
    get_db, 
    get_current_user, 
    get_conversation_cache,
    get_user_conversations_cache,
    check_rate_limit
)
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationWithMessages,
    ConversationListResponse,
    ConversationSearchRequest,
    ConversationExportResponse
)
from app.schemas.message import MessageResponse, MessageListResponse, MessageRole, SourceReference
from app.services.conversation_service import ConversationService
from app.database.models import User
from app.cache.conversation_cache import ConversationCache, UserConversationsCache

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conversation_service(
    db: AsyncSession = Depends(get_db),
    conv_cache: Optional[ConversationCache] = Depends(get_conversation_cache),
    user_conv_cache: Optional[UserConversationsCache] = Depends(get_user_conversations_cache)
) -> ConversationService:
    """Get conversation service with dependencies."""
    return ConversationService(db, conv_cache, user_conv_cache)


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
    _: None = Depends(check_rate_limit)
):
    """Create a new conversation."""
    conversation = await service.create_conversation(
        user_id=str(current_user.id),
        title=request.title,
        metadata=request.metadata
    )
    
    return ConversationResponse(
        id=str(conversation.id),
        user_id=str(conversation.user_id),
        title=conversation.title,
        summary=conversation.summary,
        message_count=conversation.message_count,
        metadata=dict(conversation.metadata_) if conversation.metadata_ else None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """List user's conversations with pagination."""
    logger.info(f"[CONVERSATIONS] Listing conversations for user: {current_user.id}, page={page}, page_size={page_size}")
    result = await service.list_conversations(
        user_id=str(current_user.id),
        page=page,
        page_size=page_size
    )
    logger.info(f"[CONVERSATIONS] Found {result.total} conversations, returning {len(result.conversations)} on page {page}")
    return result


@router.get("/search", response_model=ConversationListResponse)
async def search_conversations(
    query: str = Query(..., min_length=1, max_length=500),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Search user's conversations by title or content."""
    return await service.search_conversations(
        user_id=str(current_user.id),
        query=query,
        page=page,
        page_size=page_size
    )


@router.get("/frequent-questions")
async def get_frequent_questions(
    limit: int = Query(4, ge=1, le=10),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get frequently asked questions based on the user's message history.
    Returns the top N most common user messages/questions.
    """
    from sqlalchemy import func, select
    from app.database.models import Message, Conversation
    
    try:
        # Query to get user messages from the user's conversations
        # Group by content and count occurrences
        query = (
            select(
                Message.content,
                func.count(Message.id).label('count')
            )
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == current_user.id,
                Conversation.is_deleted == False,
                Message.is_deleted == False,
                Message.role == 'user'
            )
            .group_by(Message.content)
            .order_by(func.count(Message.id).desc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        # Extract just the question texts
        questions = [row.content for row in rows if row.content and len(row.content) < 200]
        
        logger.info(f"[CONVERSATIONS] Frequent questions for user {current_user.id}: {len(questions)} found")
        
        return {"questions": questions}
    except Exception as e:
        logger.error(f"[CONVERSATIONS] Error fetching frequent questions: {e}")
        return {"questions": []}


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Get a specific conversation."""
    conversation = await service.get_conversation(
        conversation_id=conversation_id,
        user_id=str(current_user.id)
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return ConversationResponse(
        id=str(conversation.id),
        user_id=str(conversation.user_id),
        title=conversation.title,
        summary=conversation.summary,
        message_count=conversation.message_count,
        metadata=dict(conversation.metadata_) if conversation.metadata_ else None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("/{conversation_id}/full", response_model=ConversationWithMessages)
async def get_conversation_with_messages(
    conversation_id: str,
    message_limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Get a conversation with its messages."""
    logger.info(f"[CONVERSATIONS] Getting conversation with messages: {conversation_id} for user: {current_user.id}")
    result = await service.get_conversation_with_messages(
        conversation_id=conversation_id,
        user_id=str(current_user.id),
        message_limit=message_limit
    )
    
    if not result:
        logger.warning(f"[CONVERSATIONS] Conversation not found: {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    conversation = result["conversation"]
    messages = result["messages"]
    
    logger.info(f"[CONVERSATIONS] Found conversation: id={conversation.id}, title={conversation.title}, messages={len(messages)}")
    
    return ConversationWithMessages(
        id=str(conversation.id),
        user_id=str(conversation.user_id),
        title=conversation.title,
        summary=conversation.summary,
        message_count=conversation.message_count,
        metadata=dict(conversation.metadata_) if conversation.metadata_ else None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageResponse(
                id=str(m.id),
                conversation_id=str(m.conversation_id),
                role=m.role,
                content=m.content,
                sources=m.sources,
                confidence_score=m.confidence_score,
                metadata=dict(m.metadata_) if m.metadata_ else None,
                created_at=m.created_at
            )
            for m in messages
        ]
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    update: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Update a conversation."""
    conversation = await service.update_conversation(
        conversation_id=conversation_id,
        user_id=str(current_user.id),
        update=update
    )
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return ConversationResponse(
        id=str(conversation.id),
        user_id=str(conversation.user_id),
        title=conversation.title,
        summary=conversation.summary,
        message_count=conversation.message_count,
        metadata=dict(conversation.metadata_) if conversation.metadata_ else None,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Delete a conversation (soft delete)."""
    success = await service.delete_conversation(
        conversation_id=conversation_id,
        user_id=str(current_user.id)
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )


@router.get("/{conversation_id}/messages", response_model=MessageListResponse)
async def get_conversation_messages(
    conversation_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Get messages for a conversation with pagination."""
    result = await service.get_messages(
        conversation_id=conversation_id,
        user_id=str(current_user.id),
        page=page,
        page_size=page_size
    )
    
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages, total, has_more = result
    
    # Helper to convert sources dict to list of SourceReference
    def convert_sources(sources_data):
        if sources_data is None:
            return None
        if isinstance(sources_data, list):
            return [SourceReference(**s) if isinstance(s, dict) else s for s in sources_data]
        return None
    
    return MessageListResponse(
        messages=[
            MessageResponse(
                id=str(m.id),
                conversation_id=str(m.conversation_id),
                role=MessageRole(m.role),
                content=m.content,
                sources=convert_sources(m.sources),
                confidence_score=m.confidence_score,
                metadata=dict(m.metadata_) if m.metadata_ else None,
                created_at=m.created_at
            )
            for m in messages
        ],
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more
    )


@router.get("/{conversation_id}/export", response_model=ConversationExportResponse)
async def export_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """Export a conversation for download."""
    result = await service.export_conversation(
        conversation_id=conversation_id,
        user_id=str(current_user.id)
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return ConversationExportResponse(**result)

"""
Tests for conversation memory features in ChatService.

Tests:
1. Auto-load conversation history
2. Smart context window (token limits)
3. Conversation caching
4. Message summarization for long conversations
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.chat_service import ChatService
from app.cache.conversation_cache import ConversationCache


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_conv_cache():
    """Mock conversation cache."""
    cache = AsyncMock(spec=ConversationCache)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def chat_service(mock_conv_cache):
    """Create ChatService instance with mocked dependencies."""
    with patch('app.services.chat_service.RetrievalService'):
        with patch('app.services.chat_service.OllamaClient'):
            with patch('app.services.chat_service.PromptBuilder'):
                with patch('app.services.chat_service.OutputValidator'):
                    with patch('app.services.chat_service.ConfidenceScorer'):
                        with patch('app.services.chat_service.get_query_rewriter'):
                            service = ChatService()
                            service.set_conversation_cache(mock_conv_cache)
                            return service


@pytest.mark.asyncio
async def test_load_conversation_history_from_db(chat_service, mock_session, mock_conv_cache):
    """Test #1: Auto-load conversation history from database."""
    conversation_id = "conv-123"
    
    # Mock ConversationCache.get_messages (not found in cache)
    mock_conv_cache.get_messages = AsyncMock(return_value=None)
    mock_conv_cache.set_messages = AsyncMock()
    
    # Mock MessageRepository
    mock_messages = [
        MagicMock(role="user", content="What's the weather?"),
        MagicMock(role="assistant", content="I don't have weather data."),
        MagicMock(role="user", content="Tell me about meetings."),
    ]
    
    with patch('app.repositories.message_repository.MessageRepository') as MockRepo:
        mock_repo = MockRepo.return_value
        mock_repo.get_by_conversation = AsyncMock(return_value=mock_messages)
        
        # Load history
        history = await chat_service._load_conversation_history(conversation_id, mock_session)
    
    # Verify
    assert len(history) == 3
    # Messages are reversed (oldest first for conversation context)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Tell me about meetings."  # Actually last in DB, first after reversed
    assert history[-1]["content"] == "What's the weather?"  # Actually first in DB, last after reversed
    mock_conv_cache.set_messages.assert_called_once()


@pytest.mark.asyncio
async def test_load_conversation_history_with_cache(chat_service, mock_session, mock_conv_cache):
    """Test #4: Conversation caching - should use cached history."""
    conversation_id = "conv-456"
    cached_history = [
        {"role": "user", "content": "Cached question"},
        {"role": "assistant", "content": "Cached answer"},
    ]
    
    # Mock cache hit
    mock_conv_cache.get_messages = AsyncMock(return_value=cached_history)
    
    # Load history
    history = await chat_service._load_conversation_history(conversation_id, mock_session)
    
    # Verify cache was checked
    mock_conv_cache.get_messages.assert_called_once()
    assert history == cached_history


@pytest.mark.asyncio
async def test_compress_conversation_history_within_limits(chat_service):
    """Test #2: Smart context window - messages within token limit."""
    history = [
        {"role": "user", "content": "Short question"},
        {"role": "assistant", "content": "Short answer"},
    ]
    
    compressed = chat_service._compress_conversation_history(history)
    
    # Should keep all messages if within limits
    assert len(compressed) == 2
    assert compressed == history


@pytest.mark.asyncio
async def test_compress_conversation_history_exceeds_message_limit(chat_service):
    """Test #2: Smart context window - message limit is handled by _prepare_conversation_context."""
    # _compress_conversation_history only handles TOKEN limits, not message count
    # Create 15 short messages
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
        for i in range(15)
    ]
    
    compressed = chat_service._compress_conversation_history(history)
    
    # All messages fit within token budget (they're very short)
    # Message limit is enforced in _prepare_conversation_context, not here
    assert len(compressed) == 15


@pytest.mark.asyncio
async def test_compress_conversation_history_exceeds_token_limit(chat_service):
    """Test #2: Smart context window - token budget management."""
    # Create messages that exceed token budget (2000 tokens)
    # Each message ~600 chars = ~150 tokens
    long_text = "a" * 600
    history = [
        {"role": "user", "content": f"{long_text} {i}"}
        for i in range(20)  # 20 * 150 = 3000 tokens > 2000 limit
    ]
    
    compressed = chat_service._compress_conversation_history(history)
    
    # Should truncate to fit within token budget
    assert len(compressed) < len(history)
    
    # Verify total tokens are under budget (2000 tokens max)
    total_chars = sum(len(msg["content"]) for msg in compressed)
    estimated_tokens = total_chars // 4
    assert estimated_tokens <= 2000


@pytest.mark.asyncio
async def test_summarize_old_messages(chat_service):
    """Test #3: Summarization for long conversations."""
    old_messages = [
        {"role": "user", "content": "What projects did we discuss?"},
        {"role": "assistant", "content": "We discussed Project Alpha and Beta."},
        {"role": "user", "content": "Tell me more about Alpha."},
        {"role": "assistant", "content": "Alpha focuses on ML infrastructure."},
    ]
    
    # Mock LLM generate
    chat_service.llm.generate = AsyncMock(
        return_value="User asked about projects. Discussed Project Alpha (ML infrastructure) and Project Beta."
    )
    
    summary = await chat_service._summarize_old_messages(old_messages)
    
    # Verify summary was generated
    assert "Project Alpha" in summary
    assert "Project Beta" in summary
    assert chat_service.llm.generate.called


@pytest.mark.asyncio
async def test_prepare_conversation_context_manual_history(chat_service):
    """Test prepare_conversation_context with manually provided history."""
    manual_history = [
        {"role": "user", "content": "Manual question"},
        {"role": "assistant", "content": "Manual answer"},
    ]
    
    # Prepare context (should use manual history, not auto-load)
    result = await chat_service._prepare_conversation_context(
        conversation_id="conv-789",
        conversation_history=manual_history,
        session=None
    )
    
    # Should return manual history without DB query
    assert result == manual_history


@pytest.mark.asyncio
async def test_prepare_conversation_context_auto_load(chat_service, mock_session, mock_conv_cache):
    """Test prepare_conversation_context with auto-loading from DB."""
    conversation_id = "conv-abc"
    
    # Mock MessageRepository
    mock_messages = [
        MagicMock(role="user", content="Auto-loaded question"),
        MagicMock(role="assistant", content="Auto-loaded answer"),
    ]
    
    mock_conv_cache.get_messages = AsyncMock(return_value=None)
    mock_conv_cache.set_messages = AsyncMock()
    
    with patch('app.repositories.message_repository.MessageRepository') as MockRepo:
        mock_repo = MockRepo.return_value
        mock_repo.get_by_conversation = AsyncMock(return_value=mock_messages)
        
        # Prepare context (should auto-load from DB)
        result = await chat_service._prepare_conversation_context(
            conversation_id=conversation_id,
            conversation_history=None,
            session=mock_session
        )
    
    # Verify history loaded and compressed
    assert len(result) == 2
    # Messages are reversed
    assert result[0]["content"] == "Auto-loaded answer"  # Reversed order


@pytest.mark.asyncio
async def test_prepare_conversation_context_with_summarization(chat_service, mock_session, mock_conv_cache):
    """Test prepare_conversation_context with summarization for long conversations."""
    conversation_id = "conv-def"
    
    # Create 25 messages (exceeds summarize_threshold=20)
    messages = [
        MagicMock(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
        for i in range(25)
    ]
    
    mock_conv_cache.get_messages = AsyncMock(return_value=None)
    mock_conv_cache.set_messages = AsyncMock()
    
    # Mock summarization
    chat_service.llm.generate = AsyncMock(return_value="Summary of old messages")
    
    with patch('app.repositories.message_repository.MessageRepository') as MockRepo:
        mock_repo = MockRepo.return_value
        mock_repo.get_by_conversation = AsyncMock(return_value=messages)
        
        # Prepare context
        result = await chat_service._prepare_conversation_context(
            conversation_id=conversation_id,
            conversation_history=None,
            session=mock_session
        )
    
    # Verify summarization was applied
    # Should have: 1 system message (summary) + last 10 messages
    assert len(result) == 11
    assert result[0]["role"] == "system"
    assert "Summary" in result[0]["content"]
    # Messages are reversed from DB, so first message in result is last from DB
    assert result[-1]["content"] == "Message 0"  # Reversed order


@pytest.mark.asyncio
async def test_prepare_conversation_context_no_conversation_id(chat_service):
    """Test prepare_conversation_context without conversation_id."""
    # No conversation_id provided
    result = await chat_service._prepare_conversation_context(
        conversation_id=None,
        conversation_history=None,
        session=None
    )
    
    # Should return empty list
    assert result == []


@pytest.mark.asyncio
async def test_answer_with_conversation_id(chat_service, mock_session, mock_conv_cache):
    """Integration test: answer() with conversation_id auto-loads history."""
    conversation_id = "conv-integration"
    
    # Mock MessageRepository and cache
    mock_messages = [
        MagicMock(role="user", content="Previous question"),
        MagicMock(role="assistant", content="Previous answer"),
    ]
    
    mock_conv_cache.get_messages = AsyncMock(return_value=None)
    mock_conv_cache.set_messages = AsyncMock()
    
    # Mock retrieval and dependencies
    chat_service.retrieval.search = AsyncMock(return_value=[])
    chat_service.confidence_scorer.calculate = MagicMock(return_value=0.5)
    chat_service.query_rewriter.rewrite_if_needed = AsyncMock(return_value={
        "query": "What did we discuss?",
        "was_rewritten": False
    })
    # Mock query_processor.process
    with patch.object(chat_service, 'query_processor') as mock_processor:
        mock_processor.process = AsyncMock(return_value={
            "intent": "query",
            "confidence": 0.9,
            "rewrite_info": None
        })
    # Mock query_processor.process
    with patch.object(chat_service, 'query_processor') as mock_processor:
        mock_processor.process = AsyncMock(return_value={
            "intent": "query",
            "confidence": 0.9,
            "rewrite_info": None
        })
    
        with patch('app.repositories.message_repository.MessageRepository') as MockRepo:
            mock_repo = MockRepo.return_value
            mock_repo.get_by_conversation = AsyncMock(return_value=mock_messages)
            
            # Mock _generate_response
            with patch.object(chat_service, '_generate_response', new=AsyncMock(return_value={
                "answer": "Test answer",
                "sources": [],
            })):
                result = await chat_service.answer(
                    query="What did we discuss?",
                    conversation_id=conversation_id,
                    session=mock_session,
                )
    
    # Verify answer was generated
    assert "answer" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

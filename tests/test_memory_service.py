"""
Tests for Memory Service.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import uuid

from app.services.memory_service import MemoryService
from app.database.models import Conversation, Message
from app.schemas.message import MessageRole, ConversationContext


# ============== Fixtures ==============

@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_conv_repo():
    """Create a mock conversation repository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_msg_repo():
    """Create a mock message repository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_conv_cache():
    """Create a mock conversation cache."""
    cache = AsyncMock()
    return cache


@pytest.fixture
def sample_messages():
    """Create sample messages."""
    messages = []
    for i, (role, content) in enumerate([
        ("user", "Hello, can you help me?"),
        ("assistant", "Of course! How can I assist you today?"),
        ("user", "What was discussed in the last meeting?"),
        ("assistant", "Based on the transcripts, the team discussed the product roadmap."),
    ]):
        msg = MagicMock(spec=Message)
        msg.id = str(uuid.uuid4())
        msg.role = role
        msg.content = content
        msg.created_at = datetime.utcnow()
        messages.append(msg)
    return messages


@pytest.fixture
def sample_conversation():
    """Create a sample conversation."""
    conv = MagicMock(spec=Conversation)
    conv.id = str(uuid.uuid4())
    conv.title = "Test Conversation"
    conv.summary = "Previous discussion about Q4 budget."
    return conv


# ============== Tests ==============

class TestMemoryService:
    """Tests for MemoryService."""
    
    def test_init(self, mock_session):
        """Test MemoryService initialization."""
        with patch('app.services.memory_service.ConversationRepository') as mock_conv, \
             patch('app.services.memory_service.MessageRepository') as mock_msg:
            
            service = MemoryService(session=mock_session)
            
            assert service.session == mock_session
            mock_conv.assert_called_once_with(mock_session)
            mock_msg.assert_called_once_with(mock_session)
    
    def test_init_with_cache(self, mock_session, mock_conv_cache):
        """Test MemoryService initialization with cache."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session, conversation_cache=mock_conv_cache)
            
            assert service.conv_cache == mock_conv_cache
    
    def test_estimate_tokens(self, mock_session):
        """Test token estimation."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session)
            
            # ~4 chars per token
            assert service._estimate_tokens("test") == 1
            assert service._estimate_tokens("hello world") == 2
            assert service._estimate_tokens("a" * 100) == 25
    
    @pytest.mark.asyncio
    async def test_get_context_from_cache(self, mock_session, mock_conv_cache):
        """Test getting context from cache."""
        mock_conv_cache.get_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session, conversation_cache=mock_conv_cache)
            
            result = await service.get_context("conv-123")
            
            assert len(result.messages) == 2
            assert result.messages[0].role == MessageRole.USER
            assert result.messages[0].content == "Hello"
    
    @pytest.mark.asyncio
    async def test_get_context_from_database(self, mock_session, sample_messages):
        """Test getting context from database when cache miss."""
        with patch('app.services.memory_service.ConversationRepository') as mock_conv_cls, \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls:
            
            mock_msg = AsyncMock()
            mock_msg.get_context_messages.return_value = (sample_messages, False)
            mock_msg.count_by_conversation.return_value = 4
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session)
            
            result = await service.get_context("conv-123")
            
            assert len(result.messages) == 4
            assert result.total_messages == 4
    
    @pytest.mark.asyncio
    async def test_get_context_updates_cache(self, mock_session, mock_conv_cache, sample_messages):
        """Test that cache is updated after database fetch."""
        mock_conv_cache.get_messages.return_value = None  # Cache miss
        
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls:
            
            mock_msg = AsyncMock()
            mock_msg.get_context_messages.return_value = (sample_messages, False)
            mock_msg.count_by_conversation.return_value = 4
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session, conversation_cache=mock_conv_cache)
            
            await service.get_context("conv-123")
            
            mock_conv_cache.set_messages.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_context_with_summary(self, mock_session, sample_messages, sample_conversation):
        """Test getting context includes summary for long conversations."""
        with patch('app.services.memory_service.ConversationRepository') as mock_conv_cls, \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls, \
             patch('app.services.memory_service.settings') as mock_settings:
            
            mock_settings.MIN_HISTORY_MESSAGES = 5
            mock_settings.MAX_HISTORY_MESSAGES = 20
            mock_settings.CONTEXT_TOKEN_BUDGET = 2000
            mock_settings.SUMMARIZE_THRESHOLD = 10  # Low threshold
            
            mock_conv = AsyncMock()
            mock_conv.get_by_id.return_value = sample_conversation
            mock_conv_cls.return_value = mock_conv
            
            mock_msg = AsyncMock()
            mock_msg.get_context_messages.return_value = (sample_messages, False)
            mock_msg.count_by_conversation.return_value = 15  # > threshold
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session)
            
            result = await service.get_context("conv-123")
            
            assert result.summary == sample_conversation.summary
    
    @pytest.mark.asyncio
    async def test_get_context_truncated(self, mock_session, sample_messages):
        """Test context indicates truncation."""
        with patch('app.services.memory_service.ConversationRepository') as mock_conv_cls, \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls:
            
            # Mock conversation repository
            mock_conv = AsyncMock()
            mock_conv.get_by_id = AsyncMock(return_value=None)  # No summary
            mock_conv_cls.return_value = mock_conv
            
            # Mock message repository  
            mock_msg = AsyncMock()
            mock_msg.get_context_messages = AsyncMock(return_value=(sample_messages, True))
            mock_msg.count_by_conversation = AsyncMock(return_value=100)  # Many more messages
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session)
            
            result = await service.get_context("conv-123")
            
            assert result.truncated is True
    
    @pytest.mark.asyncio
    async def test_get_context_for_chat(self, mock_session, mock_conv_cache):
        """Test getting context formatted for chat API."""
        mock_conv_cache.get_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session, conversation_cache=mock_conv_cache)
            
            result = await service.get_context_for_chat("conv-123")
            
            assert isinstance(result, list)
            assert result[0]["role"] == "user"
            assert result[0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_get_context_for_chat_with_summary(self, mock_session, sample_messages, sample_conversation):
        """Test chat context includes summary as system message."""
        with patch('app.services.memory_service.ConversationRepository') as mock_conv_cls, \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls, \
             patch('app.services.memory_service.settings') as mock_settings:
            
            mock_settings.MIN_HISTORY_MESSAGES = 5
            mock_settings.MAX_HISTORY_MESSAGES = 20
            mock_settings.CONTEXT_TOKEN_BUDGET = 2000
            mock_settings.SUMMARIZE_THRESHOLD = 3  # Low threshold
            
            mock_conv = AsyncMock()
            mock_conv.get_by_id.return_value = sample_conversation
            mock_conv_cls.return_value = mock_conv
            
            mock_msg = AsyncMock()
            mock_msg.get_context_messages.return_value = (sample_messages, False)
            mock_msg.count_by_conversation.return_value = 10
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session)
            
            result = await service.get_context_for_chat("conv-123")
            
            # First message should be system with summary
            assert result[0]["role"] == "system"
            assert "summary" in result[0]["content"].lower()
    
    @pytest.mark.asyncio
    async def test_should_summarize_true(self, mock_session):
        """Test should_summarize returns True for long conversations."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls, \
             patch('app.services.memory_service.settings') as mock_settings:
            
            mock_settings.MIN_HISTORY_MESSAGES = 5
            mock_settings.MAX_HISTORY_MESSAGES = 20
            mock_settings.CONTEXT_TOKEN_BUDGET = 2000
            mock_settings.SUMMARIZE_THRESHOLD = 10
            
            mock_msg = AsyncMock()
            mock_msg.count_by_conversation.return_value = 15  # > threshold
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session)
            
            result = await service.should_summarize("conv-123")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_should_summarize_false(self, mock_session):
        """Test should_summarize returns False for short conversations."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository') as mock_msg_cls, \
             patch('app.services.memory_service.settings') as mock_settings:
            
            mock_settings.MIN_HISTORY_MESSAGES = 5
            mock_settings.MAX_HISTORY_MESSAGES = 20
            mock_settings.CONTEXT_TOKEN_BUDGET = 2000
            mock_settings.SUMMARIZE_THRESHOLD = 10
            
            mock_msg = AsyncMock()
            mock_msg.count_by_conversation.return_value = 5  # < threshold
            mock_msg_cls.return_value = mock_msg
            
            service = MemoryService(session=mock_session)
            
            result = await service.should_summarize("conv-123")
            
            assert result is False


class TestMemoryServiceTokenEstimation:
    """Additional tests for token estimation."""
    
    def test_empty_string(self, mock_session):
        """Test token estimation for empty string."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session)
            
            assert service._estimate_tokens("") == 0
    
    def test_unicode_string(self, mock_session):
        """Test token estimation for unicode string."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session)
            
            # Unicode chars should be counted
            result = service._estimate_tokens("Hello 世界")
            assert result >= 2
    
    def test_long_string(self, mock_session):
        """Test token estimation for long string."""
        with patch('app.services.memory_service.ConversationRepository'), \
             patch('app.services.memory_service.MessageRepository'):
            
            service = MemoryService(session=mock_session)
            
            long_text = "a" * 4000
            assert service._estimate_tokens(long_text) == 1000

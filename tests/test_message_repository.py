"""
Tests for MessageRepository.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.message_repository import MessageRepository
from app.database.models import Message, Conversation


class TestMessageRepository:
    """Tests for MessageRepository class."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock(spec=AsyncSession)
        return session
    
    @pytest.fixture
    def repo(self, mock_session):
        """Create a MessageRepository instance."""
        return MessageRepository(mock_session)
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        msg = MagicMock(spec=Message)
        msg.id = "msg-123"
        msg.conversation_id = "conv-456"
        msg.role = "user"
        msg.content = "Hello, how are you?"
        msg.created_at = datetime.utcnow()
        msg.deleted_at = None
        msg.sources = None
        msg.confidence_score = None
        msg.metadata = {}
        return msg
    
    @pytest.mark.asyncio
    async def test_get_by_conversation(self, repo, mock_session, sample_message):
        """Test getting messages for a conversation."""
        # Mock the result
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        messages = await repo.get_by_conversation("conv-456")
        
        assert len(messages) == 1
        assert messages[0].conversation_id == "conv-456"
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_conversation_descending(self, repo, mock_session, sample_message):
        """Test getting messages in descending order."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        messages = await repo.get_by_conversation("conv-456", ascending=False)
        
        assert len(messages) == 1
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_conversation_include_deleted(self, repo, mock_session, sample_message):
        """Test getting messages including deleted ones."""
        sample_message.deleted_at = datetime.utcnow()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        messages = await repo.get_by_conversation("conv-456", include_deleted=True)
        
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_get_recent_messages(self, repo, mock_session, sample_message):
        """Test getting most recent messages."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        messages = await repo.get_recent_messages("conv-456", limit=20)
        
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_count_by_conversation(self, repo, mock_session):
        """Test counting messages in a conversation."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 15
        mock_session.execute.return_value = mock_result
        
        count = await repo.count_by_conversation("conv-456")
        
        assert count == 15
    
    @pytest.mark.asyncio
    async def test_count_by_conversation_include_deleted(self, repo, mock_session):
        """Test counting messages including deleted."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 20
        mock_session.execute.return_value = mock_result
        
        count = await repo.count_by_conversation("conv-456", include_deleted=True)
        
        assert count == 20
    
    @pytest.mark.asyncio
    async def test_add_message(self, repo, mock_session, sample_message):
        """Test adding a message."""
        # Mock the create method from base repository
        with patch.object(repo, 'create', return_value=sample_message) as mock_create:
            result = await repo.add_message(
                conversation_id="conv-456",
                role="user",
                content="Hello!",
                sources=[{"doc": "test.pdf"}],
                confidence_score=0.9,
                metadata={"key": "value"}
            )
            
            mock_create.assert_called_once_with(
                conversation_id="conv-456",
                role="user",
                content="Hello!",
                sources=[{"doc": "test.pdf"}],
                confidence_score=0.9,
                metadata={"key": "value"}
            )
            assert result == sample_message
    
    @pytest.mark.asyncio
    async def test_add_user_message(self, repo, sample_message):
        """Test adding a user message."""
        with patch.object(repo, 'add_message', return_value=sample_message) as mock_add:
            result = await repo.add_user_message(
                conversation_id="conv-456",
                content="User query",
                metadata={"source": "web"}
            )
            
            mock_add.assert_called_once_with(
                conversation_id="conv-456",
                role="user",
                content="User query",
                metadata={"source": "web"}
            )
    
    @pytest.mark.asyncio
    async def test_add_assistant_message(self, repo, sample_message):
        """Test adding an assistant message."""
        with patch.object(repo, 'add_message', return_value=sample_message) as mock_add:
            result = await repo.add_assistant_message(
                conversation_id="conv-456",
                content="Assistant response",
                sources=[{"doc": "source.pdf"}],
                confidence_score=0.85,
                metadata={"model": "llama"}
            )
            
            mock_add.assert_called_once_with(
                conversation_id="conv-456",
                role="assistant",
                content="Assistant response",
                sources=[{"doc": "source.pdf"}],
                confidence_score=0.85,
                metadata={"model": "llama"}
            )
    
    @pytest.mark.asyncio
    async def test_get_last_message(self, repo, mock_session, sample_message):
        """Test getting the last message in a conversation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_message
        mock_session.execute.return_value = mock_result
        
        message = await repo.get_last_message("conv-456")
        
        assert message == sample_message
    
    @pytest.mark.asyncio
    async def test_get_last_message_by_role(self, repo, mock_session, sample_message):
        """Test getting the last message by role."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_message
        mock_session.execute.return_value = mock_result
        
        message = await repo.get_last_message("conv-456", role="user")
        
        assert message == sample_message
    
    @pytest.mark.asyncio
    async def test_get_last_message_not_found(self, repo, mock_session):
        """Test getting last message when none exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        message = await repo.get_last_message("conv-456")
        
        assert message is None
    
    @pytest.mark.asyncio
    async def test_get_context_messages(self, repo, mock_session):
        """Test getting context messages with token limit."""
        # Create messages with varying lengths
        messages = []
        for i in range(5):
            msg = MagicMock(spec=Message)
            msg.id = f"msg-{i}"
            msg.content = "A" * 100  # 100 chars = ~25 tokens
            messages.append(msg)
        
        with patch.object(repo, 'get_recent_messages', return_value=messages):
            result, truncated = await repo.get_context_messages(
                "conv-456",
                limit=15,
                max_tokens=100
            )
            
            # Should return some messages and indicate truncation
            assert isinstance(result, list)
            assert isinstance(truncated, bool)
    
    @pytest.mark.asyncio
    async def test_get_context_messages_no_truncation(self, repo, mock_session):
        """Test getting context messages without truncation."""
        # Create small messages
        messages = []
        for i in range(3):
            msg = MagicMock(spec=Message)
            msg.id = f"msg-{i}"
            msg.content = "Hi"  # Very small
            messages.append(msg)
        
        with patch.object(repo, 'get_recent_messages', return_value=messages):
            result, truncated = await repo.get_context_messages(
                "conv-456",
                limit=15,
                max_tokens=2000
            )
            
            assert len(result) == 3
            assert truncated is False
    
    @pytest.mark.asyncio
    async def test_search_in_conversation(self, repo, mock_session, sample_message):
        """Test searching messages within a conversation."""
        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5
        
        # Mock search query
        mock_search_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_search_result.scalars.return_value = mock_scalars
        
        mock_session.execute.side_effect = [mock_count_result, mock_search_result]
        
        messages, total = await repo.search_in_conversation(
            "conv-456",
            "hello",
            skip=0,
            limit=20
        )
        
        assert total == 5
        assert len(messages) == 1
    
    @pytest.mark.asyncio
    async def test_search_user_messages(self, repo, mock_session):
        """Test searching messages across user conversations."""
        # Mock message and conversation
        mock_message = MagicMock(spec=Message)
        mock_message.id = "msg-1"
        mock_message.content = "Search result"
        
        mock_conversation = MagicMock(spec=Conversation)
        mock_conversation.id = "conv-1"
        
        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 3
        
        # Mock search query
        mock_search_result = MagicMock()
        mock_search_result.all.return_value = [(mock_message, mock_conversation)]
        
        mock_session.execute.side_effect = [mock_count_result, mock_search_result]
        
        results, total = await repo.search_user_messages(
            "user-123",
            "search",
            skip=0,
            limit=20
        )
        
        assert total == 3
        assert len(results) == 1
        assert "message" in results[0]
        assert "conversation" in results[0]
    
    @pytest.mark.asyncio
    async def test_search_user_messages_with_role(self, repo, mock_session):
        """Test searching user messages with role filter."""
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1
        
        mock_search_result = MagicMock()
        mock_search_result.all.return_value = []
        
        mock_session.execute.side_effect = [mock_count_result, mock_search_result]
        
        results, total = await repo.search_user_messages(
            "user-123",
            "search",
            role="assistant"
        )
        
        assert total == 1
    
    @pytest.mark.asyncio
    async def test_get_messages_by_role(self, repo, mock_session, sample_message):
        """Test getting messages filtered by role."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        messages = await repo.get_messages_by_role("conv-456", "user", limit=50)
        
        assert len(messages) == 1
        assert messages[0].role == "user"
    
    @pytest.mark.asyncio
    async def test_bulk_delete_by_conversation(self, repo, mock_session, sample_message):
        """Test soft deleting all messages in a conversation."""
        sample_message.deleted_at = None
        
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_message]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_session.flush = AsyncMock()
        
        count = await repo.bulk_delete_by_conversation("conv-456")
        
        assert count == 1
        assert sample_message.deleted_at is not None
        mock_session.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_delete_no_messages(self, repo, mock_session):
        """Test bulk delete when no messages exist."""
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_session.flush = AsyncMock()
        
        count = await repo.bulk_delete_by_conversation("conv-456")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_conversation_export(self, repo, sample_message):
        """Test getting messages for export."""
        with patch.object(repo, 'get_by_conversation', return_value=[sample_message]):
            result = await repo.get_conversation_export("conv-456")
            
            # Method should call get_by_conversation with high limit
            repo.get_by_conversation.assert_called_once_with(
                "conv-456",
                limit=10000,
                ascending=True
            )

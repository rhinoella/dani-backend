"""
Tests for conversation storage and memory features.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone
import uuid

from app.database.models import User, Conversation, Message
from app.repositories.user_repository import UserRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.message_repository import MessageRepository
from app.services.user_service import UserService
from app.services.conversation_service import ConversationService
from app.services.memory_service import MemoryService
from app.schemas.conversation import ConversationCreate, ConversationUpdate
from app.schemas.message import MessageRole
from app.core.auth import GoogleUser


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# ============== Fixtures ==============

@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_user():
    """Create a sample user."""
    user = MagicMock(spec=User)
    user.id = str(uuid.uuid4())
    user.google_id = "google-123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.picture_url = "https://example.com/photo.jpg"
    user.created_at = utc_now()
    user.updated_at = utc_now()
    user.last_login_at = utc_now()
    user.deleted_at = None
    return user


@pytest.fixture
def sample_conversation(sample_user):
    """Create a sample conversation."""
    conv = MagicMock(spec=Conversation)
    conv.id = str(uuid.uuid4())
    conv.user_id = sample_user.id
    conv.title = "Test Conversation"
    conv.summary = None
    conv.message_count = 0
    conv.metadata = {}
    conv.created_at = utc_now()
    conv.updated_at = utc_now()
    conv.deleted_at = None
    conv.messages = []
    return conv


@pytest.fixture
def sample_messages(sample_conversation):
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
        msg.conversation_id = sample_conversation.id
        msg.role = role
        msg.content = content
        msg.sources = [] if role == "assistant" else None
        msg.confidence_score = 0.85 if role == "assistant" else None
        msg.metadata = {}
        msg.created_at = utc_now() + timedelta(seconds=i)
        msg.deleted_at = None
        messages.append(msg)
    return messages


@pytest.fixture
def google_user():
    """Create a Google user from token verification."""
    return GoogleUser(
        google_id="google-123",
        email="test@example.com",
        name="Test User",
        picture_url="https://example.com/photo.jpg",
        email_verified=True,
    )


# ============== User Repository Tests ==============

class TestUserRepository:
    """Tests for UserRepository."""
    
    @pytest.mark.asyncio
    async def test_get_by_google_id(self, mock_session, sample_user):
        """Test finding user by Google ID."""
        repo = UserRepository(mock_session)
        
        # Mock the query result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_google_id("google-123")
        
        assert result == sample_user
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_email(self, mock_session, sample_user):
        """Test finding user by email."""
        repo = UserRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_email("test@example.com")
        
        assert result == sample_user
    
    @pytest.mark.asyncio
    async def test_create_or_update_from_google_new_user(self, mock_session):
        """Test creating a new user from Google auth."""
        repo = UserRepository(mock_session)
        
        # First call returns None (user doesn't exist)
        # Second call returns the created user
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch.object(repo, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock(id="new-user-id", email="new@example.com")
            
            result = await repo.create_or_update_from_google(
                google_id="new-google-id",
                email="new@example.com",
                name="New User",
                picture_url=None,
            )
            
            mock_create.assert_called_once()


# ============== Conversation Repository Tests ==============

class TestConversationRepository:
    """Tests for ConversationRepository."""
    
    @pytest.mark.asyncio
    async def test_get_by_user(self, mock_session, sample_user, sample_conversation):
        """Test getting conversations for a user."""
        repo = ConversationRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [sample_conversation]
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_user(sample_user.id, skip=0, limit=20)
        
        assert len(result) == 1
        assert result[0] == sample_conversation
    
    @pytest.mark.asyncio
    async def test_count_by_user(self, mock_session, sample_user):
        """Test counting user conversations."""
        repo = ConversationRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute.return_value = mock_result
        
        result = await repo.count_by_user(sample_user.id)
        
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_verify_ownership_success(self, mock_session, sample_user, sample_conversation):
        """Test verifying conversation ownership."""
        repo = ConversationRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result
        
        result = await repo.verify_ownership(sample_conversation.id, sample_user.id)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_ownership_failure(self, mock_session, sample_conversation):
        """Test verifying ownership fails for wrong user."""
        repo = ConversationRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_session.execute.return_value = mock_result
        
        result = await repo.verify_ownership(sample_conversation.id, "other-user-id")
        
        assert result is False


# ============== Message Repository Tests ==============

class TestMessageRepository:
    """Tests for MessageRepository."""
    
    @pytest.mark.asyncio
    async def test_get_by_conversation(self, mock_session, sample_conversation, sample_messages):
        """Test getting messages for a conversation."""
        repo = MessageRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sample_messages
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_by_conversation(sample_conversation.id)
        
        assert len(result) == 4
    
    @pytest.mark.asyncio
    async def test_get_recent_messages(self, mock_session, sample_conversation, sample_messages):
        """Test getting recent messages."""
        repo = MessageRepository(mock_session)
        
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = list(reversed(sample_messages[:2]))
        mock_session.execute.return_value = mock_result
        
        result = await repo.get_recent_messages(sample_conversation.id, limit=2)
        
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_add_user_message(self, mock_session, sample_conversation):
        """Test adding a user message."""
        repo = MessageRepository(mock_session)
        
        with patch.object(repo, 'add_message', new_callable=AsyncMock) as mock_add:
            mock_msg = MagicMock()
            mock_msg.id = "new-msg-id"
            mock_msg.role = "user"
            mock_add.return_value = mock_msg
            
            result = await repo.add_user_message(
                conversation_id=sample_conversation.id,
                content="Test message",
            )
            
            mock_add.assert_called_once_with(
                conversation_id=sample_conversation.id,
                role="user",
                content="Test message",
                metadata=None,
            )
    
    @pytest.mark.asyncio
    async def test_add_assistant_message(self, mock_session, sample_conversation):
        """Test adding an assistant message."""
        repo = MessageRepository(mock_session)
        
        with patch.object(repo, 'add_message', new_callable=AsyncMock) as mock_add:
            mock_msg = MagicMock()
            mock_msg.id = "new-msg-id"
            mock_msg.role = "assistant"
            mock_add.return_value = mock_msg
            
            result = await repo.add_assistant_message(
                conversation_id=sample_conversation.id,
                content="Here's the answer.",
                sources=[{"title": "Meeting 1"}],
                confidence_score=0.9,
            )
            
            mock_add.assert_called_once()


# ============== Conversation Service Tests ==============

class TestConversationService:
    """Tests for ConversationService."""
    
    @pytest.mark.asyncio
    async def test_create_conversation(self, mock_session, sample_user):
        """Test creating a new conversation."""
        service = ConversationService(mock_session)
        
        with patch.object(service.conv_repo, 'create', new_callable=AsyncMock) as mock_create:
            mock_conv = MagicMock()
            mock_conv.id = "new-conv-id"
            mock_conv.user_id = sample_user.id
            mock_conv.title = "New Conversation"
            mock_create.return_value = mock_conv
            
            result = await service.create_conversation(
                user_id=sample_user.id,
                title="New Conversation",
            )
            
            assert result.id == "new-conv-id"
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_conversation_success(self, mock_session, sample_user, sample_conversation):
        """Test getting a conversation by ID."""
        service = ConversationService(mock_session)
        
        with patch.object(service.conv_repo, 'get_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_conversation
            
            result = await service.get_conversation(
                conversation_id=sample_conversation.id,
                user_id=sample_user.id,
            )
            
            assert result == sample_conversation
    
    @pytest.mark.asyncio
    async def test_get_conversation_wrong_user(self, mock_session, sample_conversation):
        """Test getting a conversation fails for wrong user."""
        service = ConversationService(mock_session)
        
        with patch.object(service.conv_repo, 'get_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_conversation
            
            result = await service.get_conversation(
                conversation_id=sample_conversation.id,
                user_id="different-user-id",
            )
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_conversation(self, mock_session, sample_user, sample_conversation):
        """Test soft deleting a conversation."""
        service = ConversationService(mock_session)
        
        with patch.object(service.conv_repo, 'get_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_conversation
            
            with patch.object(service.msg_repo, 'bulk_delete_by_conversation', new_callable=AsyncMock) as mock_del_msgs:
                with patch.object(service.conv_repo, 'soft_delete', new_callable=AsyncMock) as mock_del:
                    mock_del.return_value = True
                    
                    result = await service.delete_conversation(
                        conversation_id=sample_conversation.id,
                        user_id=sample_user.id,
                    )
                    
                    assert result is True
                    mock_del_msgs.assert_called_once()
                    mock_del.assert_called_once()


# ============== Memory Service Tests ==============

class TestMemoryService:
    """Tests for MemoryService."""
    
    @pytest.mark.asyncio
    async def test_get_context(self, mock_session, sample_conversation, sample_messages):
        """Test building conversation context."""
        service = MemoryService(mock_session)
        
        with patch.object(service.msg_repo, 'get_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (sample_messages, False)
            
            with patch.object(service.msg_repo, 'count_by_conversation', new_callable=AsyncMock) as mock_count:
                mock_count.return_value = 4
                
                result = await service.get_context(sample_conversation.id)
                
                assert result.total_messages == 4
                assert len(result.messages) == 4
                assert result.truncated is False
    
    @pytest.mark.asyncio
    async def test_get_context_for_chat(self, mock_session, sample_conversation, sample_messages):
        """Test getting context formatted for chat."""
        service = MemoryService(mock_session)
        
        with patch.object(service, 'get_context', new_callable=AsyncMock) as mock_get_ctx:
            from app.schemas.message import ConversationContext, MessageBase, MessageRole
            mock_get_ctx.return_value = ConversationContext(
                messages=[
                    MessageBase(role=MessageRole.USER, content="Hello"),
                    MessageBase(role=MessageRole.ASSISTANT, content="Hi there!"),
                ],
                summary=None,
                total_messages=2,
                context_token_count=10,
                truncated=False,
            )
            
            result = await service.get_context_for_chat(sample_conversation.id)
            
            assert len(result) == 2
            assert result[0]["role"] == "user"
            assert result[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_should_summarize_threshold(self, mock_session, sample_conversation):
        """Test summarization threshold check."""
        service = MemoryService(mock_session)
        
        # Below threshold
        with patch.object(service.msg_repo, 'count_by_conversation', new_callable=AsyncMock) as mock_count:
            mock_count.return_value = 10
            
            result = await service.should_summarize(sample_conversation.id)
            
            assert result is False
        
        # At threshold (default 20)
        with patch.object(service.msg_repo, 'count_by_conversation', new_callable=AsyncMock) as mock_count:
            mock_count.return_value = 20
            
            result = await service.should_summarize(sample_conversation.id)
            
            assert result is True
    
    def test_token_estimation(self, mock_session):
        """Test token estimation."""
        service = MemoryService(mock_session)
        
        # 100 chars should be ~25 tokens (4 chars per token)
        text = "a" * 100
        result = service._estimate_tokens(text)
        
        assert result == 25


# ============== User Service Tests ==============

class TestUserService:
    """Tests for UserService."""
    
    @pytest.mark.asyncio
    async def test_get_or_create_from_google(self, mock_session, google_user, sample_user):
        """Test getting or creating user from Google auth."""
        service = UserService(mock_session)
        
        with patch.object(service.user_repo, 'create_or_update_from_google', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = sample_user
            
            result = await service.get_or_create_from_google(google_user)
            
            assert result == sample_user
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_profile(self, mock_session, sample_user):
        """Test getting user profile with stats."""
        service = UserService(mock_session)
        
        with patch.object(service.user_repo, 'get_by_id', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_user
            
            with patch.object(service.user_repo, 'get_user_stats', new_callable=AsyncMock) as mock_stats:
                mock_stats.return_value = {
                    "conversation_count": 5,
                    "message_count": 42,
                }
                
                result = await service.get_user_profile(sample_user.id)
                
                assert result["conversation_count"] == 5
                assert result["message_count"] == 42


# ============== Integration-like Tests ==============

class TestConversationFlow:
    """Integration-like tests for conversation flow."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, mock_session, sample_user):
        """Test a full conversation flow."""
        conv_service = ConversationService(mock_session)
        
        # 1. Create conversation
        mock_conv = MagicMock()
        mock_conv.id = "test-conv-id"
        mock_conv.user_id = sample_user.id
        mock_conv.title = None
        mock_conv.message_count = 0
        
        with patch.object(conv_service.conv_repo, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_conv
            
            conv = await conv_service.create_conversation(
                user_id=sample_user.id,
            )
            
            assert conv.id == "test-conv-id"
        
        # 2. Add user message
        with patch.object(conv_service.conv_repo, 'get_by_id', new_callable=AsyncMock) as mock_get:
            mock_conv.message_count = 0  # First message
            mock_get.return_value = mock_conv
            
            with patch.object(conv_service.msg_repo, 'add_message', new_callable=AsyncMock) as mock_add:
                mock_msg = MagicMock()
                mock_msg.id = "msg-1"
                mock_msg.created_at = utc_now()
                mock_add.return_value = mock_msg
                
                with patch.object(conv_service.conv_repo, 'update_message_count', new_callable=AsyncMock):
                    with patch.object(conv_service.conv_repo, 'auto_generate_title', new_callable=AsyncMock):
                        msg = await conv_service.add_message(
                            conversation_id=conv.id,
                            user_id=sample_user.id,
                            role="user",
                            content="What was discussed in Monday's meeting?",
                        )
                        
                        assert msg.id == "msg-1"
        
        # 3. Add assistant response
        with patch.object(conv_service.conv_repo, 'get_by_id', new_callable=AsyncMock) as mock_get:
            mock_conv.message_count = 1
            mock_get.return_value = mock_conv
            
            with patch.object(conv_service.msg_repo, 'add_message', new_callable=AsyncMock) as mock_add:
                mock_msg = MagicMock()
                mock_msg.id = "msg-2"
                mock_msg.created_at = utc_now()
                mock_add.return_value = mock_msg
                
                with patch.object(conv_service.conv_repo, 'update_message_count', new_callable=AsyncMock):
                    msg = await conv_service.add_message(
                        conversation_id=conv.id,
                        user_id=sample_user.id,
                        role="assistant",
                        content="Based on the meeting transcripts...",
                        confidence_score=0.92,
                    )
                    
                    assert msg.id == "msg-2"


class TestRateLimitIntegration:
    """Tests for rate limiting integration."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_check(self, mock_session, sample_user):
        """Test rate limit checking."""
        from app.cache.rate_limiter import RateLimiter, RateLimitResult
        
        mock_limiter = MagicMock(spec=RateLimiter)
        mock_limiter.check_rate_limit = AsyncMock(return_value=RateLimitResult(
            allowed=True,
            remaining=19,
            reset_at=1234567890.0,
            limit=20,
        ))
        
        service = UserService(mock_session, mock_limiter)
        
        allowed, limit_info = await service.check_rate_limit(sample_user.id)
        
        assert allowed is True
        assert limit_info is None
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, mock_session, sample_user):
        """Test rate limit exceeded."""
        from app.cache.rate_limiter import RateLimiter, RateLimitResult
        
        mock_limiter = MagicMock(spec=RateLimiter)
        mock_limiter.check_rate_limit = AsyncMock(return_value=RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=1234567890.0,
            limit=20,
            retry_after=30,
        ))
        mock_limiter.get_status = AsyncMock(return_value={
            "minute": {"used": 20, "limit": 20, "reset_in": 30},
            "day": {"used": 100, "limit": 500, "reset_in": 86400},
        })
        
        service = UserService(mock_session, mock_limiter)
        
        allowed, limit_info = await service.check_rate_limit(sample_user.id)
        
        assert allowed is False
        assert limit_info["retry_after"] == 30


import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from app.services.memory_service import MemoryService
from app.database.models import Message
from app.schemas.message import MessageBase, MessageRole
from app.schemas.message import MessageBase

@pytest.fixture
def mock_session():
    return AsyncMock()

@pytest.fixture
def memory_service(mock_session):
    return MemoryService(mock_session)

@pytest.mark.asyncio
async def test_get_context_for_chat_references(memory_service):
    """
    Test that memory service correctly retrieves a sequence of messages
    that allows for referencing past context.
    """
    conversation_id = "test-conv-123"
    
    # Simulate a conversation thread
    # 1. User: "Tell me about the Q1 marketing plan."
    # 2. Assistant: "The Q1 plan focuses on social media..."
    # 3. User: "What is the budget for that?" -> "that" refers to the plan
    
    mock_messages = [
        Message(
            id="1", 
            role=MessageRole.USER, 
            content="Tell me about the Q1 marketing plan.",
            created_at=datetime.now()
        ),
        Message(
            id="2", 
            role=MessageRole.ASSISTANT, 
            content="The Q1 plan focuses on social media growth and partnership expansion.",
            created_at=datetime.now()
        ),
        Message(
            id="3", 
            role=MessageRole.USER, 
            content="What is the budget for that?",
            created_at=datetime.now()
        )
    ]
    
    # Mock the repository to return these messages when requested
    # We mock get_context_messages to simulating fetching the history
    memory_service.msg_repo.get_context_messages = AsyncMock(return_value=(mock_messages, False))
    memory_service.msg_repo.count_by_conversation = AsyncMock(return_value=3)
    
    # Execute
    context = await memory_service.get_context_for_chat(conversation_id)
    
    # Verify
    assert len(context) == 3
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "Tell me about the Q1 marketing plan."
    assert context[1]["role"] == "assistant" 
    assert "social media" in context[1]["content"]
    assert context[2]["role"] == "user"
    assert context[2]["content"] == "What is the budget for that?"
    
    # Print for user visibility
    print("\n--- Memory Context Retrieval Test ---")
    for msg in context:
        print(f"{msg['role'].upper()}: {msg['content']}")
    print("-------------------------------------")
    print("✅ Successfully retrieved conversation history for context referencing.")


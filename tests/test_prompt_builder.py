"""
Tests for PromptBuilder
"""
import pytest
from app.llm.prompt_builder import PromptBuilder
from app.persona.system_prompt import DANI_SYSTEM_PROMPT


@pytest.fixture
def prompt_builder():
    """Create PromptBuilder instance"""
    return PromptBuilder()


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        {
            "title": "Q1 Strategy Meeting",
            "date": 1734480000000,
            "text": "We discussed launching the mobile app in Q1 2025. Budget approved for $500K. Sarah will lead the team.",
            "speakers": ["Bunmi", "Sarah"],
            "transcript_id": "test-001",
        },
        {
            "title": "Engineering Sync",
            "date": 1734566400000,
            "text": "Backend API needs to be ready by January 15. Team decided on React Native framework for better ecosystem.",
            "speakers": ["David", "Tech Team"],
            "transcript_id": "test-002",
        },
    ]


def test_build_chat_prompt_basic(prompt_builder, sample_chunks):
    """Test building basic chat prompt"""
    query = "What is our mobile strategy?"
    
    prompt = prompt_builder.build_chat_prompt(query, sample_chunks)
    
    # Verify system prompt is included
    assert DANI_SYSTEM_PROMPT in prompt
    
    # Verify query is included
    assert query in prompt
    
    # Verify context from chunks is included
    assert "mobile app" in prompt
    assert "Q1 Strategy Meeting" in prompt
    assert "Engineering Sync" in prompt
    
    # Verify prompt structure
    assert "MEETING SOURCES:" in prompt
    assert "User Question:" in prompt
    assert "Answer:" in prompt


def test_build_chat_prompt_no_chunks(prompt_builder):
    """Test building prompt when no chunks are available"""
    query = "What is our strategy?"
    
    prompt = prompt_builder.build_chat_prompt(query, [])
    
    # Should still have structure
    assert DANI_SYSTEM_PROMPT in prompt
    assert query in prompt
    assert "MEETING SOURCES:" in prompt
    assert "No relevant meeting notes found" in prompt


def test_build_chat_prompt_with_summary_format(prompt_builder, sample_chunks):
    """Test building prompt with summary output format"""
    query = "Summarize our mobile strategy"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="summary"
    )
    
    # Verify output format instructions are included
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "SUMMARY" in prompt.upper()
    assert "bullet points" in prompt.lower()


def test_build_chat_prompt_with_decisions_format(prompt_builder, sample_chunks):
    """Test building prompt with decisions output format"""
    query = "What decisions were made?"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="decisions"
    )
    
    # Verify decisions format instructions
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "DECISIONS" in prompt.upper()
    assert "WHO" in prompt


def test_build_chat_prompt_with_tasks_format(prompt_builder, sample_chunks):
    """Test building prompt with tasks output format"""
    query = "What are the action items?"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="tasks"
    )
    
    # Verify tasks format instructions
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "ACTION ITEMS" in prompt.upper()


def test_build_chat_prompt_with_email_format(prompt_builder, sample_chunks):
    """Test building prompt with email output format"""
    query = "Draft an email about the mobile launch"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="email"
    )
    
    # Verify email format instructions
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "EMAIL" in prompt.upper()
    assert "Subject" in prompt


def test_build_chat_prompt_with_whatsapp_format(prompt_builder, sample_chunks):
    """Test building prompt with WhatsApp output format"""
    query = "Send a WhatsApp update"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="whatsapp"
    )
    
    # Verify WhatsApp format instructions
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "WHATSAPP" in prompt.upper()


def test_build_chat_prompt_with_insights_format(prompt_builder, sample_chunks):
    """Test building prompt with insights output format"""
    query = "What are the strategic insights?"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="insights"
    )
    
    # Verify insights format instructions
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "INSIGHTS" in prompt.upper()


def test_build_chat_prompt_with_slides_format(prompt_builder, sample_chunks):
    """Test building prompt with slides output format"""
    query = "Create a slide deck"
    
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="slides"
    )
    
    # Verify slides format instructions
    assert "OUTPUT FORMAT REQUIRED:" in prompt
    assert "SLIDE" in prompt.upper()


def test_chunk_truncation(prompt_builder):
    """Test that long chunks are truncated"""
    long_text = "A" * 3000  # Longer than 2500 char limit
    chunks = [{
        "title": "Long Meeting",
        "date": 1734480000000,
        "text": long_text,
        "speakers": ["Speaker"],
    }]
    
    prompt = prompt_builder.build_chat_prompt("test query", chunks)
    
    # Text should be truncated to 2500 chars
    assert long_text not in prompt
    # Should contain truncated text (2500 A's)
    assert "A" * 2500 in prompt or "A" * 2499 in prompt


def test_multiple_chunks_separated(prompt_builder, sample_chunks):
    """Test that multiple chunks are properly separated"""
    query = "What was discussed?"
    
    prompt = prompt_builder.build_chat_prompt(query, sample_chunks)
    
    # Chunks should be separated by delimiter (now uses === separator)
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt
    
    # Both meeting titles should be present
    assert "Q1 Strategy Meeting" in prompt
    assert "Engineering Sync" in prompt


def test_prompt_without_output_format(prompt_builder, sample_chunks):
    """Test that prompt works without output format"""
    query = "What's our strategy?"
    
    prompt = prompt_builder.build_chat_prompt(query, sample_chunks, output_format=None)
    
    # Should not have format instructions
    assert "OUTPUT FORMAT REQUIRED:" not in prompt
    
    # Should still have basic structure
    assert DANI_SYSTEM_PROMPT in prompt
    assert query in prompt


def test_invalid_output_format(prompt_builder, sample_chunks):
    """Test handling of invalid output format"""
    query = "What's our strategy?"
    
    # Invalid format should be ignored (get_template returns empty string)
    prompt = prompt_builder.build_chat_prompt(
        query, 
        sample_chunks, 
        output_format="invalid_format"
    )
    
    # Should have basic structure but no format instructions
    assert DANI_SYSTEM_PROMPT in prompt
    assert query in prompt

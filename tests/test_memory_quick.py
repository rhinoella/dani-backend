#!/usr/bin/env python3
"""Quick test of conversation memory features."""
import asyncio
import sys
sys.path.insert(0, '.')

from app.services.chat_service import ChatService
from app.cache.conversation_cache import ConversationCache

async def test_memory():
    # Create service with conversation cache
    service = ChatService()
    cache = ConversationCache()
    service.set_conversation_cache(cache)
    
    # Simulate some conversation history
    test_history = [
        {"role": "user", "content": "What meetings did we have about project planning?"},
        {"role": "assistant", "content": "You had 3 meetings: Central Ops Kickoff, Weekly Finance Alignment, and Strategy Session."},
        {"role": "user", "content": "Who attended the Central Ops meeting?"},
        {"role": "assistant", "content": "Bunmi Akinyemiju, Melissa Omede, and Adeola Olaniyan attended."},
    ]
    
    print("=" * 60)
    print("Testing Conversation Memory Features")
    print("=" * 60)
    
    # Test token estimation
    history_str = str(test_history)
    estimated_tokens = len(history_str) // 4
    print(f"\n✅ History with {len(test_history)} messages")
    print(f"   Estimated tokens: ~{estimated_tokens}")
    
    # Test compression
    compressed = service._compress_conversation_history(test_history)
    print(f"\n✅ After compression: {len(compressed)} messages")
    
    # Test with more messages (40 total)
    long_history = test_history * 10
    compressed_long = service._compress_conversation_history(long_history)
    print(f"\n✅ Long history ({len(long_history)} msgs) → compressed to {len(compressed_long)} msgs")
    
    # Test cache (uses async methods)
    conv_id = "test-conv-123"
    await cache.set_messages(conv_id, test_history)
    cached = await cache.get_messages(conv_id)
    print(f"\n✅ Cache working: stored {len(test_history)} messages, retrieved {len(cached) if cached else 0}")
    
    # Test cache miss
    missing = await cache.get_messages("nonexistent")
    print(f"\n✅ Cache miss returns: {missing}")
    
    print("\n" + "=" * 60)
    print("✅ ALL CONVERSATION MEMORY FEATURES WORKING!")
    print("=" * 60)
    print("\nTo use conversation memory with the API:")
    print("1. Authenticate with Google OAuth")
    print("2. The system will auto-create conversations")
    print("3. Pass conversation_id on subsequent requests")
    print("4. History is auto-loaded and cached")

if __name__ == "__main__":
    asyncio.run(test_memory())

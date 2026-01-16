#!/usr/bin/env python3
"""
Live test of conversation memory features.
Tests auto-loading, caching, compression, and summarization.
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1"

def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def make_chat_request(query: str, conversation_id: str = None, stream: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Make a chat request to the API."""
    url = f"{BASE_URL}/chat"
    payload = {
        "query": query,
        "stream": stream,
        "verbose": verbose,
        "include_history": True
    }
    
    if conversation_id:
        payload["conversation_id"] = conversation_id
    
    print(f"📤 Request: {query[:80]}...")
    if conversation_id:
        print(f"   Conversation ID: {conversation_id}")
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Response received")
        print(f"   Answer: {data.get('answer', 'N/A')[:100]}...")
        
        if verbose and 'debug' in data:
            debug = data['debug']
            print(f"\n📊 Debug Info:")
            print(f"   Retrieved chunks: {debug.get('retrieved_chunks', 0)}")
            print(f"   Query intent: {debug.get('query_intent', 'N/A')}")
            confidence = debug.get('confidence', 0)
            if isinstance(confidence, (int, float)):
                print(f"   Confidence: {confidence:.2f}")
            else:
                print(f"   Confidence: {confidence}")
            if 'conversation_messages_used' in debug:
                print(f"   💬 Conversation messages used: {debug['conversation_messages_used']}")
            if 'query_rewrite' in debug:
                print(f"   🔄 Query rewritten: {debug['query_rewrite']}")
        
        # Show sources
        if 'sources' in data and data['sources']:
            print(f"\n📚 Sources ({len(data['sources'])}):")
            for i, source in enumerate(data['sources'][:3], 1):
                title = source.get('title') or source.get('document_title', 'Unknown')
                print(f"   {i}. {str(title)[:60]}")
                relevance = source.get('relevance_score', 0)
                if isinstance(relevance, (int, float)):
                    print(f"      Relevance: {relevance:.2%}")
                else:
                    print(f"      Relevance: {relevance}")
                preview = source.get('text_preview') or source.get('text', '')[:80]
                print(f"      Preview: {str(preview)[:80]}...")
        
        return data
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")
        return {}

def test_scenario_1_new_conversation():
    """Test creating a new conversation."""
    print_section("SCENARIO 1: New Conversation (No History)")
    
    # First message - creates new conversation
    response = make_chat_request(
        "What meetings did we have about project planning?",
        conversation_id=None
    )
    
    conversation_id = response.get('conversation_id')
    if conversation_id:
        print(f"\n✅ New conversation created: {conversation_id}")
    
    return conversation_id

def test_scenario_2_follow_up(conversation_id: str):
    """Test follow-up question with conversation history."""
    print_section("SCENARIO 2: Follow-up Question (Should Load History)")
    
    # Follow-up question - should load previous context
    response = make_chat_request(
        "What were the key decisions made?",  # Vague follow-up
        conversation_id=conversation_id
    )
    
    print("\n💡 This should show:")
    print("   - Conversation messages used from history")
    print("   - Query potentially rewritten with context")
    print("   - Answer referencing previous discussion")

def test_scenario_3_multiple_turns(conversation_id: str):
    """Test multiple conversation turns."""
    print_section("SCENARIO 3: Multiple Turns (Building Context)")
    
    questions = [
        "Who attended those meetings?",
        "What action items were assigned?",
        "When is the next meeting scheduled?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Turn {i} ---")
        make_chat_request(question, conversation_id=conversation_id)
        time.sleep(1)  # Small delay between requests
    
    print("\n💡 Each turn should:")
    print("   - Load conversation history")
    print("   - Use context from previous messages")
    print("   - Show increasing conversation_messages_used count")

def test_scenario_4_cache_hit(conversation_id: str):
    """Test conversation cache."""
    print_section("SCENARIO 4: Cache Test (Repeat Same Question)")
    
    # First request - will load from DB and cache
    print("First request (DB load + cache set):")
    response1 = make_chat_request(
        "Summarize what we discussed so far",
        conversation_id=conversation_id
    )
    
    time.sleep(1)
    
    # Second request - should hit cache
    print("\n\nSecond request (should hit cache):")
    response2 = make_chat_request(
        "Give me an overview of the discussion",
        conversation_id=conversation_id
    )
    
    print("\n💡 Second request should be faster due to cached conversation history")

def main():
    """Run all test scenarios."""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║             CONVERSATION MEMORY LIVE TESTING                             ║
║                                                                          ║
║  This script tests the conversation memory features:                    ║
║  1. Auto-load conversation history from database                        ║
║  2. Smart context window management                                     ║
║  3. Conversation caching                                                ║
║  4. Context-aware query rewriting                                       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            # Try root health endpoint
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code != 200:
                print("❌ Server not responding properly")
                return
    except requests.exceptions.RequestException:
        print("❌ Server not running. Start it with:")
        print("   uvicorn app.main:app --reload")
        return
    
    print("✅ Server is running\n")
    
    # Run test scenarios
    try:
        conversation_id = test_scenario_1_new_conversation()
        
        if conversation_id:
            test_scenario_2_follow_up(conversation_id)
            test_scenario_3_multiple_turns(conversation_id)
            test_scenario_4_cache_hit(conversation_id)
        
        print_section("✨ TESTING COMPLETE")
        print("""
Key things to look for in server logs:
- "Loaded X messages from DB for conversation" (first load)
- "Loaded X messages from cache" (subsequent loads)
- "Compressed history from X to Y messages"
- "Conversation has X messages, applying summarization" (if >20 messages)
- Query rewriting with conversation context
        """)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

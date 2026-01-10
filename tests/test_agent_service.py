"""
Test script for AgentService - Tool Intent Detection
"""

import pytest
from app.services.agent_service import get_agent_service, Intent


@pytest.mark.asyncio
async def test_intent_detection():
    """Test the agent's ability to detect tool use vs. chat."""
    
    agent = get_agent_service()
    
    # Test cases
    test_queries = [
        # Should trigger INFOGRAPHIC tool
        ("Create an infographic about Q3 sales performance", "infographic_generator"),
        ("Make me a visual showing our growth metrics", "infographic_generator"),
        ("Visualize the key insights from last week's board meeting", "infographic_generator"),
        
        # Should trigger GHOSTWRITER tool
        ("Write a LinkedIn post about our new AI features", "content_writer"),
        ("Draft an email to the team summarizing the sprint", "content_writer"),
        
        # Should be CHAT (no tool)
        ("What were the key decisions from yesterday's meeting?", None),
        ("Tell me about the Q3 sales results", None),
        ("Who attended the board meeting last week?", None),
        ("How is our revenue trending?", None),
    ]
    
    print("\n" + "="*80)
    print("AGENT SERVICE - INTENT DETECTION TEST")
    print("="*80)
    print()
    
    passed = 0
    failed = 0
    
    for query, expected_tool in test_queries:
        print(f"Query: {query}")
        print("-" * 80)
        
        try:
            decision = await agent.decide(query)
            
            print(f"  Intent: {decision.intent.value}")
            print(f"  Tool: {decision.tool_name}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Reasoning: {decision.reasoning}")
            
            # Check if result matches expectation
            if expected_tool is None:
                # Should be CHAT
                if decision.intent == Intent.CHAT:
                    print("  ✅ PASS - Correctly classified as CHAT")
                    passed += 1
                else:
                    print(f"  ❌ FAIL - Expected CHAT, got {decision.intent.value}")
                    failed += 1
            else:
                # Should be TOOL with specific tool_name
                if decision.intent == Intent.TOOL and decision.tool_name == expected_tool:
                    print(f"  ✅ PASS - Correctly routed to {expected_tool}")
                    passed += 1
                else:
                    print(f"  ❌ FAIL - Expected {expected_tool}, got {decision.tool_name}")
                    failed += 1
                    
                # Show extracted args if tool use
                if decision.is_tool_use():
                    print(f"  Tool Args: {decision.tool_args}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
        
        print()
    
    # Summary
    print("="*80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_queries)} tests")
    print("="*80)
    
    assert failed == 0, f"{failed} test(s) failed"


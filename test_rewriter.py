#!/usr/bin/env python3
"""Quick test for query rewriter logic."""

from app.services.query_rewriter import QueryRewriter

rewriter = QueryRewriter()

# Test cases: (query, history, expected_needs_rewrite)
test_cases = [
    # Standalone queries - should NOT be rewritten (new topics)
    ("Who are splashers", [{"role": "user", "content": "What is advancly?"}], False),
    ("What is Greenhouse?", [{"role": "user", "content": "Tell me about SME"}], False),
    ("Tell me about John", [{"role": "user", "content": "What is the budget?"}], False),
    ("Who is Bunmi?", [{"role": "user", "content": "What is advancly?"}], False),
    ("What are the key metrics?", [{"role": "user", "content": "Tell me about sales"}], False),
    ("When is the next board meeting?", [{"role": "user", "content": "Something"}], False),
    ("How does Multigate work?", [{"role": "user", "content": "Tell me about exits"}], False),
    
    # Follow-up queries - SHOULD be rewritten (need context)
    ("continue", [{"role": "user", "content": "Tell me about mobile strategy"}], True),
    ("go on", [{"role": "assistant", "content": "Here is info about X"}], True),
    ("what about the budget?", [{"role": "user", "content": "What is our Q1 plan?"}], True),
    ("Who is leading it?", [{"role": "user", "content": "Tell me about project Alpha"}], True),
    ("explain more", [{"role": "assistant", "content": "The meeting discussed..."}], True),
    ("tell me more about that", [{"role": "user", "content": "Something"}], True),
    ("What happened with it?", [{"role": "user", "content": "Tell me about the deal"}], True),
    ("How did they handle it?", [{"role": "user", "content": "Tell me about negotiations"}], True),
    ("expand on this", [{"role": "assistant", "content": "The strategy involves..."}], True),
    
    # Edge cases
    ("more", [{"role": "user", "content": "Something"}], True),  # Very short, vague
    ("What?", [{"role": "user", "content": "Something"}], False),  # Short but standalone pattern
]

print("Testing needs_rewrite logic:")
print("=" * 70)
all_passed = True
for query, history, expected in test_cases:
    result = rewriter.needs_rewrite(query, history)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"{status} {query!r:35} -> {result!s:5} (expected: {expected})")

print("=" * 70)
if all_passed:
    print("All tests passed!")
else:
    print("Some tests failed!")

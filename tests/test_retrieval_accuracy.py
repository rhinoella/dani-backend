#!/usr/bin/env python3
"""
Retrieval Accuracy Test

This script tests whether the RAG system retrieves accurate, relevant chunks
based on user queries. It validates the retrieval quality improvements.

Usage:
    python scripts/test_retrieval_accuracy.py [--query "your question"]
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from app.embeddings.client import OllamaEmbeddingClient
from app.core.config import settings


class RetrievalTester:
    """Tests retrieval accuracy against real transcript data."""
    
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embedder = OllamaEmbeddingClient()
        self.collection = "meeting_transcripts"
    
    async def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection)
        return {
            "points": info.points_count,
            "vector_dim": info.config.params.vectors.size
        }
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks for a query."""
        # Generate query embedding with prefix
        query_vec = await self.embedder.embed_query(query)
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k,
            with_payload=True
        ).points
        
        return [
            {
                "score": r.score,
                "text": r.payload.get("text", ""),
                "speakers": r.payload.get("speakers", []),
                "doc_type": r.payload.get("doc_type", "unknown"),
                "transcript_id": r.payload.get("transcript_id", ""),
                "date": r.payload.get("date", ""),
            }
            for r in results
        ]
    
    async def test_query(self, query: str, expected_keywords: List[str] = None):
        """
        Test a single query and evaluate relevance.
        
        Args:
            query: The search query
            expected_keywords: Keywords that should appear in relevant results
        """
        print(f"\n{'='*70}")
        print(f"📝 QUERY: \"{query}\"")
        print(f"{'='*70}")
        
        results = await self.retrieve(query, top_k=5)
        
        if not results:
            print("❌ No results found!")
            return {"query": query, "passed": False, "reason": "No results"}
        
        relevance_scores = []
        
        for i, result in enumerate(results, 1):
            score = result["score"]
            text = result["text"]
            speakers = result["speakers"]
            
            # Quality indicator
            if score > 0.80:
                quality = "🟢 EXCELLENT"
            elif score > 0.65:
                quality = "🟡 GOOD"
            elif score > 0.50:
                quality = "🟠 FAIR"
            else:
                quality = "🔴 POOR"
            
            # Check for expected keywords if provided
            keyword_match = ""
            if expected_keywords:
                found = [kw for kw in expected_keywords if kw.lower() in text.lower()]
                if found:
                    keyword_match = f" | Keywords found: {found}"
                    relevance_scores.append(1)
                else:
                    relevance_scores.append(0)
            
            # Truncate text for display
            preview = text[:300].replace('\n', ' ')
            if len(text) > 300:
                preview += "..."
            
            print(f"\n{quality} Result {i} (score: {score:.4f}){keyword_match}")
            print(f"   Speakers: {speakers}")
            print(f"   Words: {len(text.split())}")
            print(f"   Text: \"{preview}\"")
        
        # Calculate pass/fail
        top_score = results[0]["score"]
        passed = top_score > 0.60
        
        if expected_keywords and relevance_scores:
            keyword_accuracy = sum(relevance_scores) / len(relevance_scores)
            print(f"\n📊 Keyword Accuracy: {keyword_accuracy:.0%}")
            passed = passed and keyword_accuracy > 0.4
        
        print(f"\n{'✅ PASSED' if passed else '❌ FAILED'} - Top score: {top_score:.4f}")
        
        return {
            "query": query,
            "passed": passed,
            "top_score": top_score,
            "results_count": len(results)
        }
    
    async def run_test_suite(self):
        """Run a comprehensive test suite with various query types."""
        
        print("\n" + "="*70)
        print("🧪 RETRIEVAL ACCURACY TEST SUITE")
        print("="*70)
        
        # Get stats
        stats = await self.get_collection_stats()
        print(f"\n📊 Collection: {stats['points']} points, {stats['vector_dim']} dimensions")
        
        if stats['points'] < 10:
            print("⚠️  Not enough data yet. Wait for more ingestion to complete.")
            return
        
        # Define test cases
        test_cases = [
            {
                "query": "action items and tasks discussed",
                "expected_keywords": ["action", "task", "do", "need", "will", "should"],
                "description": "Finding action items"
            },
            {
                "query": "project timeline and deadlines",
                "expected_keywords": ["deadline", "timeline", "date", "week", "month", "when"],
                "description": "Finding timeline discussions"
            },
            {
                "query": "budget and financial discussion",
                "expected_keywords": ["budget", "cost", "money", "price", "dollar", "fund", "capital", "invest"],
                "description": "Finding financial topics"
            },
            {
                "query": "technical problems and issues",
                "expected_keywords": ["problem", "issue", "bug", "error", "fix", "broken", "challenge"],
                "description": "Finding technical discussions"
            },
            {
                "query": "team members and who is responsible",
                "expected_keywords": ["team", "responsible", "owner", "lead", "assign"],
                "description": "Finding responsibility assignments"
            },
        ]
        
        results = []
        
        for tc in test_cases:
            print(f"\n\n{'#'*70}")
            print(f"TEST: {tc['description']}")
            print(f"{'#'*70}")
            
            result = await self.test_query(tc["query"], tc.get("expected_keywords"))
            result["description"] = tc["description"]
            results.append(result)
        
        # Summary
        print("\n\n" + "="*70)
        print("📊 TEST SUITE SUMMARY")
        print("="*70)
        
        passed = sum(1 for r in results if r.get("passed", False))
        total = len(results)
        
        print(f"\n{'Test':<40} {'Score':<10} {'Status'}")
        print("-" * 60)
        
        for r in results:
            status = "✅ PASS" if r.get("passed") else "❌ FAIL"
            score = f"{r.get('top_score', 0):.4f}"
            print(f"{r.get('description', r['query'][:35]):<40} {score:<10} {status}")
        
        print("-" * 60)
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
        
        if passed == total:
            print("\n✅ ALL TESTS PASSED! Retrieval quality is good.")
        elif passed >= total * 0.6:
            print("\n🟡 Most tests passed. Retrieval quality is acceptable.")
        else:
            print("\n❌ Many tests failed. Retrieval quality needs improvement.")
        
        await self.embedder.close()
        return results
    
    async def interactive_test(self, query: str):
        """Run a single interactive query test."""
        await self.test_query(query)
        await self.embedder.close()


async def main():
    parser = argparse.ArgumentParser(description="Test retrieval accuracy")
    parser.add_argument("--query", "-q", type=str, help="Single query to test")
    parser.add_argument("--suite", "-s", action="store_true", help="Run full test suite")
    args = parser.parse_args()
    
    tester = RetrievalTester()
    
    if args.query:
        await tester.interactive_test(args.query)
    else:
        # Default: run test suite
        await tester.run_test_suite()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python
"""
Standalone Ollama Cloud Performance Verification Script

Run without pytest to get immediate performance metrics and diagnostics.

Usage:
  python scripts/test_ollama_cloud_performance.py
  python scripts/test_ollama_cloud_performance.py --quick
  python scripts/test_ollama_cloud_performance.py --load-test
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add app to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.core.config import settings
from app.llm.ollama import OllamaClient
from app.embeddings.client import OllamaEmbeddingClient


class ColoredOutput:
    """Helper for colored terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def success(text): print(f"{ColoredOutput.GREEN}✅ {text}{ColoredOutput.END}")
    @staticmethod
    def error(text): print(f"{ColoredOutput.RED}❌ {text}{ColoredOutput.END}")
    @staticmethod
    def warning(text): print(f"{ColoredOutput.YELLOW}⚠️  {text}{ColoredOutput.END}")
    @staticmethod
    def info(text): print(f"{ColoredOutput.BLUE}ℹ️  {text}{ColoredOutput.END}")
    @staticmethod
    def header(text): print(f"\n{ColoredOutput.BOLD}{ColoredOutput.HEADER}{text}{ColoredOutput.END}")
    @staticmethod
    def subheader(text): print(f"{ColoredOutput.CYAN}{ColoredOutput.BOLD}{text}{ColoredOutput.END}")


class PerformanceTester:
    """Main performance testing class."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.config_valid = False
        self.check_configuration()
    
    def check_configuration(self):
        """Verify Ollama configuration."""
        ColoredOutput.header("🔍 CONFIGURATION CHECK")
        
        print(f"  OLLAMA_ENV: {settings.OLLAMA_ENV}")
        print(f"  OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")
        print(f"  LLM_MODEL: {settings.LLM_MODEL}")
        print(f"  EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
        print(f"  API Key: {'✓ Present' if settings.OLLAMA_API_KEY else '✗ Missing'}")
        
        if settings.OLLAMA_ENV == "cloud" and not settings.OLLAMA_API_KEY:
            ColoredOutput.error("Cloud mode selected but API key missing!")
            return
        
        self.config_valid = True
        ColoredOutput.success("Configuration valid")
    
    async def test_connectivity(self):
        """Test basic connectivity."""
        ColoredOutput.header("🌐 CONNECTIVITY TEST")
        
        try:
            client = OllamaClient()
            start = time.time()
            result = await client.health_check()
            duration = (time.time() - start) * 1000
            
            if result:
                ColoredOutput.success(f"Health check passed ({duration:.0f}ms)")
                self.results.append({
                    "test": "health_check",
                    "status": "pass",
                    "duration_ms": duration
                })
            else:
                ColoredOutput.error("Health check returned False")
        except Exception as e:
            ColoredOutput.error(f"Connectivity test failed: {e}")
            self.results.append({
                "test": "health_check",
                "status": "fail",
                "error": str(e)
            })
    
    async def test_small_prompt(self):
        """Test with small prompt."""
        ColoredOutput.header("📝 SMALL PROMPT TEST")
        
        prompt = "What is artificial intelligence? Answer briefly."
        ColoredOutput.info(f"Prompt: {prompt}")
        
        try:
            client = OllamaClient()
            start = time.time()
            response = await client.generate(prompt)
            duration = (time.time() - start) * 1000
            
            tokens_per_sec = len(response.split()) / (duration / 1000)
            
            print(f"  Duration: {duration:.0f}ms")
            print(f"  Response length: {len(response)} chars")
            print(f"  Tokens/sec: {tokens_per_sec:.1f}")
            print(f"  Response: {response[:150]}...")
            
            ColoredOutput.success("Small prompt test complete")
            self.results.append({
                "test": "small_prompt",
                "status": "pass",
                "duration_ms": duration,
                "response_length": len(response),
                "tokens_per_sec": tokens_per_sec
            })
        except Exception as e:
            ColoredOutput.error(f"Small prompt test failed: {e}")
            self.results.append({
                "test": "small_prompt",
                "status": "fail",
                "error": str(e)
            })
    
    async def test_medium_prompt(self):
        """Test with medium prompt (realistic RAG scenario)."""
        ColoredOutput.header("📄 MEDIUM PROMPT TEST")
        
        prompt = """
        We held 5 meetings this week:
        - Monday: Discussed Q1 budget allocation ($2.5M), approved 15% increase
        - Tuesday: Hired 3 new engineers, onboarded 2 existing team members
        - Wednesday: Customer feedback session revealed 5 major feature requests
        - Thursday: Reviewed technical debt, identified 12 critical items to address
        - Friday: Board meeting, approved expansion into 2 new markets
        
        Summarize the key decisions and provide action items for next week.
        """
        
        ColoredOutput.info(f"Prompt length: {len(prompt)} chars")
        
        try:
            client = OllamaClient()
            start = time.time()
            response = await client.generate(prompt)
            duration = (time.time() - start) * 1000
            
            tokens_per_sec = len(response.split()) / (duration / 1000)
            
            print(f"  Duration: {duration:.0f}ms")
            print(f"  Response length: {len(response)} chars")
            print(f"  Tokens/sec: {tokens_per_sec:.1f}")
            print(f"  Response: {response[:200]}...")
            
            ColoredOutput.success("Medium prompt test complete")
            self.results.append({
                "test": "medium_prompt",
                "status": "pass",
                "duration_ms": duration,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "tokens_per_sec": tokens_per_sec
            })
        except Exception as e:
            ColoredOutput.error(f"Medium prompt test failed: {e}")
            self.results.append({
                "test": "medium_prompt",
                "status": "fail",
                "error": str(e)
            })
    
    async def test_huge_prompt(self):
        """Test with huge prompt (like full RAG context)."""
        ColoredOutput.header("🗂️  HUGE PROMPT TEST")
        
        # Build context similar to RAG retrieval with many documents
        context_items = [
            "Meeting 1: Q1 planning approved, budget $2.5M, 3 teams assigned",
            "Meeting 2: Engineering standup, 5 engineers onboarded, productivity +15%",
            "Meeting 3: Customer feedback, 5 feature requests, prioritize now",
            "Meeting 4: Board meeting, approved 2 new market expansions, hiring plan",
            "Meeting 5: Tech review, 12 debt items identified, 3 critical this quarter",
            "Meeting 6: Security audit planned Feb, compliance review Q2",
            "Meeting 7: Customer retention program launching March 1st",
            "Meeting 8: Cloud migration to AWS, target June completion",
        ] * 3  # Repeat to make it substantial
        
        prompt = f"""
        CONTEXT FROM MEETINGS:
        {chr(10).join(f'• {item}' for item in context_items)}
        
        Based on ALL of this context, provide:
        1. Executive summary (3-5 bullets)
        2. Top 5 action items with owners
        3. Critical blockers and risks
        4. Timeline for next 30 days
        5. Resource requirements
        
        Be comprehensive and cite specific meetings.
        """
        
        ColoredOutput.info(f"Prompt length: {len(prompt)} chars (simulates heavy RAG context)")
        
        try:
            client = OllamaClient()
            start = time.time()
            response = await client.generate(prompt)
            duration = (time.time() - start) * 1000
            
            tokens_per_sec = len(response.split()) / (duration / 1000)
            
            print(f"  Duration: {duration:.0f}ms")
            print(f"  Response length: {len(response)} chars")
            print(f"  Tokens/sec: {tokens_per_sec:.1f}")
            print(f"  Response: {response[:250]}...")
            
            if duration > 8000:
                ColoredOutput.warning(f"Large prompt took {duration:.0f}ms - consider streaming")
            else:
                ColoredOutput.success("Huge prompt test complete")
            
            self.results.append({
                "test": "huge_prompt",
                "status": "pass",
                "duration_ms": duration,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "tokens_per_sec": tokens_per_sec
            })
        except asyncio.TimeoutError:
            ColoredOutput.error("Huge prompt test timed out - consider streaming or reducing context")
            self.results.append({
                "test": "huge_prompt",
                "status": "fail",
                "error": "timeout"
            })
        except Exception as e:
            ColoredOutput.error(f"Huge prompt test failed: {e}")
            self.results.append({
                "test": "huge_prompt",
                "status": "fail",
                "error": str(e)
            })
    
    async def test_streaming(self):
        """Test streaming response."""
        ColoredOutput.header("🌊 STREAMING TEST")
        
        prompt = "List 10 cloud architecture best practices with brief explanations."
        ColoredOutput.info(f"Prompt: {prompt}")
        
        try:
            client = OllamaClient()
            start = time.time()
            tokens = 0
            response = ""
            
            print("  Streaming response: ", end="", flush=True)
            async for token in client.generate_stream(prompt):
                response += token
                tokens += 1
                if tokens % 50 == 0:
                    print(".", end="", flush=True)
            print(" done")
            
            duration = (time.time() - start) * 1000
            
            print(f"  Duration: {duration:.0f}ms")
            print(f"  Tokens streamed: {tokens}")
            print(f"  Tokens/sec: {tokens / (duration / 1000):.1f}")
            print(f"  Response: {response[:200]}...")
            
            ColoredOutput.success("Streaming test complete")
            self.results.append({
                "test": "streaming",
                "status": "pass",
                "duration_ms": duration,
                "tokens_streamed": tokens,
                "tokens_per_sec": tokens / (duration / 1000)
            })
        except Exception as e:
            ColoredOutput.error(f"Streaming test failed: {e}")
            self.results.append({
                "test": "streaming",
                "status": "fail",
                "error": str(e)
            })
    
    async def test_embeddings(self):
        """Test embedding generation."""
        ColoredOutput.header("🧩 EMBEDDING TEST")
        
        texts = [
            "What are the key decisions from recent meetings?",
            "How should we prioritize Q1 work items?",
            "What is our budget allocation for engineering?",
            "Summarize the technical debt identified",
            "List the new market expansion opportunities",
        ]
        
        ColoredOutput.info(f"Embedding {len(texts)} documents")
        
        try:
            embedder = OllamaEmbeddingClient()
            start = time.time()
            embeddings = await embedder.embed_documents(texts, batch_size=3)
            duration = (time.time() - start) * 1000
            
            docs_per_sec = len(embeddings) / (duration / 1000)
            
            print(f"  Duration: {duration:.0f}ms")
            print(f"  Docs embedded: {len(embeddings)}")
            print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
            print(f"  Docs/sec: {docs_per_sec:.1f}")
            
            ColoredOutput.success("Embedding test complete")
            self.results.append({
                "test": "embeddings",
                "status": "pass",
                "duration_ms": duration,
                "documents": len(embeddings),
                "docs_per_sec": docs_per_sec,
                "embedding_dim": len(embeddings[0]) if embeddings else 0
            })
        except Exception as e:
            ColoredOutput.error(f"Embedding test failed: {e}")
            self.results.append({
                "test": "embeddings",
                "status": "fail",
                "error": str(e)
            })
    
    async def test_load_simulation(self):
        """Simulate load with concurrent requests."""
        ColoredOutput.header("⚡ LOAD SIMULATION TEST")
        
        ColoredOutput.info("Sending 3 concurrent prompts")
        
        prompts = [
            "Explain machine learning in 2 sentences.",
            "What is cloud computing? Answer briefly.",
            "Describe DevOps practices concisely.",
        ]
        
        try:
            client = OllamaClient()
            start = time.time()
            
            # Run requests concurrently
            tasks = [client.generate(p) for p in prompts]
            responses = await asyncio.gather(*tasks)
            
            duration = (time.time() - start) * 1000
            total_chars = sum(len(r) for r in responses)
            
            print(f"  Duration: {duration:.0f}ms (concurrent)")
            print(f"  Requests: {len(responses)}")
            print(f"  Total response chars: {total_chars}")
            print(f"  Average response: {total_chars // len(responses)} chars")
            
            ColoredOutput.success("Load simulation complete")
            self.results.append({
                "test": "concurrent_load",
                "status": "pass",
                "duration_ms": duration,
                "concurrent_requests": len(responses),
                "total_response_chars": total_chars
            })
        except Exception as e:
            ColoredOutput.error(f"Load simulation failed: {e}")
            self.results.append({
                "test": "concurrent_load",
                "status": "fail",
                "error": str(e)
            })
    
    def print_summary(self):
        """Print test summary."""
        ColoredOutput.header("📊 PERFORMANCE SUMMARY")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("status") == "pass")
        failed = total - passed
        
        print(f"  Total tests: {total}")
        print(f"  Passed: {ColoredOutput.GREEN}{passed}{ColoredOutput.END}")
        print(f"  Failed: {ColoredOutput.RED}{failed}{ColoredOutput.END}")
        
        # Aggregate timing
        durations = [r.get("duration_ms", 0) for r in self.results if r.get("duration_ms")]
        if durations:
            print(f"  Total duration: {sum(durations):.0f}ms")
            print(f"  Average per test: {sum(durations) / len(durations):.0f}ms")
        
        # Save results
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "env": settings.OLLAMA_ENV,
                "base_url": settings.OLLAMA_BASE_URL,
                "model": settings.LLM_MODEL,
            },
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
            },
            "results": self.results,
        }
        
        report_file = Path(__file__).parent.parent.parent / "performance_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        ColoredOutput.info(f"Report saved: {report_file}")
    
    async def run_all(self, quick=False):
        """Run all tests."""
        if not self.config_valid:
            ColoredOutput.error("Configuration check failed - aborting tests")
            return
        
        self.start_time = time.time()
        
        await self.test_connectivity()
        await self.test_small_prompt()
        await self.test_medium_prompt()
        
        if not quick:
            await self.test_huge_prompt()
            await self.test_streaming()
            await self.test_embeddings()
            await self.test_load_simulation()
        
        self.print_summary()
        
        total_time = time.time() - self.start_time
        print(f"\n⏱️  Total test time: {total_time:.1f} seconds")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ollama Cloud Performance")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--load-test", action="store_true", help="Focus on load testing")
    args = parser.parse_args()
    
    ColoredOutput.header("🚀 OLLAMA CLOUD PERFORMANCE TESTER")
    
    tester = PerformanceTester()
    
    try:
        await tester.run_all(quick=args.quick)
    except KeyboardInterrupt:
        ColoredOutput.warning("Tests interrupted by user")
    except Exception as e:
        ColoredOutput.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

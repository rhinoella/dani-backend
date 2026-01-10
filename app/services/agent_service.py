"""
Agent Service for tool routing and execution.

Detects when user queries require tool use and orchestrates tool execution.
"""

from __future__ import annotations

import logging
import json
import re
from typing import Dict, Any, Optional
from enum import Enum

from app.llm.ollama import OllamaClient
from app.llm.prompts.agent import (
    TOOL_INTENT_PROMPT,
    TOOL_ARGS_PROMPT,
    TOOL_SCHEMAS,
)

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """User intent classification."""
    CHAT = "CHAT"
    TOOL = "TOOL"


class ToolDecision:
    """Represents a tool routing decision."""
    
    def __init__(
        self,
        intent: Intent,
        tool_name: Optional[str] = None,
        confidence: float = 0.0,
        reasoning: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
    ):
        self.intent = intent
        self.tool_name = tool_name
        self.confidence = confidence
        self.reasoning = reasoning
        self.tool_args = tool_args or {}
    
    def is_tool_use(self) -> bool:
        """Check if this is a tool use intent."""
        return self.intent == Intent.TOOL and self.tool_name is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging/debugging."""
        return {
            "intent": self.intent.value,
            "tool_name": self.tool_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "tool_args": self.tool_args,
        }


class AgentService:
    """
    Agentic routing service.
    
    Determines when to use tools vs. normal RAG chat.
    """
    
    def __init__(self):
        self.llm = OllamaClient()
        # Confidence threshold for tool use (be conservative)
        self.confidence_threshold = 0.7
    
    async def decide(self, query: str) -> ToolDecision:
        """
        Analyze user query and decide if a tool should be used.
        
        Args:
            query: User's input query
            
        Returns:
            ToolDecision with intent and optional tool routing
        """
        logger.info(f"[AGENT] Analyzing query: {query[:100]}...")
        
        # Step 1: Classify intent
        try:
            intent_result = await self._classify_intent(query)
        except Exception as e:
            logger.error(f"[AGENT] Intent classification failed: {e}")
            # Fallback to CHAT on error
            return ToolDecision(
                intent=Intent.CHAT,
                reasoning=f"Intent classification failed: {str(e)}"
            )
        
        # If not a tool use, return immediately
        if intent_result["intent"] != "TOOL" or not intent_result.get("tool_name"):
            return ToolDecision(
                intent=Intent.CHAT,
                confidence=intent_result.get("confidence", 0.0),
                reasoning=intent_result.get("reasoning", "Classified as chat query")
            )
        
        # Check confidence threshold
        confidence = intent_result.get("confidence", 0.0)
        if confidence < self.confidence_threshold:
            logger.info(
                f"[AGENT] Tool confidence too low ({confidence:.2f} < {self.confidence_threshold}), "
                "falling back to CHAT"
            )
            return ToolDecision(
                intent=Intent.CHAT,
                confidence=confidence,
                reasoning=f"Tool confidence below threshold ({confidence:.2f})"
            )
        
        # Step 2: Extract tool arguments
        # Normalize tool name to lowercase (LLM may return UPPERCASE)
        tool_name = intent_result["tool_name"].lower()
        try:
            tool_args = await self._extract_tool_args(query, tool_name)
        except Exception as e:
            logger.error(f"[AGENT] Argument extraction failed: {e}")
            return ToolDecision(
                intent=Intent.CHAT,
                reasoning=f"Argument extraction failed: {str(e)}"
            )
        
        logger.info(
            f"[AGENT] Tool decision: {tool_name} (confidence: {confidence:.2f})"
        )
        
        return ToolDecision(
            intent=Intent.TOOL,
            tool_name=tool_name,
            confidence=confidence,
            reasoning=intent_result.get("reasoning", ""),
            tool_args=tool_args,
        )
    
    async def _classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user intent using LLM.
        
        Returns:
            Dict with: intent, tool_name, confidence, reasoning
        """
        prompt = TOOL_INTENT_PROMPT.format(query=query)
        
        # Use a shorter context for fast classification
        response = await self.llm.generate(prompt=prompt)
        
        # Parse JSON response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)
            response = response.rstrip("`").strip()
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"[AGENT] Failed to parse intent JSON: {e}")
            logger.error(f"[AGENT] Response was: {response[:200]}...")
            raise ValueError(f"Invalid JSON from intent classification: {str(e)}")
    
    async def _extract_tool_args(
        self,
        query: str,
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Extract tool arguments from query.
        
        Args:
            query: User's query
            tool_name: Name of the tool to extract args for
            
        Returns:
            Dict of tool arguments
        """
        schema = TOOL_SCHEMAS.get(tool_name)
        if not schema:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        prompt = TOOL_ARGS_PROMPT.format(
            tool_name=tool_name,
            query=query,
            schema=schema
        )
        
        response = await self.llm.generate(prompt=prompt)
        
        # Parse JSON response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)
            response = response.rstrip("`").strip()
        
        try:
            args = json.loads(response)
            return args
        except json.JSONDecodeError as e:
            logger.error(f"[AGENT] Failed to parse tool args JSON: {e}")
            logger.error(f"[AGENT] Response was: {response[:200]}...")
            raise ValueError(f"Invalid JSON from argument extraction: {str(e)}")


# Singleton instance
_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """Get or create agent service singleton."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service

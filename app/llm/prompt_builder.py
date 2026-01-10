from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime

from app.persona.system_prompt import DANI_SYSTEM_PROMPT
from app.persona.templates import get_template


class PromptBuilder:
    """
    Builds persona-enforced, grounded prompts with optional structured output formats.
    
    Enhanced with:
    - Source numbering for citation
    - Speaker information per chunk
    - Increased context length (2500 chars per chunk)
    - Chronological ordering option
    """
    
    def _format_date(self, date_value: Any) -> str:
        """Format date for display, handling various input formats."""
        if not date_value:
            return "Unknown date"
        
        # Handle Unix timestamp (milliseconds)
        if isinstance(date_value, (int, float)):
            try:
                if date_value > 1e12:  # Milliseconds
                    date_value = date_value / 1000
                dt = datetime.fromtimestamp(date_value)
                return dt.strftime("%B %d, %Y")
            except Exception:
                return "Unknown date"
        
        # Handle string dates
        if isinstance(date_value, str):
            # Try to parse ISO format
            try:
                if "T" in date_value:
                    dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                    return dt.strftime("%B %d, %Y")
                return date_value
            except Exception:
                return date_value
        
        return str(date_value)

    def build_chat_prompt(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]],
        output_format: Optional[str] = None,
        order_by_date: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build a prompt with RAG context and optional conversation history.
        
        Args:
            query: User's current question
            chunks: Retrieved context chunks from vector store
            output_format: Optional output format template
            order_by_date: Sort chunks by date
            conversation_history: List of {"role": "user"|"assistant", "content": str}
        """
        if not chunks:
            context = "No relevant meeting notes found."
        else:
            # Optionally sort by date (newest first or chronological)
            if order_by_date:
                chunks = sorted(chunks, key=lambda c: c.get("date", 0), reverse=True)
            
            blocks = []
            for i, c in enumerate(chunks, 1):
                title = c.get("title", "Unknown meeting")
                date = self._format_date(c.get("date"))
                speakers = c.get("speakers", [])
                speakers_str = ", ".join(speakers) if speakers else "Unknown speakers"
                
                # Increased context: ~625 tokens (2500 chars) per chunk for better comprehension
                text = c.get("text", "").strip()[:2500]
                
                # Format with source number for citation
                block = f"[Source {i}] {title}\nDate: {date}\nSpeakers: {speakers_str}\n\n{text}"
                blocks.append(block)

            context = "\n\n" + "="*50 + "\n\n".join(blocks)
        
        # Add output format instructions if specified
        format_instructions = ""
        if output_format:
            template = get_template(output_format)
            if template:
                format_instructions = f"\n\nOUTPUT FORMAT REQUIRED:\n{template}"
        
        grounding_instruction = """
IMPORTANT INSTRUCTIONS:
- Base your answer on the provided meeting sources and conversation history below
- When referencing specific information, refer to the meeting context naturally (e.g., "In the discussion with [Company/Person]...", "During the [Meeting Title]...").
- You can mention "source [number]" if needed for ambiguity, but prefer using the meeting title or context description.
- If the sources don't contain enough information to answer, say "I don't have enough information about this in the meeting notes"
- Do NOT make up information that isn't in the sources
- RESPONSE FORMAT: Write in PLAIN TEXT paragraphs only. DO NOT use markdown, bullet points, or lists.
- Explain discussions in detail with a narrative flow."""

        # Format conversation history if provided (limit to last 6 exchanges for context window)
        history_section = ""
        if conversation_history:
            # Take only the last 6 messages to keep prompt size manageable
            recent_history = conversation_history[-6:]
            history_lines = []
            for msg in recent_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:500]  # Truncate long messages
                history_lines.append(f"{role}: {content}")
            
            if history_lines:
                history_section = f"""
CONVERSATION HISTORY:
{chr(10).join(history_lines)}

---
"""

        return f"""
{DANI_SYSTEM_PROMPT}

MEETING SOURCES:
{context}

{grounding_instruction}
{history_section}
User Question:
{query}
{format_instructions}

Answer:
""".strip()

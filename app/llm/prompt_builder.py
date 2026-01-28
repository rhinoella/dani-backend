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
            context = "No relevant meeting notes or documents found."
        else:
            # Optionally sort by date (newest first or chronological)
            if order_by_date:
                chunks = sorted(chunks, key=lambda c: c.get("date", 0), reverse=True)
            
            # Separate documents from meetings
            meeting_blocks = []
            document_blocks = []
            
            for i, c in enumerate(chunks, 1):
                title = c.get("title", "Unknown")
                is_document = c.get("document_source", False) or c.get("doc_type") == "document"
                
                # Increased context: ~1250 tokens (5000 chars) per chunk for better comprehension
                text = c.get("text", "").strip()[:5000]
                
                if is_document:
                    # Format for documents
                    doc_id = c.get("document_id", "")
                    block = f"[Document {i}] {title}\n\n{text}"
                    document_blocks.append(block)
                else:
                    # Format for meetings (with date and speakers)
                    date = self._format_date(c.get("date"))
                    speakers = c.get("speakers", [])
                    speakers_str = ", ".join(speakers) if speakers else "Unknown speakers"
                    block = f"[Meeting {i}] {title}\nDate: {date}\nSpeakers: {speakers_str}\n\n{text}"
                    meeting_blocks.append(block)

            # Build combined context
            context_parts = []
            if meeting_blocks:
                context_parts.append("MEETING TRANSCRIPTS:\n" + "\n\n" + "="*50 + "\n\n".join(meeting_blocks))
            if document_blocks:
                context_parts.append("UPLOADED DOCUMENTS:\n" + "\n\n" + "="*50 + "\n\n".join(document_blocks))
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant content found."
        
        # Add output format instructions if specified
        format_instructions = ""
        if output_format:
            template = get_template(output_format)
            if template:
                format_instructions = f"\n\nOUTPUT FORMAT REQUIRED:\n{template}"
        
        grounding_instruction = """
IMPORTANT INSTRUCTIONS:
- FOCUS ON THE CURRENT QUESTION: Answer based ONLY on the KNOWLEDGE BASE SOURCES provided above that are relevant to the current question.
- TOPIC SWITCHING: If the user asks about a NEW topic (different from conversation history), focus ENTIRELY on the new topic. Do NOT reference or blend in information from previous topics discussed in the conversation.
- When referencing meeting information, use natural language (e.g., "In the discussion with [Company/Person]...", "During the [Meeting Title]...").
- When referencing document information, cite the document name (e.g., "According to the [Document Name]...", "The document states...").
- If the sources don't contain enough information to answer, say "I don't have enough information about this in the available content"
- Do NOT make up information that isn't in the sources
- RESPONSE FORMAT: Write in PLAIN TEXT paragraphs only. DO NOT use markdown, bullet points, or lists.
- Explain discussions in detail with a narrative flow. Provide comprehensive answers that fully address the question.
- CONVERSATION HISTORY is provided only for context (to understand pronouns like "it", "they", follow-up questions). It is NOT a source of information - use only the KNOWLEDGE BASE SOURCES above."""

        # Format conversation history if provided (limit to last 4 exchanges - reduced to avoid topic bleeding)
        history_section = ""
        if conversation_history:
            # Take only the last 4 messages - enough for pronoun resolution but not overwhelming
            recent_history = conversation_history[-4:]
            history_lines = []
            for msg in recent_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                # Truncate assistant messages more aggressively to reduce topic bleeding
                max_len = 200 if msg.get("role") == "assistant" else 300
                content = msg.get("content", "")[:max_len]
                if len(msg.get("content", "")) > max_len:
                    content += "..."
                history_lines.append(f"{role}: {content}")
            
            if history_lines:
                history_section = f"""
CONVERSATION CONTEXT (for understanding follow-up questions only - NOT a source of information):
{chr(10).join(history_lines)}

---
"""

        return f"""
{DANI_SYSTEM_PROMPT}

KNOWLEDGE BASE SOURCES (USE THESE TO ANSWER):
{context}

{grounding_instruction}
{history_section}
User Question:
{query}
{format_instructions}

Answer:
""".strip()

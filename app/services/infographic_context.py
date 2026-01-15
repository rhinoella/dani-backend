"""
Enhanced Infographic Context Builder.

Fixes the infographic generator's context issues by:
1. Including conversation history context
2. Aggregating multiple RAG chunks into coherent context
3. Extracting specific data points for visualization
4. Building rich prompts for image generation
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from app.llm.ollama import OllamaClient
from app.services.retrieval_service import RetrievalService
from app.services.enhanced_retrieval import EnhancedRetriever, EnhancedRetrievalConfig
from app.schemas.retrieval import MetadataFilter

logger = logging.getLogger(__name__)


class InfographicContextBuilder:
    """
    Builds rich context for infographic generation.
    
    The problem: Infographics were getting poor context because:
    1. No conversation history was passed
    2. RAG chunks were used directly without aggregation
    3. Data extraction wasn't tailored for visualization
    
    This builder:
    1. Aggregates conversation context + RAG results
    2. Extracts visualization-specific data (numbers, comparisons, timelines)
    3. Structures data optimally for image generation prompts
    """
    
    DATA_EXTRACTION_PROMPT = """You are an expert data visualization specialist extracting data for a HIGH-IMPACT INFOGRAPHIC.

USER REQUEST: {request}

CONVERSATION CONTEXT:
{conversation_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

Extract data that can be VISUALIZED in an infographic:

{{
    "main_topic": "The primary subject - be specific (e.g., 'Q4 2025 Revenue Performance')",
    "headline": "IMPACTFUL 5-8 word headline with SPECIFIC DATA. MUST include a number or percentage. Example: 'Q4 Revenue Surges 35% to $2.5M'",
    "subtitle": "Supporting context with key metric (10-15 words)",
    
    "statistics": [
        {{"value": "$2.5M", "label": "Q4 Revenue", "trend": "up", "icon": "💰"}},
        {{"value": "35%", "label": "QoQ Growth", "trend": "up", "icon": "📈"}},
        // EXACTLY 4-5 stats. EVERY stat MUST have:
        // - A NUMERIC value with unit ($, %, x, etc.) - NEVER vague words like 'High' or 'Good'
        // - A clear descriptive label
        // - An emoji icon (💰📈🎯👥🌏📊🚀💡✅🏆)
    ],
    
    "key_points": [
        "Specific insight with NUMBER: 'Enterprise sales grew 45% YoY'",
        "Actionable finding with DATA: 'APAC outperformed at 52% growth'",
        // EXACTLY 3-4 points. Each MUST:
        // - Be 8-15 words
        // - Include specific numbers/percentages from context
        // - Start with capital letter, action-oriented
    ],
    
    "comparisons": [
        {{"item1": "Q3 Revenue: $1.85M", "item2": "Q4 Revenue: $2.5M", "metric": "35% increase"}},
        // Any before/after, A vs B comparisons with NUMBERS
    ],
    
    "timeline": [
        {{"date": "Dec 2025", "event": "Q4 revenue hit $2.5M", "icon": "🎯"}},
        // Chronological events if relevant
    ],
    
    "categories": [
        {{"name": "Enterprise", "value": "$1.5M", "percentage": "60%"}},
        {{"name": "SMB", "value": "$1M", "percentage": "40%"}},
        // Categorical breakdowns with percentages (for pie/bar charts)
    ],
    
    "data_source": "Meeting/document name with date"
}}

CRITICAL QUALITY RULES:
1. NEVER use vague words: 'High', 'Good', 'Strong', 'Significant', 'Improved'
2. ALWAYS use specific numbers from the context
3. EVERY statistic needs a numeric value AND an emoji icon
4. HEADLINE must grab attention with a specific achievement
5. Scan the context for ALL numbers: $, %, counts, dates, scores

Output ONLY valid JSON:"""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm: Optional[OllamaClient] = None,
    ):
        self.retrieval = retrieval_service or RetrievalService()
        self.llm = llm or OllamaClient()
        
        # Use enhanced retrieval for better context
        self.enhanced_retriever = EnhancedRetriever(
            self.retrieval,
            EnhancedRetrievalConfig(
                expand_queries=True,
                num_query_variants=2,
                use_reranking=True,
                compress_chunks=False,  # Keep full chunks for infographics
                min_relevance_score=0.55,
            )
        )
    
    async def build_context(
        self,
        request: str,
        topic: Optional[str] = None,
        doc_type: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for infographic generation.
        
        Args:
            request: User's infographic request
            topic: Specific topic to search (defaults to request)
            doc_type: Filter by document type
            conversation_history: Past conversation for context
        
        Returns:
            Dict with:
            - structured_data: Extracted visualization data
            - raw_context: Original context used
            - sources: Source documents
            - confidence: Retrieval confidence
        """
        search_query = topic or request
        
        # Build conversation context string
        conversation_context = self._format_conversation_history(conversation_history)
        
        # Retrieve relevant chunks with enhanced retrieval
        metadata_filter = None
        if doc_type and doc_type != "all":
            metadata_filter = MetadataFilter(doc_type=doc_type)
        
        retrieval_result = await self.enhanced_retriever.retrieve(
            query=search_query,
            limit=8,  # More chunks for infographics
            metadata_filter=metadata_filter,
            conversation_history=conversation_history,
        )
        
        chunks = retrieval_result["chunks"]
        confidence = retrieval_result["confidence"]
        
        if not chunks:
            return {
                "error": "No relevant context found for the infographic",
                "suggestion": "Try a different topic or broader search terms",
            }
        
        # Format RAG context
        rag_context = self._format_chunks(chunks)
        
        # Build sources list
        sources = self._extract_sources(chunks)
        
        # Extract structured data using LLM
        structured_data = await self._extract_visualization_data(
            request=request,
            conversation_context=conversation_context,
            rag_context=rag_context,
        )
        
        if "error" in structured_data:
            return structured_data
        
        return {
            "structured_data": structured_data,
            "raw_context": {
                "conversation": conversation_context,
                "rag": rag_context,
            },
            "sources": sources,
            "confidence": confidence,
            "chunks_used": len(chunks),
        }
    
    def _format_conversation_history(
        self, 
        history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Format conversation history for context."""
        if not history:
            return "No previous conversation context."
        
        # Take recent relevant messages
        recent = history[-6:]  # Last 6 messages
        
        lines = []
        for msg in recent:
            role = msg.get("role", "user").title()
            content = msg.get("content", "")[:500]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format RAG chunks into readable context."""
        context_parts = []
        
        for chunk in chunks:
            title = chunk.get("title") or chunk.get("meeting_title") or "Untitled"
            text = chunk.get("text", "")
            date = chunk.get("date", "")
            speakers = chunk.get("speakers", [])
            score = chunk.get("combined_score", chunk.get("score", 0))
            
            # Format speaker info
            speaker_info = f" (Speakers: {', '.join(speakers)})" if speakers else ""
            
            context_parts.append(
                f"From '{title}'{speaker_info} [{date}] (relevance: {score:.2f}):\n{text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from chunks."""
        sources = []
        seen_titles = set()
        
        for chunk in chunks:
            title = chunk.get("title") or chunk.get("meeting_title") or "Untitled"
            
            if title not in seen_titles:
                seen_titles.add(title)
                sources.append({
                    "title": title,
                    "date": chunk.get("date"),
                    "speakers": chunk.get("speakers", []),
                    "doc_type": chunk.get("doc_type", "meeting"),
                    "transcript_id": chunk.get("transcript_id"),
                })
        
        return sources
    
    async def _extract_visualization_data(
        self,
        request: str,
        conversation_context: str,
        rag_context: str,
    ) -> Dict[str, Any]:
        """Extract structured data for visualization."""
        import json
        
        prompt = self.DATA_EXTRACTION_PROMPT.format(
            request=request,
            conversation_context=conversation_context,
            rag_context=rag_context[:4000],  # Limit context size
        )
        
        try:
            response = await self.llm.generate(prompt)
            
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                parts = response.split("```")
                if len(parts) > 1:
                    response = parts[1]
                    if response.startswith("json"):
                        response = response[4:]
            
            data = json.loads(response)
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse visualization data: {e}")
            logger.debug(f"Raw response: {response[:500] if response else 'empty'}")
            return {"error": "Failed to extract structured data"}
        except Exception as e:
            logger.error(f"Visualization data extraction failed: {e}")
            return {"error": str(e)}


def build_enhanced_image_prompt(
    structured_data: Dict[str, Any],
    style: str = "modern",
    conversation_context: Optional[str] = None,
) -> str:
    """
    Build an enhanced image generation prompt with full context.
    
    This creates a more detailed prompt that gives the image generator
    better understanding of what to create.
    """
    
    headline = structured_data.get("headline", "Infographic")
    subtitle = structured_data.get("subtitle", "")
    main_topic = structured_data.get("main_topic", "")
    
    # Format statistics
    stats = structured_data.get("statistics", [])
    stats_text = ""
    if stats:
        stats_lines = []
        for s in stats[:6]:
            trend_icon = {"up": "📈", "down": "📉", "neutral": "➡️"}.get(s.get("trend", "neutral"), "📊")
            stats_lines.append(f"- {trend_icon} {s.get('value')}: {s.get('label')}")
        stats_text = "\n".join(stats_lines)
    
    # Format key points
    key_points = structured_data.get("key_points", [])
    points_text = "\n".join([f"• {p}" for p in key_points[:5]]) if key_points else ""
    
    # Format comparisons
    comparisons = structured_data.get("comparisons", [])
    comparison_text = ""
    if comparisons:
        comp_lines = [f"- {c.get('item1')} vs {c.get('item2')} ({c.get('metric')})" for c in comparisons[:3]]
        comparison_text = "COMPARISONS:\n" + "\n".join(comp_lines)
    
    # Format timeline
    timeline = structured_data.get("timeline", [])
    timeline_text = ""
    if timeline:
        time_lines = [f"- {t.get('date')}: {t.get('event')}" for t in timeline[:5]]
        timeline_text = "TIMELINE:\n" + "\n".join(time_lines)
    
    # Format categories (for pie/bar charts)
    categories = structured_data.get("categories", [])
    categories_text = ""
    if categories:
        cat_lines = [f"- {c.get('name')}: {c.get('value')} ({c.get('percentage', 'N/A')})" for c in categories[:6]]
        categories_text = "CATEGORIES/BREAKDOWN:\n" + "\n".join(cat_lines)
    
    # Style instructions
    style_instructions = {
        "modern": "Clean modern design with gradient backgrounds, sans-serif typography, blue/white color scheme",
        "corporate": "Professional corporate style, navy/gray colors, formal typography, executive quality",
        "minimal": "Minimalist design, white space, simple icons, black/white with accent color",
        "vibrant": "Colorful energetic design, bold colors, dynamic layout, eye-catching visuals",
        "dark": "Dark theme, dark background, neon accents, high contrast, tech aesthetic",
    }.get(style, "modern professional design")
    
    # Build comprehensive prompt
    prompt = f"""Create a professional INFOGRAPHIC visualization with:

TOPIC: {main_topic}
TITLE: {headline}
{f'SUBTITLE: {subtitle}' if subtitle else ''}

KEY STATISTICS TO VISUALIZE:
{stats_text if stats_text else '(Include relevant data visualizations)'}

KEY POINTS:
{points_text if points_text else '(Main takeaways)'}

{comparison_text}

{timeline_text}

{categories_text}

CONTEXT: This infographic should visualize data from {structured_data.get('data_source', 'internal meetings and documents')}.

VISUAL STYLE: {style_instructions}

DESIGN REQUIREMENTS:
1. Professional infographic layout (NOT a photo or illustration)
2. Clear visual hierarchy - title prominent at top
3. Statistics displayed with icons, charts, or visual elements
4. Use appropriate chart types:
   - Bar/column charts for comparisons
   - Line charts for timelines/trends
   - Pie/donut charts for proportions
   - Icon arrays for counts
5. Balanced composition with clear sections
6. All text must be clearly legible
7. Include visual separators between sections
8. Color-coded elements for easy scanning

Generate a complete, presentation-ready infographic image."""

    return prompt


# Singleton instance
_context_builder: Optional[InfographicContextBuilder] = None


def get_infographic_context_builder() -> InfographicContextBuilder:
    """Get or create the context builder singleton."""
    global _context_builder
    if _context_builder is None:
        _context_builder = InfographicContextBuilder()
    return _context_builder

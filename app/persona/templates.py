from __future__ import annotations

from typing import Dict, Callable


# Output format templates with specific instructions
SUMMARY_TEMPLATE = """
You must provide a SUMMARY of the discussion.

Format:
- Start with a 1-2 sentence executive summary
- Follow with 3-5 key points using bullet points
- Each point should be one concise sentence
- Focus on outcomes and decisions, not narration
- Be factual - do not add speculation

Example:
"The team decided to prioritize mobile development for Q1. Key points:
• Mobile app will launch by March 2025
• Budget approved for $500K
• Sarah will lead the mobile team
• Backend API must be ready by January"
"""

DECISIONS_TEMPLATE = """
You must extract DECISIONS that were made.

Format:
- List each decision as a numbered item
- Include WHO made the decision (if mentioned)
- Include WHAT was decided
- Include WHY if stated
- Only include EXPLICIT decisions, not implied ones
- If no decisions were made, respond with: "No explicit decisions were documented in this discussion."

Example:
"1. Bunmi decided to hire 3 engineers for the mobile team (need to scale faster)
2. Team agreed to use React Native over Flutter (better ecosystem)
3. Launch date set for March 15, 2025 (aligned with investor timeline)"
"""

INSIGHTS_TEMPLATE = """
You must extract STRATEGIC INSIGHTS and implications.

Format:
- Provide 3-5 strategic insights
- Each insight should connect facts to implications
- Focus on what this means for the business/strategy
- Avoid obvious observations
- Be specific, not generic

Example:
"• The shift to mobile-first indicates competitive pressure from fintech startups
• $500K budget suggests confidence in market timing despite recent funding challenges
• Choice of React Native over native development prioritizes speed-to-market over performance
• January API deadline creates dependency risk for Q1 launch"
"""

TASKS_TEMPLATE = """
You must extract ACTION ITEMS and follow-ups.

Format:
- List each task with: [WHO] - [WHAT] - [WHEN if mentioned]
- Be specific about the action
- Include deadlines if stated
- If no tasks were assigned, respond with: "No action items were explicitly assigned in this discussion."

Example:
"• Sarah - Hire 3 mobile engineers - by end of January
• Dev team - Complete backend API - by January 15
• Bunmi - Finalize vendor contracts - this week
• Marketing - Prepare launch campaign - by February 1"
"""

EMAIL_TEMPLATE = """
You must draft a PROFESSIONAL EMAIL based on the context.

Format:
- Subject line (clear and specific)
- Greeting (professional)
- 2-3 short paragraphs maximum
- Clear call-to-action if needed
- Professional sign-off
- Write in Bunmi's voice: direct, decisive, no fluff
- ONLY use information from the context - do NOT add external details

Tone: Professional, confident, concise.
"""

WHATSAPP_TEMPLATE = """
You must draft a WHATSAPP MESSAGE based on the context.

Format:
- 2-4 short sentences maximum
- Professional but conversational
- Use Bunmi's voice: direct, friendly, to-the-point
- Can use appropriate emojis sparingly (✅, 🚀, 💡)
- No formal greeting/sign-off needed
- ONLY use information from the context

Tone: Conversational yet professional.
"""

SLIDES_TEMPLATE = """
You must create a SLIDE OUTLINE with bullet points.

Format:
- Suggest 3-5 slides
- Each slide has: Title + 3-5 bullet points
- One clear idea per bullet
- Bullets should be short (max 10 words each)
- Focus on key facts and insights
- No slide should have more than 5 bullets

Example:
Slide 1: Mobile Strategy Overview
• Launching mobile app Q1 2025
• $500K budget approved
• React Native framework selected

Slide 2: Team & Timeline
• 3 engineers being hired
• Backend ready by Jan 15
• Launch date: March 15
"""

INFOGRAPHIC_TEMPLATE = """
You must produce structured INFOGRAPHIC CONTENT.

Format:
- Main headline (8 words max)
- 4-6 key stats or facts
- Each fact: [NUMBER/METRIC] - [WHAT IT MEANS]
- Keep text minimal and impactful
- Focus on quantifiable information
- Use only facts from the context

Example:
Headline: Mobile-First Strategy for 2025

📱 March 2025 - Launch Date
💰 $500K - Approved Budget
👥 3 Engineers - New Hires Needed
⚡ React Native - Tech Stack
🎯 Q1 Priority - Mobile Development
"""


# Template registry
TEMPLATES: Dict[str, str] = {
    "summary": SUMMARY_TEMPLATE,
    "decisions": DECISIONS_TEMPLATE,
    "insights": INSIGHTS_TEMPLATE,
    "tasks": TASKS_TEMPLATE,
    "email": EMAIL_TEMPLATE,
    "whatsapp": WHATSAPP_TEMPLATE,
    "slides": SLIDES_TEMPLATE,
    "infographic": INFOGRAPHIC_TEMPLATE,
}


def get_template(format_type: str) -> str:
    """Get template for specified output format."""
    return TEMPLATES.get(format_type, "")


def validate_output_format(output_format: str) -> bool:
    """Check if output format is supported."""
    return output_format in TEMPLATES

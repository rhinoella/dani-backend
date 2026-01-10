"""
Agent prompts for tool use detection and routing.

These prompts help DANI determine when to use tools and extract parameters.
"""

# Intent classification prompt - determines if a tool is needed
TOOL_INTENT_PROMPT = """You are an intent classifier for DANI, an AI assistant.

Analyze the user's query and determine if it requires using a TOOL or if it's a normal CHAT question.

Available Tools:
1. INFOGRAPHIC_GENERATOR - Creates visual infographics from data/insights
   - Trigger words: "create infographic", "make infographic", "visualize", "show me a chart", "create a visual"
   - Example: "Create an infographic about Q3 sales performance"

2. CONTENT_WRITER (Ghostwriter) - Generates written content (emails, posts, blogs)
   - Trigger words: "write email", "draft email", "draft post", "create blog", "write message", "draft a message", "compose"
   - Example: "Write a LinkedIn post about our new feature" or "Draft an email to the team"

User Query: {query}

Respond with ONLY a JSON object:
{{
    "intent": "TOOL" or "CHAT",
    "tool_name": "infographic_generator" or "content_writer" or null,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Rules:
- Use "TOOL" ONLY if explicitly requesting creation/generation
- Use "CHAT" for questions, clarifications, or general conversation
- Be conservative: when in doubt, use "CHAT"
"""

# Tool argument extraction prompt - extracts parameters for tool calls
TOOL_ARGS_PROMPT = """You are extracting arguments for the {tool_name} tool.

User Request: {query}

Extract the following parameters in JSON format:
{schema}

Output ONLY valid JSON that matches the schema.
If a parameter is not mentioned, use null or a reasonable default.
"""

# Infographic tool schema
INFOGRAPHIC_ARGS_SCHEMA = """{
    "request": "What the user wants the infographic to show (required)",
    "topic": "Specific topic/keyword to search for (optional, defaults to request)",
    "style": "modern | corporate | minimal | vibrant | dark (default: modern)",
    "doc_type": "meeting | email | document | note | all (default: all)"
}"""

# Ghostwriter tool schema
GHOSTWRITER_ARGS_SCHEMA = """{
    "content_type": "linkedin_post | email | blog_draft | tweet_thread | newsletter (required)",
    "request": "What the user wants written (required)",
    "topic": "Topic to search for context (optional)",
    "tone": "formal | casual | urgent | friendly (optional)",
    "doc_type": "meeting | email | document | note | all (default: all)"
}"""

TOOL_SCHEMAS = {
    "infographic_generator": INFOGRAPHIC_ARGS_SCHEMA,
    "content_writer": GHOSTWRITER_ARGS_SCHEMA,
}

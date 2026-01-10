from __future__ import annotations

import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Phrases that indicate potential hallucination or uncertainty
UNCERTAINTY_PHRASES = [
    "i think",
    "i believe",
    "probably",
    "might be",
    "could be",
    "i'm not sure",
    "i don't know",
    "it's possible",
    "as an ai",
    "as a language model",
    "i cannot",
    "i can't",
]

# Phrases that indicate the model is making things up
HALLUCINATION_INDICATORS = [
    "based on my training",
    "in my knowledge",
    "typically",
    "generally speaking",
    "in most cases",
]


class OutputValidator:
    """
    Validates LLM output for quality and safety.
    Ensures responses are grounded in evidence.
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def validate(self, text: str) -> Dict[str, Any]:
        """Validate and clean LLM output."""
        cleaned = text.strip()
        warnings: List[str] = []

        if not cleaned:
            logger.warning("Empty response from LLM")
            return {
                "answer": "I don't have a record of that discussion.",
                "validated": True,
                "warnings": ["empty_response"],
            }
        
        # Check for uncertainty phrases
        lower_text = cleaned.lower()
        for phrase in UNCERTAINTY_PHRASES:
            if phrase in lower_text:
                warnings.append(f"uncertainty_detected: {phrase}")
                logger.debug(f"Uncertainty phrase detected: {phrase}")
        
        # Check for potential hallucination indicators
        for phrase in HALLUCINATION_INDICATORS:
            if phrase in lower_text:
                warnings.append(f"hallucination_risk: {phrase}")
                logger.warning(f"Potential hallucination indicator: {phrase}")
        
        # In strict mode, reject responses with hallucination indicators
        if self.strict_mode and any("hallucination_risk" in w for w in warnings):
            logger.warning("Strict mode: Rejecting response with hallucination risk")
            return {
                "answer": "I don't have sufficient evidence to answer that question accurately.",
                "validated": False,
                "warnings": warnings,
            }
        
        # Clean up common artifacts
        cleaned = self._clean_artifacts(cleaned)
        
        return {
            "answer": cleaned,
            "validated": True,
            "warnings": warnings if warnings else None,
        }
    
    def _clean_artifacts(self, text: str) -> str:
        """Remove common LLM output artifacts."""
        # Remove leading "Answer:" or "Response:" prefixes
        text = re.sub(r'^(Answer|Response|Output):\s*', '', text, flags=re.IGNORECASE)
        
        # Remove trailing incomplete sentences (ends with ...)
        if text.endswith('...'):
            # Find the last complete sentence
            sentences = re.split(r'(?<=[.!?])\s+', text[:-3])
            if sentences:
                text = ' '.join(sentences)
        
        return text.strip()
    
    def check_has_citation(self, text: str) -> bool:
        """Check if the response includes meeting/date citations."""
        citation_patterns = [
            r'meeting on [A-Za-z]+ \d+',  # "meeting on Dec 9"
            r'\d{4}-\d{2}-\d{2}',  # ISO date
            r'[A-Za-z]+ \d{1,2},? \d{4}',  # "December 9, 2024"
            r'discussed during',
            r'mentioned in',
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

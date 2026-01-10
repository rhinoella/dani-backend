"""Meeting category inference from metadata.

This module provides runtime inference of meeting categories from existing
metadata fields (title, organizer_email, speakers) without requiring changes
to the ingestion schema.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class CategoryPattern:
    """Pattern definition for category matching."""
    title_patterns: List[str]
    title_exclusions: List[str] = field(default_factory=list)
    email_patterns: List[str] = field(default_factory=list)
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


# Category pattern definitions - used for runtime matching
CATEGORY_PATTERNS: Dict[str, CategoryPattern] = {
    "board": CategoryPattern(
        title_patterns=[
            r"\bboard\b",
            r"\bgovernance\b", 
            r"\bdirectors?\b",
            r"\bexecutive\s+committee\b",
            r"\bec\s+meeting\b",
            r"\bquarterly\s+review\b",
            r"\bannual\s+review\b",
        ],
        title_exclusions=[
            r"\bonboard",
            r"\bdashboard\b",
            r"\bwhiteboard\b",
        ],
    ),
    "1on1": CategoryPattern(
        title_patterns=[
            r"\b1[:\-]?1\b",
            r"\b1[\s\-]?on[\s\-]?1\b",
            r"\bone[\s\-]?on[\s\-]?one\b",
            r"\bcheck[\s\-]?in\s+with\b",
            r"\bcatch[\s\-]?up\s+with\b",
            r"\bsync\s+with\b",
            r"\bweekly\s+with\b",
        ],
        max_speakers=2,  # 1:1s typically have 2 speakers max
    ),
    "standup": CategoryPattern(
        title_patterns=[
            r"\bstand[\s\-]?up\b",
            r"\bdaily\b",
            r"\bscrum\b",
            r"\bsprint\s+planning\b",
            r"\bretrospective\b",
            r"\bretro\b",
            r"\bsprint\s+review\b",
            r"\bgrooming\b",
            r"\brefinement\b",
        ],
    ),
    "client": CategoryPattern(
        title_patterns=[
            r"\bclient\b",
            r"\bcustomer\b",
            r"\bprospect\b",
            r"\bdemo\b",
            r"\bsales\s+call\b",
            r"\bdiscovery\s+call\b",
            r"\bkickoff\b",
            r"\bonboarding\b",
            r"\bimplementation\b",
            r"\bqbr\b",  # Quarterly Business Review
        ],
    ),
    "internal": CategoryPattern(
        title_patterns=[
            r"\bteam\s+meeting\b",
            r"\ball[\s\-]?hands\b",
            r"\btown\s+hall\b",
            r"\bcompany\s+meeting\b",
            r"\bdepartment\b",
            r"\binternal\b",
            r"\beng\s+sync\b",
            r"\bproduct\s+sync\b",
        ],
    ),
}

# Common internal email domains (can be extended via config)
DEFAULT_INTERNAL_DOMAINS = [
    "company.com",
    "corp.com",
    "internal.com",
]


class MeetingCategoryMatcher:
    """Matches meeting metadata to categories at runtime."""
    
    def __init__(
        self,
        patterns: Optional[Dict[str, CategoryPattern]] = None,
        internal_domains: Optional[List[str]] = None,
    ):
        """
        Initialize the category matcher.
        
        Args:
            patterns: Custom pattern definitions (defaults to CATEGORY_PATTERNS)
            internal_domains: List of internal email domains for internal/external detection
        """
        self.patterns = patterns or CATEGORY_PATTERNS
        self.internal_domains = internal_domains or DEFAULT_INTERNAL_DOMAINS
        
        # Compile all regex patterns for performance
        self._compiled_patterns: Dict[str, Tuple[List[re.Pattern], List[re.Pattern]]] = {}
        for category, pattern in self.patterns.items():
            includes = [re.compile(p, re.IGNORECASE) for p in pattern.title_patterns]
            excludes = [re.compile(p, re.IGNORECASE) for p in pattern.title_exclusions]
            self._compiled_patterns[category] = (includes, excludes)
    
    def infer_category(
        self,
        title: Optional[str] = None,
        organizer_email: Optional[str] = None,
        speakers: Optional[List[str]] = None,
        attendee_emails: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Infer meeting category from metadata.
        
        Args:
            title: Meeting title
            organizer_email: Organizer's email address
            speakers: List of speaker names from transcript
            attendee_emails: List of attendee email addresses
            
        Returns:
            Tuple of (category, confidence) where confidence is 0.0-1.0
            Returns (None, 0.0) if no category can be inferred
        """
        if not title:
            return None, 0.0
        
        best_match: Optional[str] = None
        best_confidence = 0.0
        
        # Check each category pattern
        for category, (includes, excludes) in self._compiled_patterns.items():
            # Check exclusions first
            if any(exc.search(title) for exc in excludes):
                continue
            
            # Check title patterns
            for pattern in includes:
                if pattern.search(title):
                    confidence = 0.8  # Base confidence for title match
                    
                    # Boost confidence based on additional criteria
                    pattern_def = self.patterns[category]
                    
                    if speakers:
                        speaker_count = len(speakers)
                        if pattern_def.max_speakers and speaker_count <= pattern_def.max_speakers:
                            confidence += 0.1
                        if pattern_def.min_speakers and speaker_count >= pattern_def.min_speakers:
                            confidence += 0.1
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = category
                    break
        
        # Special handling for internal/external based on email domains
        if not best_match and attendee_emails:
            external_count = sum(
                1 for email in attendee_emails 
                if not any(domain in email.lower() for domain in self.internal_domains)
            )
            
            if external_count > 0:
                best_match = "external"
                best_confidence = 0.6
            else:
                best_match = "internal"
                best_confidence = 0.5
        
        return best_match, min(best_confidence, 1.0)
    
    def matches_category(
        self,
        target_category: str,
        title: Optional[str] = None,
        organizer_email: Optional[str] = None,
        speakers: Optional[List[str]] = None,
        attendee_emails: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ) -> bool:
        """
        Check if meeting metadata matches a target category.
        
        Args:
            target_category: Category to check against
            title: Meeting title
            organizer_email: Organizer's email address
            speakers: List of speaker names
            attendee_emails: List of attendee emails
            confidence_threshold: Minimum confidence for a match
            
        Returns:
            True if the meeting matches the target category
        """
        if target_category == "all":
            return True
        
        inferred, confidence = self.infer_category(
            title=title,
            organizer_email=organizer_email,
            speakers=speakers,
            attendee_emails=attendee_emails,
        )
        
        return inferred == target_category and confidence >= confidence_threshold
    
    def get_title_patterns_for_category(self, category: str) -> List[str]:
        """
        Get the title patterns used for a category.
        
        Useful for building Qdrant text search filters.
        
        Args:
            category: Category name
            
        Returns:
            List of pattern strings (without regex syntax)
        """
        if category not in self.patterns:
            return []
        
        # Extract keywords from patterns (simplified for text search)
        keywords = []
        for pattern in self.patterns[category].title_patterns:
            # Remove regex syntax to get keywords
            keyword = re.sub(r'\\[bs]|\[[\w\-]+\]|\?|\+|\*|\(|\)|\|', '', pattern)
            keyword = keyword.replace('\\', '').strip()
            if keyword:
                keywords.append(keyword)
        
        return keywords


# Module-level singleton for convenience
_default_matcher: Optional[MeetingCategoryMatcher] = None


def get_category_matcher() -> MeetingCategoryMatcher:
    """Get the default category matcher singleton."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = MeetingCategoryMatcher()
    return _default_matcher


def infer_meeting_category(
    title: Optional[str] = None,
    organizer_email: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    **kwargs: Any,
) -> Tuple[Optional[str], float]:
    """
    Convenience function to infer meeting category.
    
    Args:
        title: Meeting title
        organizer_email: Organizer's email
        speakers: List of speakers
        **kwargs: Additional arguments passed to infer_category
        
    Returns:
        Tuple of (category, confidence)
    """
    return get_category_matcher().infer_category(
        title=title,
        organizer_email=organizer_email,
        speakers=speakers,
        **kwargs,
    )


def matches_meeting_category(
    target_category: str,
    title: Optional[str] = None,
    organizer_email: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    **kwargs: Any,
) -> bool:
    """
    Convenience function to check if meeting matches a category.
    
    Args:
        target_category: Category to match
        title: Meeting title
        organizer_email: Organizer's email
        speakers: List of speakers
        **kwargs: Additional arguments
        
    Returns:
        True if the meeting matches the category
    """
    return get_category_matcher().matches_category(
        target_category=target_category,
        title=title,
        organizer_email=organizer_email,
        speakers=speakers,
        **kwargs,
    )

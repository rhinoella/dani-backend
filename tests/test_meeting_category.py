"""Tests for meeting category inference module."""
from __future__ import annotations

import pytest
from app.utils.meeting_category import (
    MeetingCategoryMatcher,
    get_category_matcher,
    infer_meeting_category,
    matches_meeting_category,
    CategoryPattern,
    CATEGORY_PATTERNS,
)


class TestCategoryPatterns:
    """Test category pattern definitions."""
    
    def test_patterns_defined(self):
        """Verify all expected categories have patterns."""
        expected_categories = ["board", "1on1", "standup", "client", "internal"]
        for category in expected_categories:
            assert category in CATEGORY_PATTERNS
    
    def test_pattern_structure(self):
        """Verify pattern structure is correct."""
        for category, pattern in CATEGORY_PATTERNS.items():
            assert isinstance(pattern, CategoryPattern)
            assert isinstance(pattern.title_patterns, list)
            assert len(pattern.title_patterns) > 0


class TestMeetingCategoryMatcher:
    """Test the MeetingCategoryMatcher class."""
    
    @pytest.fixture
    def matcher(self):
        """Create a matcher instance."""
        return MeetingCategoryMatcher()
    
    # Board meeting tests
    @pytest.mark.parametrize("title,expected", [
        ("Board Meeting Q4 2024", "board"),
        ("Monthly Board Review", "board"),
        ("Directors Meeting", "board"),
        ("Executive Committee Meeting", "board"),
        ("EC Meeting - Budget", "board"),
        ("Quarterly Review with Leadership", "board"),
        ("Annual Review Session", "board"),
    ])
    def test_board_meetings(self, matcher, title, expected):
        """Test board meeting detection."""
        category, confidence = matcher.infer_category(title=title)
        assert category == expected
        assert confidence >= 0.5
    
    @pytest.mark.parametrize("title", [
        "Onboarding Session",
        "Dashboard Review",
        "Whiteboard Brainstorm",
    ])
    def test_board_exclusions(self, matcher, title):
        """Test board exclusion patterns."""
        category, _ = matcher.infer_category(title=title)
        assert category != "board"
    
    # 1:1 meeting tests
    @pytest.mark.parametrize("title,expected", [
        ("1:1 with John", "1on1"),
        ("1-1 Weekly", "1on1"),
        ("1 on 1 - Career Discussion", "1on1"),
        ("One-on-One with Manager", "1on1"),
        ("Check-in with Sarah", "1on1"),
        ("Catch-up with Team Lead", "1on1"),
        ("Weekly sync with Mike", "1on1"),
    ])
    def test_1on1_meetings(self, matcher, title, expected):
        """Test 1:1 meeting detection."""
        category, confidence = matcher.infer_category(title=title)
        assert category == expected
        assert confidence >= 0.5
    
    def test_1on1_speaker_boost(self, matcher):
        """Test that 1:1s with 2 speakers get confidence boost."""
        _, confidence_no_speakers = matcher.infer_category(title="1:1 with John")
        _, confidence_with_speakers = matcher.infer_category(
            title="1:1 with John",
            speakers=["Alice", "John"],
        )
        assert confidence_with_speakers > confidence_no_speakers
    
    # Standup/Scrum tests
    @pytest.mark.parametrize("title,expected", [
        ("Daily Standup", "standup"),
        ("Stand-up Meeting", "standup"),
        ("Morning Scrum", "standup"),
        ("Sprint Planning", "standup"),
        ("Team Retrospective", "standup"),
        ("Sprint Retro", "standup"),
        ("Sprint Review", "standup"),
        ("Backlog Grooming", "standup"),
        ("Refinement Session", "standup"),
    ])
    def test_standup_meetings(self, matcher, title, expected):
        """Test standup/scrum meeting detection."""
        category, confidence = matcher.infer_category(title=title)
        assert category == expected
        assert confidence >= 0.5
    
    # Client meeting tests
    @pytest.mark.parametrize("title,expected", [
        ("Client Call - Acme Corp", "client"),
        ("Customer Success Review", "client"),
        ("Product Demo", "client"),
        ("Sales Call with Prospect", "client"),
        ("Discovery Call - New Lead", "client"),
        ("Project Kickoff - Client X", "client"),
        ("Onboarding Call", "client"),
        ("Implementation Review", "client"),
        ("QBR with Customer", "client"),
    ])
    def test_client_meetings(self, matcher, title, expected):
        """Test client meeting detection."""
        category, confidence = matcher.infer_category(title=title)
        assert category == expected
        assert confidence >= 0.5
    
    # Internal meeting tests
    @pytest.mark.parametrize("title,expected", [
        ("Team Meeting", "internal"),
        ("All-Hands Meeting", "internal"),
        ("Town Hall", "internal"),
        ("Company Meeting", "internal"),
        ("Department Sync", "internal"),
        ("Internal Review", "internal"),
        ("Eng Sync", "internal"),
        ("Product Sync", "internal"),
    ])
    def test_internal_meetings(self, matcher, title, expected):
        """Test internal meeting detection."""
        category, confidence = matcher.infer_category(title=title)
        assert category == expected
        assert confidence >= 0.5
    
    # No match tests
    def test_no_match_returns_none(self, matcher):
        """Test that ambiguous titles return None."""
        category, confidence = matcher.infer_category(title="Random Meeting Title")
        # Either None or very low confidence
        assert category is None or confidence < 0.5
    
    def test_empty_title_returns_none(self, matcher):
        """Test that empty title returns None."""
        category, confidence = matcher.infer_category(title="")
        assert category is None
        assert confidence == 0.0
    
    def test_none_title_returns_none(self, matcher):
        """Test that None title returns None."""
        category, confidence = matcher.infer_category(title=None)
        assert category is None
        assert confidence == 0.0


class TestMatchesCategory:
    """Test the matches_category method."""
    
    @pytest.fixture
    def matcher(self):
        return MeetingCategoryMatcher()
    
    def test_matches_all_returns_true(self, matcher):
        """Test that 'all' category always matches."""
        assert matcher.matches_category("all", title="Any Meeting")
        assert matcher.matches_category("all", title="Board Meeting")
        assert matcher.matches_category("all", title=None)
    
    def test_matches_correct_category(self, matcher):
        """Test matching correct category."""
        assert matcher.matches_category("board", title="Board Meeting")
        assert matcher.matches_category("1on1", title="1:1 with John")
        assert matcher.matches_category("standup", title="Daily Standup")
    
    def test_does_not_match_wrong_category(self, matcher):
        """Test non-matching categories."""
        assert not matcher.matches_category("board", title="Daily Standup")
        assert not matcher.matches_category("1on1", title="Board Meeting")
        assert not matcher.matches_category("standup", title="Client Call")


class TestGetTitlePatterns:
    """Test the get_title_patterns_for_category method."""
    
    @pytest.fixture
    def matcher(self):
        return MeetingCategoryMatcher()
    
    def test_returns_keywords_for_valid_category(self, matcher):
        """Test that keywords are returned for valid categories."""
        keywords = matcher.get_title_patterns_for_category("board")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "board" in keywords
    
    def test_returns_empty_for_invalid_category(self, matcher):
        """Test that empty list is returned for invalid category."""
        keywords = matcher.get_title_patterns_for_category("nonexistent")
        assert keywords == []


class TestModuleFunctions:
    """Test module-level convenience functions."""
    
    def test_get_category_matcher_singleton(self):
        """Test that get_category_matcher returns singleton."""
        matcher1 = get_category_matcher()
        matcher2 = get_category_matcher()
        assert matcher1 is matcher2
    
    def test_infer_meeting_category_function(self):
        """Test the convenience function."""
        category, confidence = infer_meeting_category(title="Board Meeting")
        assert category == "board"
        assert confidence >= 0.5
    
    def test_matches_meeting_category_function(self):
        """Test the convenience function."""
        assert matches_meeting_category("board", title="Board Meeting")
        assert not matches_meeting_category("standup", title="Board Meeting")
        assert matches_meeting_category("all", title="Any Title")


class TestCaseInsensitivity:
    """Test that matching is case-insensitive."""
    
    @pytest.fixture
    def matcher(self):
        return MeetingCategoryMatcher()
    
    @pytest.mark.parametrize("title", [
        "BOARD MEETING",
        "board meeting",
        "Board Meeting",
        "BoArD mEeTiNg",
    ])
    def test_board_case_insensitive(self, matcher, title):
        """Test board matching is case insensitive."""
        category, _ = matcher.infer_category(title=title)
        assert category == "board"
    
    @pytest.mark.parametrize("title", [
        "DAILY STANDUP",
        "daily standup",
        "Daily Standup",
    ])
    def test_standup_case_insensitive(self, matcher, title):
        """Test standup matching is case insensitive."""
        category, _ = matcher.infer_category(title=title)
        assert category == "standup"


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    @pytest.fixture
    def matcher(self):
        return MeetingCategoryMatcher()
    
    def test_multiple_category_keywords(self, matcher):
        """Test title with keywords from multiple categories."""
        # "Board" keyword + "1:1" keyword - should pick one
        category, _ = matcher.infer_category(title="Board 1:1 Meeting")
        assert category in ["board", "1on1"]
    
    def test_very_long_title(self, matcher):
        """Test handling of very long titles."""
        long_title = "This is a very long meeting title " * 10 + "Board Meeting"
        category, _ = matcher.infer_category(title=long_title)
        assert category == "board"
    
    def test_title_with_special_characters(self, matcher):
        """Test titles with special characters."""
        category, _ = matcher.infer_category(title="[Board] Meeting @Q4 #2024")
        assert category == "board"
    
    def test_confidence_capped_at_1(self, matcher):
        """Test that confidence is capped at 1.0."""
        _, confidence = matcher.infer_category(
            title="1:1 with John",
            speakers=["Alice", "John"],  # Boost from speakers
        )
        assert confidence <= 1.0


class TestCustomPatterns:
    """Test using custom patterns."""
    
    def test_custom_patterns(self):
        """Test matcher with custom patterns."""
        custom_patterns = {
            "strategy": CategoryPattern(
                title_patterns=[r"\bstrategy\b", r"\bplanning\b"],
            ),
        }
        matcher = MeetingCategoryMatcher(patterns=custom_patterns)
        
        category, _ = matcher.infer_category(title="Strategy Session")
        assert category == "strategy"
        
        # Should not match board (not in custom patterns)
        category, _ = matcher.infer_category(title="Board Meeting")
        assert category is None
    
    def test_custom_internal_domains(self):
        """Test matcher with custom internal domains."""
        custom_domains = ["mycompany.com", "mycompany.org"]
        matcher = MeetingCategoryMatcher(internal_domains=custom_domains)
        
        # Test that external detection uses custom domains
        category, _ = matcher.infer_category(
            title="Generic Meeting",
            attendee_emails=["user@external.com"],
        )
        assert category == "external"
        
        category, _ = matcher.infer_category(
            title="Generic Meeting",
            attendee_emails=["user@mycompany.com"],
        )
        assert category == "internal"

"""
Tests for Output Validator.
"""

import pytest
from app.llm.output_validator import OutputValidator, UNCERTAINTY_PHRASES, HALLUCINATION_INDICATORS


# ============== Tests ==============

class TestOutputValidator:
    """Tests for OutputValidator."""
    
    def test_init_default(self):
        """Test default initialization."""
        validator = OutputValidator()
        assert validator.strict_mode is False
    
    def test_init_strict_mode(self):
        """Test initialization with strict mode."""
        validator = OutputValidator(strict_mode=True)
        assert validator.strict_mode is True
    
    def test_validate_normal_text(self):
        """Test validation of normal text."""
        validator = OutputValidator()
        result = validator.validate("The meeting discussed the Q4 budget.")
        
        assert result["validated"] is True
        assert result["answer"] == "The meeting discussed the Q4 budget."
        assert result["warnings"] is None
    
    def test_validate_empty_text(self):
        """Test validation of empty text."""
        validator = OutputValidator()
        result = validator.validate("")
        
        assert result["validated"] is True
        assert result["answer"] == "I don't have a record of that discussion."
        assert "empty_response" in result["warnings"]
    
    def test_validate_whitespace_only(self):
        """Test validation of whitespace-only text."""
        validator = OutputValidator()
        result = validator.validate("   \n\t  ")
        
        assert result["validated"] is True
        assert result["answer"] == "I don't have a record of that discussion."
    
    def test_validate_detects_uncertainty_phrases(self):
        """Test detection of uncertainty phrases."""
        validator = OutputValidator()
        
        for phrase in UNCERTAINTY_PHRASES[:3]:  # Test first few
            result = validator.validate(f"Well, {phrase} that might be correct.")
            
            assert any("uncertainty_detected" in w for w in result["warnings"])
    
    def test_validate_detects_i_think(self):
        """Test detection of 'I think'."""
        validator = OutputValidator()
        result = validator.validate("I think the meeting was about budgets.")
        
        assert any("uncertainty_detected: i think" in w for w in result["warnings"])
    
    def test_validate_detects_i_believe(self):
        """Test detection of 'I believe'."""
        validator = OutputValidator()
        result = validator.validate("I believe this is correct.")
        
        assert any("uncertainty_detected: i believe" in w for w in result["warnings"])
    
    def test_validate_detects_probably(self):
        """Test detection of 'probably'."""
        validator = OutputValidator()
        result = validator.validate("The answer is probably yes.")
        
        assert any("uncertainty_detected: probably" in w for w in result["warnings"])
    
    def test_validate_detects_hallucination_indicators(self):
        """Test detection of hallucination indicators."""
        validator = OutputValidator()
        
        for phrase in HALLUCINATION_INDICATORS[:3]:  # Test first few
            result = validator.validate(f"Well, {phrase}, this is true.")
            
            assert any("hallucination_risk" in w for w in result["warnings"])
    
    def test_validate_strict_mode_rejects_hallucination(self):
        """Test strict mode rejects hallucination indicators."""
        validator = OutputValidator(strict_mode=True)
        result = validator.validate("Based on my training, the answer is yes.")
        
        assert result["validated"] is False
        assert "sufficient evidence" in result["answer"].lower()
    
    def test_validate_non_strict_allows_hallucination(self):
        """Test non-strict mode allows hallucination indicators."""
        validator = OutputValidator(strict_mode=False)
        result = validator.validate("Based on my training, the answer is yes.")
        
        assert result["validated"] is True
        assert any("hallucination_risk" in w for w in result["warnings"])
    
    def test_clean_artifacts_answer_prefix(self):
        """Test cleaning 'Answer:' prefix."""
        validator = OutputValidator()
        result = validator.validate("Answer: The meeting was productive.")
        
        assert result["answer"] == "The meeting was productive."
    
    def test_clean_artifacts_response_prefix(self):
        """Test cleaning 'Response:' prefix."""
        validator = OutputValidator()
        result = validator.validate("Response: The budget was approved.")
        
        assert result["answer"] == "The budget was approved."
    
    def test_clean_artifacts_output_prefix(self):
        """Test cleaning 'Output:' prefix."""
        validator = OutputValidator()
        result = validator.validate("Output: Here is the information.")
        
        assert result["answer"] == "Here is the information."
    
    def test_clean_artifacts_case_insensitive(self):
        """Test prefix cleaning is case insensitive."""
        validator = OutputValidator()
        result = validator.validate("ANSWER: The meeting notes.")
        
        assert result["answer"] == "The meeting notes."
    
    def test_clean_artifacts_trailing_ellipsis(self):
        """Test handling of trailing ellipsis."""
        validator = OutputValidator()
        result = validator.validate("The meeting discussed budgets. More details...")
        
        # Should remove incomplete sentence
        assert "..." not in result["answer"] or result["answer"].endswith(".")
    
    def test_clean_artifacts_preserves_complete_text(self):
        """Test that complete text is preserved."""
        validator = OutputValidator()
        text = "The Q4 meeting discussed revenue targets. The team agreed on $10M."
        result = validator.validate(text)
        
        assert result["answer"] == text
    
    def test_check_has_citation_meeting_date(self):
        """Test citation detection with meeting date."""
        validator = OutputValidator()
        
        assert validator.check_has_citation("As discussed in the meeting on Dec 9...") is True
        assert validator.check_has_citation("The meeting on January 15 covered...") is True
    
    def test_check_has_citation_iso_date(self):
        """Test citation detection with ISO date."""
        validator = OutputValidator()
        
        assert validator.check_has_citation("On 2024-12-15, the team...") is True
        assert validator.check_has_citation("Reference: 2023-01-20") is True
    
    def test_check_has_citation_full_date(self):
        """Test citation detection with full date format."""
        validator = OutputValidator()
        
        assert validator.check_has_citation("December 9, 2024 meeting notes") is True
        assert validator.check_has_citation("January 15 2024 discussion") is True
    
    def test_check_has_citation_discussed_during(self):
        """Test citation detection with 'discussed during'."""
        validator = OutputValidator()
        
        assert validator.check_has_citation("This was discussed during the board meeting") is True
    
    def test_check_has_citation_mentioned_in(self):
        """Test citation detection with 'mentioned in'."""
        validator = OutputValidator()
        
        assert validator.check_has_citation("As mentioned in the transcript") is True
    
    def test_check_has_citation_no_citation(self):
        """Test citation detection with no citation."""
        validator = OutputValidator()
        
        assert validator.check_has_citation("The budget is $10 million.") is False
        assert validator.check_has_citation("Yes, that is correct.") is False
    
    def test_multiple_warnings(self):
        """Test text with multiple warnings."""
        validator = OutputValidator()
        result = validator.validate("I think, based on my training, this might be correct.")
        
        warnings = result["warnings"]
        assert len(warnings) >= 2
        assert any("uncertainty_detected" in w for w in warnings)
        assert any("hallucination_risk" in w for w in warnings)
    
    def test_as_an_ai_detected(self):
        """Test detection of 'as an ai'."""
        validator = OutputValidator()
        result = validator.validate("As an AI, I cannot access real-time data.")
        
        assert any("uncertainty_detected: as an ai" in w for w in result["warnings"])
    
    def test_as_language_model_detected(self):
        """Test detection of 'as a language model'."""
        validator = OutputValidator()
        result = validator.validate("As a language model, I don't have opinions.")
        
        assert any("uncertainty_detected: as a language model" in w for w in result["warnings"])
    
    def test_i_cannot_detected(self):
        """Test detection of 'I cannot'."""
        validator = OutputValidator()
        result = validator.validate("I cannot provide that information.")
        
        assert any("uncertainty_detected: i cannot" in w for w in result["warnings"])
    
    def test_typically_detected(self):
        """Test detection of 'typically'."""
        validator = OutputValidator()
        result = validator.validate("Typically, meetings are held weekly.")
        
        assert any("hallucination_risk: typically" in w for w in result["warnings"])
    
    def test_generally_speaking_detected(self):
        """Test detection of 'generally speaking'."""
        validator = OutputValidator()
        result = validator.validate("Generally speaking, this approach works.")
        
        assert any("hallucination_risk: generally speaking" in w for w in result["warnings"])


class TestUncertaintyPhrases:
    """Tests for uncertainty phrase constants."""
    
    def test_uncertainty_phrases_exist(self):
        """Test that uncertainty phrases are defined."""
        assert len(UNCERTAINTY_PHRASES) > 0
    
    def test_uncertainty_phrases_lowercase(self):
        """Test that phrases are lowercase."""
        for phrase in UNCERTAINTY_PHRASES:
            assert phrase == phrase.lower()


class TestHallucinationIndicators:
    """Tests for hallucination indicator constants."""
    
    def test_hallucination_indicators_exist(self):
        """Test that indicators are defined."""
        assert len(HALLUCINATION_INDICATORS) > 0
    
    def test_hallucination_indicators_lowercase(self):
        """Test that indicators are lowercase."""
        for indicator in HALLUCINATION_INDICATORS:
            assert indicator == indicator.lower()

"""Tests for the pipeline module."""

import unittest
from pipelines.pipeline import parse_trait_response


class TestPipeline(unittest.TestCase):
    """Tests for the pipeline module."""
    
    def test_parse_trait_response_clear_format(self):
        """Test parsing a response with a clear score format."""
        # Test a response where the score is at the beginning
        response = "4: The text demonstrates high openness with curiosity about new ideas."
        score, rationale = parse_trait_response(response)
        self.assertEqual(score, 4)
        self.assertEqual(rationale, "The text demonstrates high openness with curiosity about new ideas.")
        
        # Test a response where the score is after a newline
        response = "Based on the text,\n5. The person shows very high conscientiousness."
        score, rationale = parse_trait_response(response)
        self.assertEqual(score, 5)
        self.assertEqual(rationale, "The person shows very high conscientiousness.")
    
    def test_parse_trait_response_ambiguous_format(self):
        """Test parsing responses with less clear formats."""
        # Test when the score is in the middle of a sentence
        response = "I would rate this as a 2 on the scale because it lacks extraversion."
        score, rationale = parse_trait_response(response)
        self.assertEqual(score, 2)
        
        # Test when there are multiple numbers but we take the first 1-5
        response = "In 2023, I would give this a 3 out of 5 for agreeableness."
        score, rationale = parse_trait_response(response)
        self.assertEqual(score, 3)
    
    def test_parse_trait_response_no_score(self):
        """Test parsing a response with no clear score."""
        response = "The text shows moderate levels of neuroticism."
        score, rationale = parse_trait_response(response)
        # Should return the default score (3)
        self.assertEqual(score, 3)
        self.assertEqual(rationale, "Could not parse a score from the response.")


if __name__ == "__main__":
    unittest.main() 
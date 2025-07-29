"""Tests for the trait_prompter module."""

import unittest
from pipelines.trait_prompter import make_prompt, TRAIT_DEFINITIONS


class TestTraitPrompter(unittest.TestCase):
    """Tests for the trait_prompter module."""
    
    def test_make_prompt_valid_trait(self):
        """Test that make_prompt works with valid traits."""
        text = "Sample text for testing."
        
        for trait in TRAIT_DEFINITIONS:
            prompt = make_prompt(trait, text)
            
            # Check that the prompt contains the trait name
            self.assertIn(trait, prompt)
            
            # Check that the prompt contains the trait definition
            self.assertIn(TRAIT_DEFINITIONS[trait], prompt)
            
            # Check that the prompt contains the text to analyze
            self.assertIn(text, prompt)
            
            # Check that the prompt asks for a rating on a 1-5 scale
            self.assertIn("1", prompt)
            self.assertIn("5", prompt)
    
    def test_make_prompt_invalid_trait(self):
        """Test that make_prompt raises ValueError for invalid traits."""
        text = "Sample text for testing."
        invalid_trait = "InvalidTrait"
        
        with self.assertRaises(ValueError):
            make_prompt(invalid_trait, text)


if __name__ == "__main__":
    unittest.main() 
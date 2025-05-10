"""Module for generating prompts to evaluate Big Five personality traits."""

TRAIT_DEFINITIONS = {
    "Conscientiousness": (
        "The tendency to be organized, responsible, and hardworking. "
        "High conscientiousness indicates careful, diligent, and methodical behavior."
    ),
    "Agreeableness": (
        "The tendency to be compassionate, cooperative, and considerate. "
        "High agreeableness indicates warmth, empathy, and a desire for social harmony."
    ),
    "Neuroticism": (
        "The tendency to experience negative emotions like anxiety, depression, and anger. "
        "High neuroticism indicates emotional instability and frequent mood changes."
    ),
    "Openness": (
        "The tendency to be creative, curious, and open to new experiences. "
        "High openness indicates imagination, intellectual curiosity, and appreciation for art and beauty."
    ),
    "Extraversion": (
        "The tendency to be outgoing, energetic, and sociable. "
        "High extraversion indicates assertiveness, talkativeness, and a preference for social interaction."
    )
}

def make_prompt(trait: str, text: str) -> str:
    """Generate a prompt to evaluate a specific Big Five personality trait.
    
    Args:
        trait: One of the Big Five personality traits
        text: The text to evaluate for the trait
        
    Returns:
        A formatted prompt string
    """
    if trait not in TRAIT_DEFINITIONS:
        raise ValueError(f"Unknown trait: {trait}. Must be one of {list(TRAIT_DEFINITIONS.keys())}")
    
    trait_definition = TRAIT_DEFINITIONS[trait]
    
    prompt = f"""Analyze the following text for its level of {trait}.

Definition of {trait}: {trait_definition}

Text to analyze:
\"{text}\"

On a scale from 1 (very low) to 5 (very high), rate the level of {trait} expressed in the text. 
Provide your rating as a single digit (1-5) followed by a brief explanation in one sentence.
"""
    
    return prompt 
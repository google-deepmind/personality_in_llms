"""Main pipeline module for evaluating Big Five personality traits across multiple LLMs."""

import re
import os
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from pipelines.trait_prompter import make_prompt
from pipelines.llm_client import get_llm_client, LLMClient


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration as a dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_trait_response(response: str) -> Tuple[int, str]:
    """Extract the numeric score and rationale from an LLM response.
    
    Args:
        response: The LLM's response to a trait prompt
        
    Returns:
        A tuple of (score, rationale) where score is an integer 1-5
        and rationale is the explanation text
    """
    # Try to find a digit between 1 and 5 at the start of the response or after newlines
    match = re.search(r'(?:^|\n)\s*([1-5])\s*[.:]?\s*', response)
    
    if match:
        score = int(match.group(1))
        # Everything after the score is the rationale
        rationale = response[match.end():].strip()
        return score, rationale
    
    # Fallback: look for any digit 1-5 in the response
    match = re.search(r'([1-5])', response)
    if match:
        score = int(match.group(1))
        return score, "No clear rationale provided."
    
    # If no score found, return default
    return 3, "Could not parse a score from the response."


class BigFiveTraitPipeline:
    """Pipeline for evaluating Big Five personality traits across multiple LLMs."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with a configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        self.llm_clients: Dict[str, LLMClient] = {}
        
        # Initialize LLM clients
        for llm_config in self.config.get('llms', []):
            client = get_llm_client(llm_config)
            if client:
                self.llm_clients[llm_config['name']] = client
    
    def run(self, input_text: Optional[str] = None) -> pd.DataFrame:
        """Run the pipeline on the provided text.
        
        Args:
            input_text: Text to evaluate. If None, uses the text from config.
            
        Returns:
            DataFrame with trait scores for each LLM
        """
        if not self.llm_clients:
            raise RuntimeError("No LLM clients were initialized successfully")
        
        # Use provided text or fall back to config
        text = input_text or self.config.get('input_text', '')
        if not text:
            raise ValueError("No input text provided and none in config")
        
        # Get traits from config
        traits = self.config.get('traits', [])
        if not traits:
            raise ValueError("No traits defined in configuration")
        
        results = []
        
        # For each LLM and trait, get a score
        for llm_name, llm_client in self.llm_clients.items():
            llm_results = {'llm': llm_name}
            
            for trait in traits:
                # Generate the prompt for this trait
                prompt = make_prompt(trait, text)
                
                # Get the LLM's response
                response = llm_client.invoke(prompt)
                
                # Parse the response to get score and rationale
                score, rationale = parse_trait_response(response)
                
                # Store results
                llm_results[f"{trait}_score"] = score
                llm_results[f"{trait}_rationale"] = rationale
            
            results.append(llm_results)
        
        # Convert to DataFrame
        return pd.DataFrame(results)
    
    def save_results(self, results: pd.DataFrame, output_path: str) -> None:
        """Save the results to a CSV file.
        
        Args:
            results: DataFrame of results
            output_path: Path to save the results
        """
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def run_pipeline(config_path: str, 
                 input_text: Optional[str] = None, 
                 output_path: Optional[str] = None) -> pd.DataFrame:
    """Run the Big Five trait pipeline with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        input_text: Optional text to evaluate
        output_path: Optional path to save results
        
    Returns:
        DataFrame with trait scores for each LLM
    """
    pipeline = BigFiveTraitPipeline(config_path)
    results = pipeline.run(input_text)
    
    if output_path:
        pipeline.save_results(results, output_path)
    
    return results 
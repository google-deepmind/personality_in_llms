"""Module for interacting with different LLM providers."""

import abc
import os
from typing import Dict, Any, Optional


class LLMClient(abc.ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the LLM client.
        
        Args:
            model_config: Configuration for the LLM
        """
        self.model_config = model_config
        self.name = model_config.get('name', 'unnamed_model')
        self.model = model_config.get('model', '')
        
        # Get API key from environment
        api_key_env = model_config.get('api_key_env', '')
        self.api_key = os.environ.get(api_key_env, '')
        
        if not self.api_key and api_key_env:
            print(f"Warning: Environment variable {api_key_env} not set")
    
    @abc.abstractmethod
    def invoke(self, prompt: str) -> str:
        """Send a prompt to the LLM and get the response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI's LLM API."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the OpenAI client."""
        super().__init__(model_config)
        
        # Lazy import to avoid dependencies if not used
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            print("OpenAI package not installed. Run 'pip install openai'")
            self.client = None
    
    def invoke(self, prompt: str) -> str:
        """Send a prompt to OpenAI and get the response."""
        if not self.client:
            return "Error: OpenAI client not initialized"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Use deterministic responses for consistent trait scoring
                max_tokens=100    # Limit response length
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"


class HuggingFaceClient(LLMClient):
    """Client for HuggingFace's models."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the HuggingFace client."""
        super().__init__(model_config)
        self.hf_client = None
        
        # Try to initialize the client
        try:
            from huggingface_hub import InferenceClient
            self.hf_client = InferenceClient(token=self.api_key)
        except ImportError:
            print("HuggingFace Hub package not installed. Run 'pip install huggingface_hub'")
    
    def invoke(self, prompt: str) -> str:
        """Send a prompt to HuggingFace and get the response."""
        if not self.hf_client:
            return "Error: HuggingFace client not initialized"
        
        try:
            response = self.hf_client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=100,
                temperature=0.0
            )
            return response
        except Exception as e:
            return f"Error querying HuggingFace: {str(e)}"


def get_llm_client(model_config: Dict[str, Any]) -> Optional[LLMClient]:
    """Factory function to get the appropriate LLM client.
    
    Args:
        model_config: Configuration for the LLM
    
    Returns:
        An LLM client instance or None if provider is not supported
    """
    provider = model_config.get('provider', '').lower()
    
    if provider == 'openai':
        return OpenAIClient(model_config)
    elif provider == 'huggingface':
        return HuggingFaceClient(model_config)
    else:
        print(f"Unsupported provider: {provider}")
        return None 
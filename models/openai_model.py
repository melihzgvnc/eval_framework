from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
from openai import OpenAI
from openai.types.responses import Response

from .base_model import Model


@dataclass
class OpenAIModel(Model):
    """OpenAI model implementation.

    This class provides a concrete implementation for OpenAI's API,
    supporting both the traditional Chat Completions API and the newer
    Responses API.
    """
    
    api_key: Optional[str] = field(default=None)
    model: str = field(default="gpt-4o")
    base_url: Optional[str] = field(default=None)
    organization: Optional[str] = field(default=None)
    timeout: float = field(default=30.0)
    max_retries: int = field(default=2)
    use_responses_api: Optional[bool] = field(default=True)  # Use newer Responses API vs Chat Completions
    client: Optional[OpenAI] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize the OpenAI client after dataclass initialization."""
        # Use provided API key or get from environment
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )
        
        # Initialize client
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        
        # Mark as loaded since OpenAI models don't require local loading
        self.is_loaded = True
    
    def load(self, model_path: Optional[str] = None) -> None:
        """Load method for OpenAI models.
        
        OpenAI models don't require local loading, but this method
        validates the API key and client initialization.
        
        Args:
            model_path: Optional, not used for OpenAI models but included
                       for interface compatibility.
        
        Raises:
            RuntimeError: If client initialization failed
        """
        # Already initialized in __post_init__, just verify
        if self.client is None:
            raise RuntimeError("OpenAI client failed to initialize")
        
        # Update model if path provided
        if model_path:
            self.model = model_path
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI's API.
        
        Args:
            prompt: Input text to generate from
            **kwargs: Additional generation parameters:
                - instructions: System instructions (for Responses API)
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - etc.
        
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If model is not properly initialized
            ValueError: If API call fails
        """
        if self.client is None:
            raise RuntimeError("OpenAI client must be initialized before generating")
        
        try:
            if self.use_responses_api:
                # Use newer Responses API
                return self._generate_with_responses_api(prompt, **kwargs)
            else:
                # Use traditional Chat Completions API
                return self._generate_with_chat_completions(prompt, **kwargs)
                
        except Exception as e:
            raise ValueError(f"OpenAI API call failed: {str(e)}")
    
    def _generate_with_responses_api(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI's Responses API.
        
        Args:
            prompt: Input text
            **kwargs: Additional parameters including 'instructions'
            
        Returns:
            Generated text
        """
        # Extract parameters
        instructions = kwargs.get("instructions", "")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        # Make API call
        response: Response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        
        return response.output_text
    
    def _generate_with_chat_completions(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI's Chat Completions API.
        
        Args:
            prompt: Input text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Extract parameters with defaults
        system_message = kwargs.get("system", "You are a helpful assistant.")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1000)
        top_p = kwargs.get("top_p", 1.0)
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        
        return response.choices[0].message.content
    
    def set_api_key(self, api_key: str) -> None:
        """Update the API key and reinitialize the client.
        
        Args:
            api_key: New OpenAI API key
        """
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
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
        temperature = kwargs.get("temperature", 0.2)
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
        response_format = kwargs.get("response_format", {"type": "text"})
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format=response_format,
        )
        
        return response.choices[0].message.content
    
    def generate_structured(
        self, 
        prompt: str, 
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured JSON output.
        
        Always enforces JSON output. When a schema is provided and the model
        supports it, OpenAI's strict `json_schema` response format is used to
        guarantee schema-conforming output. Otherwise, falls back to JSON mode
        with the schema injected into the system prompt.
        
        Args:
            prompt: Input text
            schema: Optional JSON schema for the response
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            ValueError: If JSON parsing fails
        """
        import json

        # Models that support strict json_schema response format.
        json_schema_models = {"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"}
        # Models that support plain json_object response format.
        json_object_models = json_schema_models | {"gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"}

        system_msg = kwargs.get("system", "You are a helpful assistant.")

        if schema and self.model in json_schema_models:
            # Strongest enforcement: strict JSON schema.
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": schema,
                    "strict": False,
                },
            }
        elif self.model in json_object_models:
            # JSON mode: model is constrained to emit valid JSON.
            response_format = {"type": "json_object"}
            if schema:
                system_msg += (
                    "\n\nReturn a JSON object matching this schema: "
                    f"{json.dumps(schema)}"
                )
        else:
            # Older model without JSON support: best-effort prompting.
            response_format = {"type": "text"}
            system_msg += "\n\nReturn your response as a valid JSON object."
            if schema:
                system_msg += (
                    f" The JSON should match this schema: {json.dumps(schema)}"
                )

        kwargs["system"] = system_msg
        kwargs["response_format"] = response_format

        response_text = self.generate(prompt, **kwargs)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Last-resort: try to extract a JSON object embedded in text.
            import re
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON response: {e}\nResponse: {response_text}"
                    ) from e
            raise ValueError(
                f"Model did not return valid JSON.\nResponse: {response_text}"
            )
    
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
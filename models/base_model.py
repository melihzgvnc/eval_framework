from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from eval_framework.core.base import BaseModel


@dataclass
class Model(BaseModel):
    """Base implementation for all concrete models.
    
    This class provides a foundation for model implementations with
    common attributes and basic implementations.
    """
    
    model_name: str
    model_path: Optional[str] = None
    is_loaded: bool = field(default=False, init=False)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt.
        
        This is a placeholder implementation that should be overridden
        by concrete model classes.
        
        Args:
            prompt: Input text to generate from
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "generate() must be implemented by concrete model classes"
        )
    
    def load(self, model_path: str) -> None:
        """Load the model from the specified path.
        
        This is a placeholder implementation that should be overridden
        by concrete model classes.
        
        Args:
            model_path: Path to load the model from
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            "load() must be implemented by concrete model classes"
        )


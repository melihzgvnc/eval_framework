from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

from .types import Dataset, EvaluationResult, PipelineResult

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: Input text to generate from
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the model from the specified path.
        
        Args:
            model_path: Path to load the model from
        """
        pass


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    @abstractmethod
    def evaluate(self, sample: str, answer: str) -> EvaluationResult:
        """Evaluate a sample against an answer.
        
        Args:
            sample: Input sample/context
            answer: Model-generated answer to evaluate
            
        Returns:
            EvaluationResult with score and details
        """
        pass


class BaseDataset(ABC):
    """Abstract base class for all datasets."""
    
    @abstractmethod
    def load(self, data_path: str) -> Dataset:
        """Load dataset from the specified path.
        
        Args:
            data_path: Path to dataset file or directory
            
        Returns:
            Loaded Dataset object
        """
        pass


class BasePipeline(ABC):
    """Abstract base class for all pipelines."""
    
    @abstractmethod
    def invoke(self, modules: List[Any]) -> PipelineResult:
        """Invoke the pipeline with the given modules.
        
        Args:
            modules: List of modules to execute in the pipeline
            
        Returns:
            PipelineResult with all evaluation results
        """
        pass
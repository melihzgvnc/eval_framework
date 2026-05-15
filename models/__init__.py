"""Model implementations for the evaluation framework.

This package contains concrete implementations of the BaseModel interface
for different model types and frameworks.
"""

from .base_model import Model

# Import other models lazily to avoid dependency issues
__all__ = [
    "Model",
    "HuggingFaceModel",
    "OpenAIModel",
]

# Lazy imports to avoid requiring all dependencies at once
def __getattr__(name):
    if name == "HuggingFaceModel":
        from .huggingface import HuggingFaceModel
        return HuggingFaceModel
    elif name == "OpenAIModel":
        from .openai_model import OpenAIModel
        return OpenAIModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
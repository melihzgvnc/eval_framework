"""Data types for the dataset layer.

Defines the standardized sample representation and dataset metadata
that all loaders produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalSample:
    """A single evaluation sample with pre-generated model output.

    This is the framework's canonical representation of one item to
    evaluate. Loaders translate raw file rows into this shape.
    """

    sample_id: str
    source_text: str          # Ground truth / context / reference
    model_response: str       # Pre-generated model output to evaluate
    query: Optional[str] = None  # Original prompt / question (if available)
    should_refuse: bool = False  # Whether the model should have refused
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_text": self.source_text,
            "model_response": self.model_response,
            "query": self.query,
            "should_refuse": self.should_refuse,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSample":
        """Reconstruct from a dictionary (e.g., cached or serialized)."""
        return cls(
            sample_id=str(data.get("sample_id", "")),
            source_text=str(data.get("source_text", "")),
            model_response=str(data.get("model_response", "")),
            query=data.get("query"),
            should_refuse=bool(data.get("should_refuse", False)),
            metadata=data.get("metadata") or {},
        )


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""

    name: str
    path: str
    format: str  # "jsonl", "json", "csv", "tsv"
    num_samples: int
    columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "format": self.format,
            "num_samples": self.num_samples,
            "columns": self.columns,
            "metadata": self.metadata,
        }

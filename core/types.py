from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Result of an evaluation operation."""
    score: float
    passed: bool
    details: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Dataset:
    """Container for dataset information."""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    results: List[EvaluationResult]
    summary: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

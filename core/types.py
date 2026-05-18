from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class EvaluationType(Enum):
    """Types of evaluation results."""
    NLI = "nli"
    LLM_JUDGE = "llm_judge"
    COMPOSITE = "composite"
    RULE_BASED = "rule_based"


@dataclass
class EvaluationResult:
    """Base result of an evaluation operation.
    
    This is a flexible container that can accommodate different types
    of evaluation results (NLI, LLM Judge, composite, etc.).
    """
    
    # Core fields
    score: float  # Primary score (0-1 or 0-100 scale)
    passed: bool  # Binary pass/fail based on threshold
    
    # Flexible details - can contain anything evaluator-specific
    details: Dict[str, Any]
    
    # Metadata
    evaluation_type: EvaluationType = EvaluationType.NLI
    metadata: Optional[Dict[str, Any]] = None
    
    # LLM Judge specific fields (optional)
    reasoning: Optional[str] = None  # Judge's reasoning/explanation
    sub_scores: Optional[Dict[str, float]] = None  # Multiple dimension scores
    confidence: Optional[float] = None  # Confidence in the judgment
    raw_response: Optional[str] = None  # Raw LLM response
    
    # NLI specific fields (optional)
    nli_relation: Optional[str] = None  # "entailment", "contradiction", "neutral"
    probabilities: Optional[List[float]] = None  # Class probabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "score": self.score,
            "passed": self.passed,
            "evaluation_type": self.evaluation_type.value,
            "details": self.details,
            "metadata": self.metadata or {},
            "reasoning": self.reasoning,
            "sub_scores": self.sub_scores,
            "confidence": self.confidence,
            "nli_relation": self.nli_relation,
        }
    
    @classmethod
    def from_nli_result(cls, nli_result: Dict[str, Any]) -> "EvaluationResult":
        """Create from NLI model result."""
        return cls(
            score=nli_result.get("confidence", 0.0),
            passed=nli_result.get("nli_relation") == "entailment",
            details=nli_result,
            evaluation_type=EvaluationType.NLI,
            nli_relation=nli_result.get("nli_relation"),
            probabilities=nli_result.get("probabilities"),
            confidence=nli_result.get("confidence"),
        )
    
    @classmethod
    def from_llm_judge_result(
        cls, 
        score: float,
        reasoning: str,
        sub_scores: Optional[Dict[str, float]] = None,
        raw_response: Optional[str] = None,
        threshold: float = 0.5,
        **kwargs
    ) -> "EvaluationResult":
        """Create from LLM Judge result."""
        return cls(
            score=score,
            passed=score >= threshold,
            details=kwargs,
            evaluation_type=EvaluationType.LLM_JUDGE,
            reasoning=reasoning,
            sub_scores=sub_scores,
            raw_response=raw_response,
        )


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

"""Data types for the pipeline layer.

Defines the mutable state that flows through pipeline steps, the
configuration object, and the final run result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from eval_framework.cache import BaseCache
from eval_framework.claims.types import DecompositionResult
from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.datasets.types import EvalSample
from eval_framework.metrics.adapters import BaseAdapter
from eval_framework.metrics.types import MetricResult, SampleResult


@dataclass
class SampleState:
    """Mutable state accumulated as a sample flows through the pipeline.

    Each step reads from and writes to this object. At the end of the
    pipeline, the state contains everything needed to produce metrics.
    """

    sample: EvalSample
    claims: Optional[DecompositionResult] = None
    evaluations: List[EvaluationResult] = field(default_factory=list)
    sample_result: Optional[SampleResult] = None
    errors: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return len(self.errors) > 0

    @property
    def claim_texts(self) -> List[str]:
        """Claim texts from decomposition, or the full response as a single claim."""
        if self.claims and self.claims.num_claims > 0:
            return self.claims.claim_texts
        return [self.sample.model_response]


@dataclass
class PipelineRunResult:
    """Complete output of a pipeline run."""

    sample_results: List[SampleResult] = field(default_factory=list)
    metrics: List[MetricResult] = field(default_factory=list)
    states: List[SampleState] = field(default_factory=list)

    # Timing
    total_time_seconds: float = 0.0
    step_timings: Dict[str, float] = field(default_factory=dict)

    # Error tracking
    failed_states: List[SampleState] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.states)

    @property
    def successful_count(self) -> int:
        return len(self.sample_results)

    @property
    def failed_count(self) -> int:
        return len(self.failed_states)

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.successful_count / self.total_samples

    def metrics_dict(self) -> Dict[str, float]:
        """Quick lookup: metric name → value."""
        return {m.name: m.value for m in self.metrics}

    def summary(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "successful": self.successful_count,
            "failed": self.failed_count,
            "success_rate": self.success_rate,
            "total_time_seconds": self.total_time_seconds,
            "metrics": self.metrics_dict(),
        }

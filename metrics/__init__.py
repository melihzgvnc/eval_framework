"""Metrics for evaluation scoring and aggregation.

This package provides metrics that operate on batches of evaluated
samples to produce aggregate scores:

- **Hallucination Rate**: Proportion of claims not supported by source.
- **Groundedness**: Proportion of claims supported by source.
- **Refusal Precision**: Of all refusals, how many were correct.
- **Refusal Recall**: Of cases requiring refusal, how many were refused.
- **Over-Refusal Rate**: Of answerable cases, how many were incorrectly refused.

Adapters bridge evaluator outputs (EvaluationResult) into the
standardized SampleResult format that metrics consume:

- **NLIAdapter**: Maps NLI relations → VerificationLabels.
- **LLMJudgeAdapter**: Thresholds judge scores → VerificationLabels.
- **PassFailAdapter**: Uses the binary passed field as a universal fallback.

Typical usage::

    from eval_framework.metrics import (
        hallucination_rate,
        groundedness,
        NLIAdapter,
        LLMJudgeAdapter,
    )

    # Adapt evaluator output, then compute metrics
    adapter = NLIAdapter()
    sample = adapter.adapt(sample_id="s1", ...)
    hr = hallucination_rate([sample])
"""

from .types import (
    ClaimVerification,
    MetricResult,
    RefusalLabel,
    SampleResult,
    VerificationLabel,
)
from .hallucination import hallucination_rate
from .groundedness import groundedness
from .refusal import over_refusal_rate, refusal_precision, refusal_recall
from .adapters import (
    BaseAdapter,
    LLMJudgeAdapter,
    NLIAdapter,
    PassFailAdapter,
)

__all__ = [
    # Types
    "ClaimVerification",
    "MetricResult",
    "RefusalLabel",
    "SampleResult",
    "VerificationLabel",
    # Metrics
    "hallucination_rate",
    "groundedness",
    "refusal_precision",
    "refusal_recall",
    "over_refusal_rate",
    # Adapters
    "BaseAdapter",
    "NLIAdapter",
    "LLMJudgeAdapter",
    "PassFailAdapter",
]

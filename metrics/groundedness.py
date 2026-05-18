"""Groundedness metric.

Measures the proportion of claims in model responses that ARE supported
(entailed) by the source material.

    groundedness = supported_claims / total_claims

A higher groundedness score indicates a more faithful model.
This is the complement of hallucination rate (when unverifiable counts
as hallucination): groundedness = 1 - hallucination_rate.
"""

from typing import List

from eval_framework.metrics.types import (
    MetricResult,
    SampleResult,
    VerificationLabel,
)


def groundedness(
    samples: List[SampleResult],
    strict: bool = True,
) -> MetricResult:
    """Compute the groundedness score across a batch of samples.

    Args:
        samples: List of evaluated samples with claim verifications.
        strict: If True (default), only claims explicitly supported
            (entailed) count toward groundedness. If False, unverifiable
            claims are excluded from the denominator (only supported vs
            contradicted are considered).

    Returns:
        MetricResult with the groundedness score (0.0 = nothing grounded,
        1.0 = all claims grounded).
    """
    supported_count = 0
    contradicted_count = 0
    unverifiable_count = 0
    total_claims = 0

    for sample in samples:
        for cv in sample.claim_verifications:
            total_claims += 1
            if cv.label == VerificationLabel.SUPPORTED:
                supported_count += 1
            elif cv.label == VerificationLabel.CONTRADICTED:
                contradicted_count += 1
            else:
                unverifiable_count += 1

    if strict:
        # All claims in denominator
        denominator = total_claims
    else:
        # Exclude unverifiable from denominator
        denominator = supported_count + contradicted_count

    score = supported_count / denominator if denominator > 0 else 0.0

    return MetricResult(
        name="groundedness",
        value=score,
        count=total_claims,
        details={
            "supported_claims": supported_count,
            "contradicted_claims": contradicted_count,
            "unverifiable_claims": unverifiable_count,
            "total_claims": total_claims,
            "total_samples": len(samples),
            "strict": strict,
            "denominator": denominator,
        },
    )

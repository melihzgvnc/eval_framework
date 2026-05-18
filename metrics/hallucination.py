"""Hallucination rate metric.

Measures the proportion of claims in model responses that are NOT
supported by the source material (i.e., contradicted or unverifiable).

    hallucination_rate = (contradicted + unverifiable) / total_claims

A lower hallucination rate indicates a more faithful model.
"""

from typing import List

from eval_framework.metrics.types import (
    MetricResult,
    SampleResult,
    VerificationLabel,
)


def hallucination_rate(
    samples: List[SampleResult],
    count_unverifiable_as_hallucination: bool = True,
) -> MetricResult:
    """Compute the hallucination rate across a batch of samples.

    Args:
        samples: List of evaluated samples with claim verifications.
        count_unverifiable_as_hallucination: If True (default), claims
            that are neither supported nor contradicted (neutral/unverifiable)
            are counted as hallucinations. Set to False to only count
            explicit contradictions.

    Returns:
        MetricResult with the hallucination rate (0.0 = no hallucination,
        1.0 = all claims hallucinated).
    """
    total_claims = 0
    hallucinated_claims = 0
    contradicted_count = 0
    unverifiable_count = 0
    supported_count = 0

    for sample in samples:
        for cv in sample.claim_verifications:
            total_claims += 1
            if cv.label == VerificationLabel.CONTRADICTED:
                hallucinated_claims += 1
                contradicted_count += 1
            elif cv.label == VerificationLabel.UNVERIFIABLE:
                unverifiable_count += 1
                if count_unverifiable_as_hallucination:
                    hallucinated_claims += 1
            else:
                supported_count += 1

    rate = hallucinated_claims / total_claims if total_claims > 0 else 0.0

    return MetricResult(
        name="hallucination_rate",
        value=rate,
        count=total_claims,
        details={
            "hallucinated_claims": hallucinated_claims,
            "supported_claims": supported_count,
            "contradicted_claims": contradicted_count,
            "unverifiable_claims": unverifiable_count,
            "total_claims": total_claims,
            "total_samples": len(samples),
            "count_unverifiable_as_hallucination": count_unverifiable_as_hallucination,
        },
    )

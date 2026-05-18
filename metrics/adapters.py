"""Adapters that bridge EvaluationResult → SampleResult.

This module provides a plug-and-play adapter layer so that metrics can
consume results from *any* evaluation strategy (NLI, LLM Judge, or
custom) without the metrics themselves needing to know which evaluator
produced the data.

Architecture:
    BaseAdapter (ABC)
    ├── NLIAdapter        — maps NLI relations to VerificationLabels
    ├── LLMJudgeAdapter   — thresholds judge scores into VerificationLabels
    └── (custom adapters)

Typical usage::

    from eval_framework.metrics.adapters import NLIAdapter, LLMJudgeAdapter

    # NLI path
    adapter = NLIAdapter()
    sample_result = adapter.adapt(
        sample_id="s1",
        source_text=context,
        model_response=answer,
        claim_texts=["claim 1", "claim 2"],
        evaluation_results=[nli_result_1, nli_result_2],
    )

    # LLM Judge path
    adapter = LLMJudgeAdapter(supported_threshold=0.7)
    sample_result = adapter.adapt(
        sample_id="s2",
        source_text=context,
        model_response=answer,
        claim_texts=["claim 1", "claim 2"],
        evaluation_results=[judge_result_1, judge_result_2],
    )

    # Then pass to any metric
    from eval_framework.metrics import hallucination_rate
    hr = hallucination_rate([sample_result])
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from eval_framework.core.types import EvaluationResult, EvaluationType
from eval_framework.metrics.types import (
    ClaimVerification,
    SampleResult,
    VerificationLabel,
)


class BaseAdapter(ABC):
    """Abstract base for evaluation-result-to-metric adapters.

    Subclasses implement the mapping logic from a specific evaluator's
    output format into the standardized ``ClaimVerification`` labels
    that metrics consume.
    """

    @abstractmethod
    def to_verification(
        self,
        claim_text: str,
        evaluation_result: EvaluationResult,
    ) -> ClaimVerification:
        """Convert a single EvaluationResult into a ClaimVerification.

        Args:
            claim_text: The claim text that was evaluated.
            evaluation_result: The evaluator's output for this claim.

        Returns:
            ClaimVerification with a discrete label.
        """
        pass

    def adapt(
        self,
        sample_id: str,
        source_text: str,
        model_response: str,
        claim_texts: List[str],
        evaluation_results: List[EvaluationResult],
        is_refusal: bool = False,
        should_refuse: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SampleResult:
        """Adapt a batch of per-claim evaluation results into a SampleResult.

        This is the main entry point. It iterates over claims and their
        corresponding evaluation results, converts each to a
        ClaimVerification, and bundles them into a SampleResult ready
        for metric computation.

        Args:
            sample_id: Unique identifier for this sample.
            source_text: The ground-truth / context text.
            model_response: The model-generated response.
            claim_texts: List of claim strings (from decomposition).
            evaluation_results: Corresponding EvaluationResult per claim.
            is_refusal: Whether the model refused to answer.
            should_refuse: Whether refusal was the expected behavior.
            metadata: Optional additional metadata.

        Returns:
            SampleResult ready for metric functions.

        Raises:
            ValueError: If claim_texts and evaluation_results have
                different lengths.
        """
        if len(claim_texts) != len(evaluation_results):
            raise ValueError(
                f"Mismatch: {len(claim_texts)} claims but "
                f"{len(evaluation_results)} evaluation results."
            )

        verifications = [
            self.to_verification(claim_text, eval_result)
            for claim_text, eval_result in zip(claim_texts, evaluation_results)
        ]

        return SampleResult(
            sample_id=sample_id,
            source_text=source_text,
            model_response=model_response,
            claim_verifications=verifications,
            is_refusal=is_refusal,
            should_refuse=should_refuse,
            metadata=metadata or {},
        )

    def adapt_single(
        self,
        sample_id: str,
        source_text: str,
        model_response: str,
        evaluation_result: EvaluationResult,
        is_refusal: bool = False,
        should_refuse: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SampleResult:
        """Adapt a single whole-response evaluation into a SampleResult.

        Use this when the evaluator scored the entire response as one
        unit (no claim decomposition). The response itself becomes the
        single "claim".

        Args:
            sample_id: Unique identifier for this sample.
            source_text: The ground-truth / context text.
            model_response: The model-generated response.
            evaluation_result: The evaluator's output for the whole response.
            is_refusal: Whether the model refused to answer.
            should_refuse: Whether refusal was the expected behavior.
            metadata: Optional additional metadata.

        Returns:
            SampleResult with a single ClaimVerification.
        """
        return self.adapt(
            sample_id=sample_id,
            source_text=source_text,
            model_response=model_response,
            claim_texts=[model_response],
            evaluation_results=[evaluation_result],
            is_refusal=is_refusal,
            should_refuse=should_refuse,
            metadata=metadata,
        )


@dataclass
class NLIAdapter(BaseAdapter):
    """Adapter for NLI evaluator results.

    Maps NLI relation labels directly to VerificationLabels:
        - entailment   → SUPPORTED
        - contradiction → CONTRADICTED
        - neutral/unknown → UNVERIFIABLE
    """

    def to_verification(
        self,
        claim_text: str,
        evaluation_result: EvaluationResult,
    ) -> ClaimVerification:
        """Map an NLI EvaluationResult to a ClaimVerification."""
        relation = evaluation_result.nli_relation

        if relation == "entailment":
            label = VerificationLabel.SUPPORTED
        elif relation == "contradiction":
            label = VerificationLabel.CONTRADICTED
        else:
            label = VerificationLabel.UNVERIFIABLE

        return ClaimVerification(
            claim_text=claim_text,
            label=label,
            confidence=evaluation_result.confidence or evaluation_result.score,
            metadata={
                "nli_relation": relation,
                "probabilities": evaluation_result.probabilities,
                "evaluation_type": EvaluationType.NLI.value,
            },
        )


@dataclass
class LLMJudgeAdapter(BaseAdapter):
    """Adapter for LLM Judge evaluator results.

    Thresholds the judge's continuous score into discrete verification
    labels using two configurable boundaries:

        score >= supported_threshold   → SUPPORTED
        score <= contradicted_threshold → CONTRADICTED
        otherwise                      → UNVERIFIABLE

    The score is expected to be normalized to 0–1 (as produced by
    LLMJudge.evaluate()).
    """

    supported_threshold: float = 0.7
    contradicted_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.contradicted_threshold >= self.supported_threshold:
            raise ValueError(
                f"contradicted_threshold ({self.contradicted_threshold}) must be "
                f"less than supported_threshold ({self.supported_threshold})."
            )

    def to_verification(
        self,
        claim_text: str,
        evaluation_result: EvaluationResult,
    ) -> ClaimVerification:
        """Map an LLM Judge EvaluationResult to a ClaimVerification."""
        score = evaluation_result.score

        if score >= self.supported_threshold:
            label = VerificationLabel.SUPPORTED
        elif score <= self.contradicted_threshold:
            label = VerificationLabel.CONTRADICTED
        else:
            label = VerificationLabel.UNVERIFIABLE

        return ClaimVerification(
            claim_text=claim_text,
            label=label,
            confidence=evaluation_result.confidence or score,
            metadata={
                "judge_score": score,
                "reasoning": evaluation_result.reasoning,
                "evaluation_type": EvaluationType.LLM_JUDGE.value,
                "supported_threshold": self.supported_threshold,
                "contradicted_threshold": self.contradicted_threshold,
            },
        )


@dataclass
class PassFailAdapter(BaseAdapter):
    """Simple adapter that uses the binary passed/failed field.

    Maps any EvaluationResult based solely on its ``passed`` boolean:
        - passed=True  → SUPPORTED
        - passed=False → CONTRADICTED

    Useful as a universal fallback for any evaluator type.
    """

    def to_verification(
        self,
        claim_text: str,
        evaluation_result: EvaluationResult,
    ) -> ClaimVerification:
        """Map based on the passed boolean."""
        label = (
            VerificationLabel.SUPPORTED
            if evaluation_result.passed
            else VerificationLabel.CONTRADICTED
        )

        return ClaimVerification(
            claim_text=claim_text,
            label=label,
            confidence=evaluation_result.confidence or evaluation_result.score,
            metadata={
                "passed": evaluation_result.passed,
                "score": evaluation_result.score,
                "evaluation_type": evaluation_result.evaluation_type.value,
            },
        )

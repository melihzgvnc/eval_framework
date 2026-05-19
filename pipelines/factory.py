"""Convenience factory functions for common pipeline configurations.

These cover the 80% use case so users don't need to wire every
component manually.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from eval_framework.cache import BaseCache
from eval_framework.claims import ClaimDecomposer
from eval_framework.metrics import (
    groundedness,
    hallucination_rate,
    over_refusal_rate,
    refusal_precision,
    refusal_recall,
)
from eval_framework.metrics.adapters import (
    LLMJudgeAdapter,
    NLIAdapter,
    PassFailAdapter,
)
from eval_framework.metrics.types import MetricResult, SampleResult
from eval_framework.pipelines.pipeline import EvalPipeline, MetricFn

# Default metric set
DEFAULT_METRICS: List[MetricFn] = [
    hallucination_rate,
    groundedness,
    refusal_precision,
    refusal_recall,
    over_refusal_rate,
]


def create_nli_pipeline(
    nli_model_path: str,
    decomposer_model: Optional[str] = "gpt-4o",
    decomposer_api_key: Optional[str] = None,
    cache: Optional[BaseCache] = None,
    metrics: Optional[List[MetricFn]] = None,
    continue_on_error: bool = True,
    use_heuristic_decomposition: bool = False,
) -> EvalPipeline:
    """Create an NLI-based evaluation pipeline.

    Args:
        nli_model_path: Path/name of the HuggingFace NLI model.
        decomposer_model: OpenAI model for claim decomposition.
            Set to None to skip LLM decomposition (uses heuristic).
        decomposer_api_key: Optional API key for the decomposer.
        cache: Optional cache instance (shared across components).
        metrics: Metric functions to compute. Defaults to all available.
        continue_on_error: Whether to continue on per-sample failures.
        use_heuristic_decomposition: If True, skip LLM decomposition
            entirely and use sentence splitting.

    Returns:
        Configured EvalPipeline ready to call ``.run(samples)``.
    """
    from eval_framework.evaluators.nli_evaluator import NLIEvaluator

    evaluator = NLIEvaluator(model_path=nli_model_path, cache=cache)
    adapter = NLIAdapter()

    decomposer = None
    if not use_heuristic_decomposition and decomposer_model:
        decomposer = ClaimDecomposer(
            model_name=decomposer_model,
            api_key=decomposer_api_key,
            cache=cache,
            fallback_to_heuristic=True,
        )

    return EvalPipeline(
        evaluator=evaluator,
        adapter=adapter,
        decomposer=decomposer,
        metrics=metrics or DEFAULT_METRICS,
        cache=cache,
        continue_on_error=continue_on_error,
    )


def create_judge_pipeline(
    judge_model: str = "gpt-4o",
    api_key: Optional[str] = None,
    criteria: str = "factuality",
    threshold: float = 0.7,
    supported_threshold: float = 0.7,
    contradicted_threshold: float = 0.3,
    decompose: bool = False,
    decomposer_model: Optional[str] = None,
    cache: Optional[BaseCache] = None,
    metrics: Optional[List[MetricFn]] = None,
    continue_on_error: bool = True,
) -> EvalPipeline:
    """Create an LLM-Judge-based evaluation pipeline.

    Args:
        judge_model: OpenAI model for judgment.
        api_key: Optional API key (falls back to env var).
        criteria: Evaluation criteria (e.g., "factuality").
        threshold: Score threshold for pass/fail.
        supported_threshold: Adapter threshold for "supported" label.
        contradicted_threshold: Adapter threshold for "contradicted" label.
        decompose: Whether to decompose into claims before judging.
        decomposer_model: Model for decomposition (defaults to judge_model).
        cache: Optional cache instance.
        metrics: Metric functions to compute.
        continue_on_error: Whether to continue on per-sample failures.

    Returns:
        Configured EvalPipeline ready to call ``.run(samples)``.
    """
    from eval_framework.evaluators.llm_judge import LLMJudge

    evaluator = LLMJudge(
        model_name=judge_model,
        api_key=api_key,
        evaluation_criteria=criteria,
        threshold=threshold,
        cache=cache,
    )
    adapter = LLMJudgeAdapter(
        supported_threshold=supported_threshold,
        contradicted_threshold=contradicted_threshold,
    )

    decomposer = None
    if decompose:
        decomposer = ClaimDecomposer(
            model_name=decomposer_model or judge_model,
            api_key=api_key,
            cache=cache,
            fallback_to_heuristic=True,
        )

    return EvalPipeline(
        evaluator=evaluator,
        adapter=adapter,
        decomposer=decomposer,
        metrics=metrics or DEFAULT_METRICS,
        cache=cache,
        continue_on_error=continue_on_error,
    )

"""Evaluation pipeline orchestration.

Composes the framework's modular components (datasets, decomposition,
evaluators, adapters, metrics) into end-to-end evaluation workflows.

Typical usage::

    from eval_framework.datasets import load_dataset
    from eval_framework.pipelines import create_nli_pipeline

    samples, info, _ = load_dataset("data/eval_set.jsonl")
    pipeline = create_nli_pipeline(nli_model_path="cross-encoder/nli-deberta-v3-base")
    result = pipeline.run(samples)

    print(result.summary())
    for m in result.metrics:
        print(f"  {m.name}: {m.value:.4f}")

Custom pipeline::

    from eval_framework.pipelines import EvalPipeline
    from eval_framework.metrics import hallucination_rate, groundedness

    pipeline = EvalPipeline(
        evaluator=my_evaluator,
        adapter=my_adapter,
        decomposer=my_decomposer,
        metrics=[hallucination_rate, groundedness],
    )
    result = pipeline.run(samples)
"""

from .factory import create_judge_pipeline, create_nli_pipeline, DEFAULT_METRICS
from .pipeline import EvalPipeline
from .steps import AdaptStep, DecomposeStep, EvaluateStep, Step
from .types import PipelineRunResult, SampleState

__all__ = [
    # Pipeline
    "EvalPipeline",
    # Steps
    "Step",
    "DecomposeStep",
    "EvaluateStep",
    "AdaptStep",
    # Types
    "SampleState",
    "PipelineRunResult",
    # Factories
    "create_nli_pipeline",
    "create_judge_pipeline",
    "DEFAULT_METRICS",
]

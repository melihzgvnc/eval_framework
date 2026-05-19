"""Main pipeline orchestrator.

EvalPipeline wires steps together and iterates over samples,
producing metrics and optionally a report at the end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from eval_framework.cache import BaseCache
from eval_framework.claims import ClaimDecomposer
from eval_framework.core.base import BaseEvaluator
from eval_framework.datasets.types import EvalSample
from eval_framework.metrics.adapters import BaseAdapter
from eval_framework.metrics.types import MetricResult, SampleResult
from eval_framework.pipelines.steps import (
    AdaptStep,
    DecomposeStep,
    EvaluateStep,
    Step,
)
from eval_framework.pipelines.types import PipelineRunResult, SampleState

logger = logging.getLogger(__name__)


# Type alias for metric functions
MetricFn = Callable[[List[SampleResult]], MetricResult]

# Type alias for progress callbacks
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class EvalPipeline:
    """Configurable evaluation pipeline.

    Composes decomposition, evaluation, adaptation, and metrics into
    a single run. Each component is optional and swappable.

    Usage::

        pipeline = EvalPipeline(
            evaluator=NLIEvaluator(model_path="..."),
            adapter=NLIAdapter(),
            decomposer=ClaimDecomposer(model_name="gpt-4o"),
            metrics=[hallucination_rate, groundedness],
        )
        result = pipeline.run(samples)
    """

    # Core components
    evaluator: BaseEvaluator
    adapter: BaseAdapter
    decomposer: Optional[ClaimDecomposer] = None
    metrics: List[MetricFn] = field(default_factory=list)

    # Optional infrastructure
    cache: Optional[BaseCache] = None

    # Behavior
    continue_on_error: bool = True
    progress_callback: Optional[ProgressCallback] = None

    # Custom steps (advanced usage — override the default step sequence)
    custom_steps: Optional[List[Step]] = None

    def run(self, samples: List[EvalSample]) -> PipelineRunResult:
        """Run the full pipeline over all samples.

        Args:
            samples: List of EvalSamples to evaluate.

        Returns:
            PipelineRunResult with metrics, sample results, and diagnostics.
        """
        total = len(samples)
        run_start = time.perf_counter()

        steps = self._build_steps()
        all_states: List[SampleState] = []
        successful_results: List[SampleResult] = []
        failed_states: List[SampleState] = []

        for i, sample in enumerate(samples):
            if self.progress_callback:
                self.progress_callback(i, total, sample.sample_id)

            state = self.run_sample(sample, steps=steps)
            all_states.append(state)

            if state.failed:
                failed_states.append(state)
                if not self.continue_on_error:
                    logger.error(
                        "Pipeline aborted at sample '%s': %s",
                        sample.sample_id, state.errors,
                    )
                    break
            elif state.sample_result is not None:
                successful_results.append(state.sample_result)

        # Final progress callback
        if self.progress_callback:
            self.progress_callback(total, total, "done")

        # Compute metrics over successful samples
        computed_metrics = self._compute_metrics(successful_results)

        # Aggregate step timings
        step_timings = self._aggregate_timings(all_states)

        total_time = time.perf_counter() - run_start

        return PipelineRunResult(
            sample_results=successful_results,
            metrics=computed_metrics,
            states=all_states,
            total_time_seconds=total_time,
            step_timings=step_timings,
            failed_states=failed_states,
        )

    def run_sample(
        self,
        sample: EvalSample,
        steps: Optional[List[Step]] = None,
    ) -> SampleState:
        """Run a single sample through all steps.

        Useful for debugging individual samples outside a full run.

        Args:
            sample: The sample to process.
            steps: Optional pre-built step list (for reuse in run()).

        Returns:
            Final SampleState after all steps.
        """
        if steps is None:
            steps = self._build_steps()

        state = SampleState(sample=sample)

        for step in steps:
            state = step(state)
            # Stop processing this sample if a step failed and we're strict
            if state.failed and not self.continue_on_error:
                break

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_steps(self) -> List[Step]:
        """Construct the step sequence from configuration."""
        if self.custom_steps is not None:
            return self.custom_steps

        steps: List[Step] = []

        # Step 1: Decompose (optional)
        if self.decomposer is not None:
            steps.append(DecomposeStep(decomposer=self.decomposer))
        else:
            # Still run heuristic decomposition for sentence-level eval
            steps.append(DecomposeStep(
                decomposer=None, use_heuristic_fallback=True
            ))

        # Step 2: Evaluate
        steps.append(EvaluateStep(evaluator=self.evaluator))

        # Step 3: Adapt to metrics format
        steps.append(AdaptStep(adapter=self.adapter))

        return steps

    def _compute_metrics(
        self, results: List[SampleResult]
    ) -> List[MetricResult]:
        """Run all configured metric functions."""
        if not results or not self.metrics:
            return []

        computed = []
        for metric_fn in self.metrics:
            try:
                result = metric_fn(results)
                computed.append(result)
            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", metric_fn.__name__, exc)
        return computed

    @staticmethod
    def _aggregate_timings(states: List[SampleState]) -> Dict[str, float]:
        """Sum per-step timings across all samples."""
        totals: Dict[str, float] = {}
        for state in states:
            for step_name, elapsed in state.timings.items():
                totals[step_name] = totals.get(step_name, 0.0) + elapsed
        return totals

"""Pipeline steps.

Each step is a callable that takes a SampleState and returns it
(mutated). Steps are composable and independently testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from eval_framework.cache import BaseCache, make_key
from eval_framework.claims import ClaimDecomposer, decompose_heuristic
from eval_framework.claims.types import DecompositionResult
from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.metrics.adapters import BaseAdapter
from eval_framework.pipelines.types import SampleState

logger = logging.getLogger(__name__)


class Step:
    """Base class for pipeline steps.

    Subclasses implement ``execute(state)`` with the actual logic.
    The ``__call__`` wrapper handles timing and error capture.
    """

    name: str = "step"

    def __call__(self, state: SampleState) -> SampleState:
        """Run the step with timing and error handling."""
        start = time.perf_counter()
        try:
            state = self.execute(state)
        except Exception as exc:
            logger.warning(
                "Step '%s' failed for sample '%s': %s",
                self.name, state.sample.sample_id, exc,
            )
            state.errors.append(f"{self.name}: {exc}")
        elapsed = time.perf_counter() - start
        state.timings[self.name] = state.timings.get(self.name, 0.0) + elapsed
        return state

    def execute(self, state: SampleState) -> SampleState:
        """Override this with step logic."""
        raise NotImplementedError


@dataclass
class DecomposeStep(Step):
    """Decompose the model response into atomic claims.

    If a decomposer is provided, uses LLM-based decomposition.
    Otherwise falls back to heuristic sentence splitting.
    """

    name: str = "decompose"
    decomposer: Optional[ClaimDecomposer] = None
    use_heuristic_fallback: bool = True

    def execute(self, state: SampleState) -> SampleState:
        text = state.sample.model_response
        context = state.sample.source_text

        if not text or not text.strip():
            state.claims = DecompositionResult(
                claims=[], original_text="", method="skip",
                metadata={"reason": "empty_response"},
            )
            return state

        if self.decomposer is not None:
            state.claims = self.decomposer.decompose(text, context=context)
        elif self.use_heuristic_fallback:
            state.claims = decompose_heuristic(text, split_clauses=True)
        else:
            # No decomposition — treat whole response as single claim
            state.claims = DecompositionResult(
                claims=[], original_text=text, method="none",
            )

        return state


@dataclass
class EvaluateStep(Step):
    """Run the evaluator on each claim (or the whole response).

    Produces one EvaluationResult per claim text. If no claims were
    decomposed, evaluates the full response as a single unit.
    """

    name: str = "evaluate"
    evaluator: BaseEvaluator = None  # type: ignore[assignment]

    def execute(self, state: SampleState) -> SampleState:
        if self.evaluator is None:
            state.errors.append("evaluate: no evaluator configured")
            return state

        source = state.sample.source_text
        claim_texts = state.claim_texts

        evaluations = []
        for claim_text in claim_texts:
            result = self.evaluator.evaluate(source, claim_text)
            evaluations.append(result)

        state.evaluations = evaluations
        return state


@dataclass
class AdaptStep(Step):
    """Convert evaluations into a metrics-ready SampleResult.

    Uses the configured adapter to bridge EvaluationResult objects
    into the standardized ClaimVerification format.
    """

    name: str = "adapt"
    adapter: BaseAdapter = None  # type: ignore[assignment]

    def execute(self, state: SampleState) -> SampleState:
        if self.adapter is None:
            state.errors.append("adapt: no adapter configured")
            return state

        if not state.evaluations:
            # No evaluations — produce an empty SampleResult
            state.sample_result = self.adapter.adapt(
                sample_id=state.sample.sample_id,
                source_text=state.sample.source_text,
                model_response=state.sample.model_response,
                claim_texts=[],
                evaluation_results=[],
                is_refusal=state.sample.should_refuse
                and not state.sample.model_response.strip(),
                should_refuse=state.sample.should_refuse,
                metadata=state.sample.metadata,
            )
            return state

        claim_texts = state.claim_texts
        evaluations = state.evaluations

        # Ensure alignment
        if len(claim_texts) != len(evaluations):
            state.errors.append(
                f"adapt: claim/evaluation count mismatch "
                f"({len(claim_texts)} vs {len(evaluations)})"
            )
            return state

        state.sample_result = self.adapter.adapt(
            sample_id=state.sample.sample_id,
            source_text=state.sample.source_text,
            model_response=state.sample.model_response,
            claim_texts=claim_texts,
            evaluation_results=evaluations,
            is_refusal=False,  # Pipeline doesn't detect refusal yet
            should_refuse=state.sample.should_refuse,
            metadata=state.sample.metadata,
        )
        return state

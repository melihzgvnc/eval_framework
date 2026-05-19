"""NLI-based evaluation strategy.

This module is the *orchestration* layer: it wires a HuggingFace NLI model
to the relation labels defined under ``eval_framework.nli`` and returns a
unified ``EvaluationResult``. NLI domain assets (relation labels, passing
relation) live under ``eval_framework.nli``.
"""

from dataclasses import dataclass
from typing import Optional

from eval_framework.cache import BaseCache, make_key
from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.models import HuggingFaceModel
from eval_framework.nli import PASSING_RELATION


@dataclass
class NLIEvaluator(BaseEvaluator):
    """Evaluator backed by a HuggingFace NLI sequence-classification model."""

    def __init__(self, model_path: str, cache: Optional[BaseCache] = None):
        """Initialize the NLI evaluator.

        Args:
            model_path: Path/name of the HuggingFace NLI model.
            cache: Optional cache for memoizing identical (premise, hypothesis,
                model) triples. ``None`` disables caching.
        """
        self.model = HuggingFaceModel(model_type="sequence_classification")
        self.model.load(model_path)
        self.model_path = model_path
        self.cache = cache

    def evaluate(self, sample: str, answer: str) -> EvaluationResult:
        """Run NLI prediction and wrap the result.

        Args:
            sample: Premise text (typically the grounding context).
            answer: Hypothesis text (typically the model-generated answer).

        Returns:
            EvaluationResult marked passed when the relation is the
            configured passing relation (entailment by default).
        """
        # Cache lookup
        cache_key = None
        if self.cache is not None:
            cache_key = make_key(
                "nli",
                premise=sample,
                hypothesis=answer,
                model=self.model_path,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return EvaluationResult.from_dict(cached)

        result = self.model.nli_predict(sample, answer)

        # Override the default pass logic in from_nli_result so the passing
        # relation comes from the nli/ module rather than being hardcoded.
        relation = result.get("nli_relation")
        evaluation = EvaluationResult.from_nli_result(result)
        evaluation.passed = relation == PASSING_RELATION.value

        if cache_key is not None:
            self.cache.set(cache_key, evaluation.to_dict())

        return evaluation

"""NLI-based evaluation strategy.

This module is the *orchestration* layer: it wires a HuggingFace NLI model
to the relation labels defined under ``eval_framework.nli`` and returns a
unified ``EvaluationResult``. NLI domain assets (relation labels, passing
relation) live under ``eval_framework.nli``.
"""

from dataclasses import dataclass

from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.models import HuggingFaceModel
from eval_framework.nli import PASSING_RELATION


@dataclass
class NLIEvaluator(BaseEvaluator):
    """Evaluator backed by a HuggingFace NLI sequence-classification model."""

    def __init__(self, model_path: str):
        self.model = HuggingFaceModel(model_type="sequence_classification")
        self.model.load(model_path)

    def evaluate(self, sample: str, answer: str) -> EvaluationResult:
        """Run NLI prediction and wrap the result.

        Args:
            sample: Premise text (typically the grounding context).
            answer: Hypothesis text (typically the model-generated answer).

        Returns:
            EvaluationResult marked passed when the relation is the
            configured passing relation (entailment by default).
        """
        result = self.model.nli_predict(sample, answer)

        # Override the default pass logic in from_nli_result so the passing
        # relation comes from the nli/ module rather than being hardcoded.
        relation = result.get("nli_relation")
        evaluation = EvaluationResult.from_nli_result(result)
        evaluation.passed = relation == PASSING_RELATION.value
        return evaluation

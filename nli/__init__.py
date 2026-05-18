"""NLI-specific assets for entailment-based evaluation.

This package holds NLI domain knowledge such as relation labels. The
orchestration logic lives in ``eval_framework.evaluators.nli_evaluator``.
"""

from .relations import NLIRelation, PASSING_RELATION

__all__ = ["NLIRelation", "PASSING_RELATION"]

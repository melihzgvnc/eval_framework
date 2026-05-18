"""NLI relation definitions.

Centralizes the set of NLI labels and the convention for which relation
counts as a "pass" in evaluation. Keeping this here (rather than in the
evaluator) means the labels can be reused by claim decomposition,
reporting, and any future NLI-driven logic.
"""

from enum import Enum


class NLIRelation(str, Enum):
    """Standard NLI relations."""

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


# Relation that represents a successful entailment check.
PASSING_RELATION: NLIRelation = NLIRelation.ENTAILMENT

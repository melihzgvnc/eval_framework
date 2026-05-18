"""Claim decomposition for evaluation.

This package handles breaking text into atomic, independently verifiable
claims before NLI evaluation. It provides both an LLM-based decomposer
(using structured output) and a lightweight heuristic fallback.

Typical usage::

    from eval_framework.claims import ClaimDecomposer

    decomposer = ClaimDecomposer(model_name="gpt-4o")
    result = decomposer.decompose("Paris is the capital of France and has a population of 2 million.")
    for claim in result.claims:
        print(claim.text)
"""

from .types import Claim, ClaimType, DecompositionResult
from .decomposer import ClaimDecomposer, decompose_heuristic
from .prompts import DECOMPOSITION_PROMPT, DECOMPOSITION_WITH_CONTEXT_PROMPT, SYSTEM_MESSAGE
from .schemas import DECOMPOSITION_SCHEMA

__all__ = [
    "Claim",
    "ClaimType",
    "ClaimDecomposer",
    "DecompositionResult",
    "decompose_heuristic",
    "DECOMPOSITION_PROMPT",
    "DECOMPOSITION_WITH_CONTEXT_PROMPT",
    "DECOMPOSITION_SCHEMA",
    "SYSTEM_MESSAGE",
]

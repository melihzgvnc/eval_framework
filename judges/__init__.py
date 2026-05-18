"""Judge-specific assets for LLM-as-a-Judge evaluation.

This package holds prompt templates, JSON schemas, and other domain
knowledge used by judge-based evaluators. The orchestration logic lives
in ``eval_framework.evaluators.llm_judge``.
"""

from .prompts import RUBRICS, SYSTEM_MESSAGE
from .schemas import JUDGMENT_SCHEMA

__all__ = ["RUBRICS", "SYSTEM_MESSAGE", "JUDGMENT_SCHEMA"]

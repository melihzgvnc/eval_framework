"""LLM-as-a-Judge evaluation strategy.

This module is the *orchestration* layer: it wires an OpenAI model to the
judge prompts and JSON schema defined under ``eval_framework.judges`` and
returns a unified ``EvaluationResult``. Domain assets (rubrics, schemas,
system message) live under ``eval_framework.judges``.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional

from eval_framework.core.base import BaseEvaluator
from eval_framework.core.types import EvaluationResult
from eval_framework.judges import JUDGMENT_SCHEMA, RUBRICS, SYSTEM_MESSAGE
from eval_framework.models import OpenAIModel


@dataclass
class LLMJudge(BaseEvaluator):
    """LLM-as-a-Judge evaluator with enforced structured JSON output."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        evaluation_criteria: str = "factuality",
        threshold: float = 0.7,
    ):
        """Initialize the LLM judge.

        Args:
            model_name: OpenAI model to use for judgment.
            api_key: Optional API key (falls back to OPENAI_API_KEY env var).
            evaluation_criteria: One of the keys in ``judges.RUBRICS``.
            threshold: Score threshold (0-1) for ``passed=True``.

        Raises:
            ValueError: If ``evaluation_criteria`` is not a known rubric.
        """
        if evaluation_criteria not in RUBRICS:
            raise ValueError(
                f"Unknown evaluation_criteria '{evaluation_criteria}'. "
                f"Must be one of: {sorted(RUBRICS)}"
            )

        # Use Chat Completions for structured output support.
        self.model = OpenAIModel(
            model=model_name,
            api_key=api_key,
            use_responses_api=False,
        )
        self.evaluation_criteria = evaluation_criteria
        self.threshold = threshold

    def evaluate(self, sample: str, answer: str) -> EvaluationResult:
        """Evaluate an answer using LLM-as-a-Judge with structured output.

        Args:
            sample: Context or question.
            answer: Model-generated answer to evaluate.

        Returns:
            EvaluationResult with LLM Judge details.

        Raises:
            ValueError: If the structured output call fails.
        """
        rubric = RUBRICS[self.evaluation_criteria]
        prompt = rubric.format(context=sample, question=sample, answer=answer)

        try:
            judgment = self.model.generate_structured(
                prompt=prompt,
                schema=JUDGMENT_SCHEMA,
                system=SYSTEM_MESSAGE,
                temperature=0.1,
                max_tokens=500,
            )
        except Exception as e:
            raise ValueError(f"LLM Judge evaluation failed: {e}") from e

        score = float(judgment.get("score", 0.0))
        reasoning = judgment.get("reasoning", "No reasoning provided")
        sub_scores = judgment.get("sub_scores")
        confidence = judgment.get("confidence")

        return EvaluationResult.from_llm_judge_result(
            score=score / 100.0,  # normalize to 0-1
            reasoning=reasoning,
            sub_scores=sub_scores,
            raw_response=json.dumps(judgment, indent=2),
            threshold=self.threshold,
            evaluation_criteria=self.evaluation_criteria,
            sample=sample,
            answer=answer,
            confidence=confidence,
        )

    def evaluate_multi_criteria(
        self,
        sample: str,
        answer: str,
        criteria_list: Optional[list[str]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Evaluate using multiple criteria.

        Args:
            sample: Context or question.
            answer: Model-generated answer.
            criteria_list: List of criteria keys to evaluate. Defaults to
                ``["factuality", "helpfulness", "coherence"]``.

        Returns:
            Dictionary mapping each criterion to its EvaluationResult.
        """
        if criteria_list is None:
            criteria_list = ["factuality", "helpfulness", "coherence"]

        results: Dict[str, EvaluationResult] = {}
        original_criteria = self.evaluation_criteria
        try:
            for criteria in criteria_list:
                if criteria not in RUBRICS:
                    raise ValueError(
                        f"Unknown evaluation_criteria '{criteria}'. "
                        f"Must be one of: {sorted(RUBRICS)}"
                    )
                self.evaluation_criteria = criteria
                results[criteria] = self.evaluate(sample, answer)
        finally:
            self.evaluation_criteria = original_criteria

        return results

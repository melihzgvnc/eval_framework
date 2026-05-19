"""Dataset validation.

Catches common problems before evaluation starts: missing fields,
empty values, duplicate IDs, type mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from eval_framework.datasets.types import EvalSample


@dataclass
class ValidationError:
    """A single validation issue."""

    sample_id: str
    field: str
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        return f"[{self.severity}] sample={self.sample_id} field={self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Outcome of validating a dataset."""

    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings are acceptable)."""
        return len(self.errors) == 0

    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)

    def summary(self) -> str:
        if self.is_valid and not self.warnings:
            return "Validation passed: no issues found."
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return f"Validation: {', '.join(parts)}."


def validate_samples(
    samples: List[EvalSample],
    require_query: bool = False,
    require_unique_ids: bool = True,
    min_source_length: int = 1,
    min_response_length: int = 1,
) -> ValidationResult:
    """Validate a list of EvalSamples.

    Args:
        samples: Samples to validate.
        require_query: If True, missing query is an error (not warning).
        require_unique_ids: If True, duplicate sample_ids are errors.
        min_source_length: Minimum character length for source_text.
        min_response_length: Minimum character length for model_response.

    Returns:
        ValidationResult with all detected issues.
    """
    result = ValidationResult()
    seen_ids: Set[str] = set()

    for i, sample in enumerate(samples):
        sid = sample.sample_id or f"index_{i}"

        # Duplicate ID check
        if require_unique_ids:
            if sid in seen_ids:
                result.errors.append(
                    ValidationError(sid, "sample_id", "Duplicate sample_id")
                )
            seen_ids.add(sid)

        # Required: source_text
        if not sample.source_text or len(sample.source_text.strip()) < min_source_length:
            result.errors.append(
                ValidationError(
                    sid, "source_text",
                    f"Missing or too short (min {min_source_length} chars)"
                )
            )

        # Required: model_response
        if not sample.model_response or len(sample.model_response.strip()) < min_response_length:
            result.errors.append(
                ValidationError(
                    sid, "model_response",
                    f"Missing or too short (min {min_response_length} chars)"
                )
            )

        # Optional: query
        if not sample.query:
            issue = ValidationError(sid, "query", "Missing query/prompt")
            if require_query:
                result.errors.append(issue)
            else:
                result.warnings.append(issue)

        # Type check: should_refuse
        if not isinstance(sample.should_refuse, bool):
            result.warnings.append(
                ValidationError(
                    sid, "should_refuse",
                    f"Expected bool, got {type(sample.should_refuse).__name__}"
                )
            )

    return result

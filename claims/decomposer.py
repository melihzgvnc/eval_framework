"""Claim decomposition orchestration.

Provides both LLM-based and heuristic-based strategies for breaking text
into atomic claims. The LLM strategy uses structured output via OpenAI;
the heuristic strategy uses sentence splitting as a lightweight fallback.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from eval_framework.cache import BaseCache, make_key
from eval_framework.claims.types import Claim, ClaimType, DecompositionResult
from eval_framework.claims.prompts import (
    DECOMPOSITION_PROMPT,
    DECOMPOSITION_WITH_CONTEXT_PROMPT,
    SYSTEM_MESSAGE,
)
from eval_framework.claims.schemas import DECOMPOSITION_SCHEMA
from eval_framework.models import OpenAIModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic decomposition (no LLM required)
# ---------------------------------------------------------------------------

# Sentence boundary pattern: split on period, exclamation, or question mark
# followed by whitespace and an uppercase letter.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])", re.UNICODE)

# Common abbreviations that should NOT be treated as sentence boundaries.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "approx"}

# Clause-level splitters for compound sentences.
_CLAUSE_SPLITTERS = re.compile(
    r"\s*(?:,\s*(?:and|but|or|while|whereas|although|however)\s+|;\s*)", re.IGNORECASE
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex heuristics.

    Handles common abbreviations by re-joining false splits.
    """
    raw_parts = _SENTENCE_SPLIT.split(text.strip())

    # Re-join parts that were split on abbreviations (e.g., "Dr. Smith")
    sentences: List[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Check if previous sentence ended with an abbreviation
        if sentences:
            prev = sentences[-1]
            # Get the last word before the period
            last_word_match = re.search(r"(\w+)\.\s*$", prev)
            if last_word_match and last_word_match.group(1).lower() in _ABBREVIATIONS:
                sentences[-1] = prev + " " + part
                continue
        sentences.append(part)

    return sentences


def _split_clauses(sentence: str, min_clause_length: int = 20) -> List[str]:
    """Split a compound sentence into clauses.

    Only splits if the resulting clauses are long enough to be
    meaningful standalone claims.
    """
    parts = _CLAUSE_SPLITTERS.split(sentence)
    clauses = [p.strip() for p in parts if p.strip()]

    # Only accept the split if all parts are substantial
    if all(len(c) >= min_clause_length for c in clauses) and len(clauses) > 1:
        return clauses
    return [sentence]


def decompose_heuristic(
    text: str,
    split_clauses: bool = False,
    min_claim_length: int = 10,
) -> DecompositionResult:
    """Decompose text into claims using sentence/clause splitting.

    This is a lightweight fallback that doesn't require an LLM. It splits
    text on sentence boundaries and optionally on clause boundaries.

    Args:
        text: The text to decompose.
        split_clauses: Whether to further split compound sentences.
        min_claim_length: Minimum character length for a valid claim.

    Returns:
        DecompositionResult with heuristic-extracted claims.
    """
    if not text or not text.strip():
        return DecompositionResult(
            claims=[],
            original_text=text or "",
            method="heuristic",
            metadata={"reason": "empty_input"},
        )

    sentences = _split_sentences(text)

    claim_texts: List[str] = []
    for sentence in sentences:
        if split_clauses:
            clauses = _split_clauses(sentence)
            claim_texts.extend(clauses)
        else:
            claim_texts.append(sentence)

    # Filter out claims that are too short to be meaningful
    claim_texts = [c for c in claim_texts if len(c) >= min_claim_length]

    claims = [
        Claim(
            text=ct,
            index=i,
            claim_type=ClaimType.UNKNOWN,
            source_sentence=ct,
        )
        for i, ct in enumerate(claim_texts)
    ]

    return DecompositionResult(
        claims=claims,
        original_text=text,
        method="heuristic",
        metadata={"split_clauses": split_clauses},
    )


# ---------------------------------------------------------------------------
# LLM-based decomposition
# ---------------------------------------------------------------------------


@dataclass
class ClaimDecomposer:
    """LLM-powered claim decomposer using structured output.

    Uses an OpenAI model to decompose text into atomic, independently
    verifiable claims via structured JSON generation.
    """

    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    fallback_to_heuristic: bool = True
    cache: Optional[BaseCache] = None
    _model: Optional[OpenAIModel] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the underlying OpenAI model."""
        try:
            self._model = OpenAIModel(
                model_name=self.model_name,
                model=self.model_name,
                api_key=self.api_key,
                use_responses_api=False,
            )
        except (ValueError, RuntimeError) as exc:
            if self.fallback_to_heuristic:
                logger.warning(
                    "Failed to initialize LLM for claim decomposition (%s). "
                    "Will fall back to heuristic decomposition.",
                    exc,
                )
                self._model = None
            else:
                raise

    def decompose(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> DecompositionResult:
        """Decompose text into atomic claims.

        Attempts LLM-based decomposition first. If the LLM is unavailable
        or fails and ``fallback_to_heuristic`` is True, falls back to
        sentence-splitting heuristics.

        Args:
            text: The text to decompose into claims.
            context: Optional context for resolving ambiguous references
                (pronouns, etc.) in the text.

        Returns:
            DecompositionResult containing the extracted claims.
        """
        if not text or not text.strip():
            return DecompositionResult(
                claims=[],
                original_text=text or "",
                method="llm",
                metadata={"reason": "empty_input"},
            )

        # Cache lookup (LLM path only — heuristics are cheap)
        cache_key = None
        if self.cache is not None and self._model is not None:
            cache_key = make_key(
                "claims",
                text=text,
                context=context,
                model=self.model_name,
                temperature=self.temperature,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return DecompositionResult.from_dict(cached)

        # Try LLM decomposition
        if self._model is not None:
            try:
                result = self._decompose_with_llm(text, context)
                if cache_key is not None:
                    self.cache.set(cache_key, result.to_dict())
                return result
            except Exception as exc:
                logger.warning(
                    "LLM decomposition failed: %s. Falling back to heuristic.",
                    exc,
                )
                if not self.fallback_to_heuristic:
                    raise

        # Fallback to heuristic
        return decompose_heuristic(text, split_clauses=True)

    def _decompose_with_llm(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> DecompositionResult:
        """Perform LLM-based claim decomposition.

        Args:
            text: Text to decompose.
            context: Optional context for reference resolution.

        Returns:
            DecompositionResult from LLM output.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        # Choose prompt based on whether context is provided
        if context:
            prompt = DECOMPOSITION_WITH_CONTEXT_PROMPT.format(
                text=text, context=context
            )
        else:
            prompt = DECOMPOSITION_PROMPT.format(text=text)

        response = self._model.generate_structured(
            prompt=prompt,
            schema=DECOMPOSITION_SCHEMA,
            system=SYSTEM_MESSAGE,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return self._parse_llm_response(response, text)

    def _parse_llm_response(
        self,
        response: dict,
        original_text: str,
    ) -> DecompositionResult:
        """Parse the structured LLM response into a DecompositionResult.

        Args:
            response: Parsed JSON dict from the LLM.
            original_text: The original text that was decomposed.

        Returns:
            DecompositionResult with typed Claim objects.

        Raises:
            ValueError: If the response structure is invalid.
        """
        raw_claims = response.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError(
                f"Expected 'claims' to be a list, got {type(raw_claims)}"
            )

        claims: List[Claim] = []
        for i, raw in enumerate(raw_claims):
            if isinstance(raw, str):
                # Handle case where LLM returns plain strings
                claim_text = raw
                claim_type = ClaimType.UNKNOWN
                source_sentence = None
            elif isinstance(raw, dict):
                claim_text = raw.get("claim", "")
                claim_type = self._parse_claim_type(raw.get("type", "unknown"))
                source_sentence = raw.get("source_sentence")
            else:
                logger.warning("Skipping unexpected claim format: %r", raw)
                continue

            if not claim_text.strip():
                continue

            claims.append(
                Claim(
                    text=claim_text.strip(),
                    index=i,
                    claim_type=claim_type,
                    source_sentence=source_sentence,
                )
            )

        return DecompositionResult(
            claims=claims,
            original_text=original_text,
            method="llm",
            metadata={"model": self.model_name},
        )

    @staticmethod
    def _parse_claim_type(type_str: str) -> ClaimType:
        """Safely parse a claim type string into the enum."""
        try:
            return ClaimType(type_str.lower())
        except ValueError:
            return ClaimType.UNKNOWN

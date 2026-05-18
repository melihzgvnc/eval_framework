"""Data types for claim decomposition.

Defines the structured representations of individual claims and the
overall decomposition result.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ClaimType(str, Enum):
    """Classification of claim types."""

    FACTUAL = "factual"
    OPINION = "opinion"
    CONDITIONAL = "conditional"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """A single atomic claim extracted from a text.

    An atomic claim is a self-contained statement that can be
    independently verified as true or false against a source.
    """

    text: str
    index: int  # Position in the decomposition sequence
    claim_type: ClaimType = ClaimType.FACTUAL
    source_sentence: Optional[str] = None  # Original sentence it came from
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Claim(index={self.index}, text={self.text!r})"


@dataclass
class DecompositionResult:
    """Result of decomposing a text into atomic claims.

    Holds the list of extracted claims along with metadata about the
    decomposition process itself.
    """

    claims: List[Claim]
    original_text: str
    method: str  # "llm" or "heuristic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_claims(self) -> int:
        """Number of claims extracted."""
        return len(self.claims)

    @property
    def claim_texts(self) -> List[str]:
        """Convenience accessor for just the claim text strings."""
        return [c.text for c in self.claims]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "original_text": self.original_text,
            "method": self.method,
            "num_claims": self.num_claims,
            "claims": [
                {
                    "index": c.index,
                    "text": c.text,
                    "claim_type": c.claim_type.value,
                    "source_sentence": c.source_sentence,
                    "metadata": c.metadata,
                }
                for c in self.claims
            ],
            "metadata": self.metadata,
        }

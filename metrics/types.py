"""Data types for metric computation.

Defines the input structures that metrics operate on and the output
structures they produce.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class VerificationLabel(str, Enum):
    """NLI verification outcome for a single claim."""

    SUPPORTED = "supported"        # Entailed by the source
    CONTRADICTED = "contradicted"  # Contradicts the source
    UNVERIFIABLE = "unverifiable"  # Neutral / cannot be verified


class RefusalLabel(str, Enum):
    """Classification of a model response with respect to refusal."""

    REFUSED = "refused"      # Model refused to answer
    ANSWERED = "answered"    # Model provided an answer


@dataclass
class ClaimVerification:
    """Verification result for a single claim against a source.

    Produced by running NLI on each decomposed claim.
    """

    claim_text: str
    label: VerificationLabel
    confidence: float = 0.0  # NLI model confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Aggregated result for a single evaluation sample.

    Bundles the model response, its claim-level verifications, and
    refusal classification together so metrics can be computed over
    a batch of these.
    """

    sample_id: str
    source_text: str                          # Ground-truth / context
    model_response: str                       # Model-generated answer
    claim_verifications: List[ClaimVerification] = field(default_factory=list)
    is_refusal: bool = False                  # Whether the model refused
    should_refuse: bool = False               # Whether refusal was expected
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_claims(self) -> int:
        return len(self.claim_verifications)

    @property
    def supported_claims(self) -> List[ClaimVerification]:
        return [c for c in self.claim_verifications if c.label == VerificationLabel.SUPPORTED]

    @property
    def contradicted_claims(self) -> List[ClaimVerification]:
        return [c for c in self.claim_verifications if c.label == VerificationLabel.CONTRADICTED]

    @property
    def unverifiable_claims(self) -> List[ClaimVerification]:
        return [c for c in self.claim_verifications if c.label == VerificationLabel.UNVERIFIABLE]


@dataclass
class MetricResult:
    """Output of a metric computation.

    Standardized container for any metric value along with supporting
    details and breakdown information.
    """

    name: str
    value: float  # Primary metric value (typically 0.0–1.0)
    count: int    # Number of samples/claims the metric was computed over
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "count": self.count,
            "details": self.details,
            "metadata": self.metadata,
        }

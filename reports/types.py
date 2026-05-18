"""Data types for evaluation reports.

A report is a structured, renderer-agnostic representation of an
evaluation run. The same Report object can be serialized to JSON,
rendered as Markdown, or exported to HTML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SectionKind(str, Enum):
    """Categorical kind for a report section.

    Renderers use this to choose how to display a section
    (e.g., metrics table vs prose vs sample list).
    """

    HEADER = "header"
    METRICS_TABLE = "metrics_table"
    METRIC_DETAIL = "metric_detail"
    SAMPLES = "samples"
    DISTRIBUTION = "distribution"
    CONFIG = "config"
    PROSE = "prose"


@dataclass
class ReportSection:
    """A single, typed chunk of a report.

    Sections are renderer-agnostic. Each renderer decides how to
    display a given ``kind`` based on the data in ``content``.
    """

    title: str
    kind: SectionKind
    content: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "kind": self.kind.value,
            "description": self.description,
            "content": self.content,
        }


@dataclass
class ReportMetadata:
    """Metadata describing the evaluation run.

    Captures everything needed to reproduce or contextualize the
    report: model, dataset, evaluator config, timestamp.
    """

    run_id: str
    model_name: str
    evaluator_type: str  # "nli", "llm_judge", "composite", etc.
    dataset_name: Optional[str] = None
    num_samples: int = 0
    num_claims: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    framework_version: str = "0.1.0"
    config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "evaluator_type": self.evaluator_type,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "num_claims": self.num_claims,
            "timestamp": self.timestamp,
            "framework_version": self.framework_version,
            "config": self.config,
            "extra": self.extra,
        }


@dataclass
class Report:
    """Structured evaluation report.

    The ``Report`` is the single source of truth for all renderers.
    Build once, render many times in different formats.
    """

    metadata: ReportMetadata
    sections: List[ReportSection] = field(default_factory=list)

    def add_section(self, section: ReportSection) -> "Report":
        """Append a section. Returns self for chaining."""
        self.sections.append(section)
        return self

    def get_section(self, kind: SectionKind) -> Optional[ReportSection]:
        """Find the first section matching the given kind."""
        for section in self.sections:
            if section.kind == kind:
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
        }

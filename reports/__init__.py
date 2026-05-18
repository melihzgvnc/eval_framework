"""Evaluation report generation.

Build structured reports from metrics and sample results, then render
them to multiple formats (JSON, Markdown, HTML).

Architecture:

- ``Report`` / ``ReportSection`` / ``ReportMetadata`` — structured data
- ``ReportBuilder`` — assembles Reports from metrics + samples
- ``BaseRenderer`` and subclasses — render Reports to formats
- ``RunStorage`` — standardized on-disk layout
- ``ComparisonReportBuilder`` — multi-run comparisons (v2)

Typical usage::

    from eval_framework.reports import (
        ReportBuilder, ReportMetadata, RunStorage,
    )

    builder = ReportBuilder(top_n_failures=10)
    report = builder.build(metrics=metrics, samples=samples,
                           metadata=ReportMetadata(
                               run_id="my_run",
                               model_name="gpt-4o",
                               evaluator_type="llm_judge",
                           ))

    storage = RunStorage(base_dir="reports/")
    storage.save(report, samples=samples)
"""

from .builder import ReportBuilder
from .comparison import ComparisonReportBuilder, RunSummary
from .renderers import BaseRenderer, JSONRenderer, MarkdownRenderer
from .storage import RunStorage
from .types import (
    Report,
    ReportMetadata,
    ReportSection,
    SectionKind,
)

__all__ = [
    # Types
    "Report",
    "ReportMetadata",
    "ReportSection",
    "SectionKind",
    "RunSummary",
    # Builders
    "ReportBuilder",
    "ComparisonReportBuilder",
    # Renderers
    "BaseRenderer",
    "JSONRenderer",
    "MarkdownRenderer",
    # Storage
    "RunStorage",
]


def __getattr__(name):
    """Lazy import for HTMLRenderer to avoid optional dependency at import time."""
    if name == "HTMLRenderer":
        from .renderers.html_renderer import HTMLRenderer
        return HTMLRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

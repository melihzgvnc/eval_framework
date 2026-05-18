"""Run output storage.

Standardizes the on-disk layout for evaluation runs so reports,
samples, and metadata are co-located and easy to find.

Layout::

    reports/
    └── <run_id>/
        ├── report.json          # Full report (source of truth)
        ├── report.md            # Human-readable summary
        ├── report.html          # (v2) Interactive view
        ├── samples.jsonl        # Per-sample raw data
        └── metadata.json        # Run metadata only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

from eval_framework.metrics.types import SampleResult
from eval_framework.reports.renderers import (
    BaseRenderer,
    JSONRenderer,
    MarkdownRenderer,
)
from eval_framework.reports.types import Report


@dataclass
class RunStorage:
    """Manages on-disk output for a single evaluation run."""

    base_dir: Union[str, Path] = "reports"

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    def run_dir(self, run_id: str) -> Path:
        """Return (and create) the directory for a given run_id."""
        d = self.base_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        report: Report,
        samples: Optional[Iterable[SampleResult]] = None,
        renderers: Optional[List[BaseRenderer]] = None,
    ) -> Path:
        """Save a report and its samples to the standard layout.

        Args:
            report: The Report to persist.
            samples: Optional iterable of SampleResults to write as JSONL.
            renderers: Optional list of renderers. Defaults to JSON + Markdown.

        Returns:
            Path to the run directory.
        """
        run_dir = self.run_dir(report.metadata.run_id)

        # Default to JSON + Markdown
        if renderers is None:
            renderers = [JSONRenderer(), MarkdownRenderer()]

        for renderer in renderers:
            renderer.render_to_file(
                report, run_dir / f"report{renderer.extension}"
            )

        # Standalone metadata file for quick lookup
        (run_dir / "metadata.json").write_text(
            json.dumps(report.metadata.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        # Per-sample raw data as JSONL
        if samples is not None:
            self._write_samples_jsonl(run_dir / "samples.jsonl", samples)

        return run_dir

    @staticmethod
    def _write_samples_jsonl(
        path: Path, samples: Iterable[SampleResult]
    ) -> None:
        """Write samples as JSONL — one JSON object per line."""
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                obj = {
                    "sample_id": sample.sample_id,
                    "source_text": sample.source_text,
                    "model_response": sample.model_response,
                    "is_refusal": sample.is_refusal,
                    "should_refuse": sample.should_refuse,
                    "metadata": sample.metadata,
                    "claim_verifications": [
                        {
                            "claim_text": cv.claim_text,
                            "label": cv.label.value,
                            "confidence": cv.confidence,
                            "metadata": cv.metadata,
                        }
                        for cv in sample.claim_verifications
                    ],
                }
                f.write(json.dumps(obj, default=str) + "\n")

    def load_report(self, run_id: str) -> dict:
        """Load a saved report.json as a dict."""
        path = self.run_dir(run_id) / "report.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self) -> List[str]:
        """List all run IDs present in the base directory."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "report.json").exists()
        )

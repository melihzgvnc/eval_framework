"""Multi-run comparison reports.

Compare two or more evaluation runs side-by-side. Useful for:
- A/B comparisons between models on the same dataset
- Longitudinal tracking of the same model over time
- Comparing evaluator strategies (NLI vs LLM Judge) on the same data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eval_framework.reports.types import (
    Report,
    ReportMetadata,
    ReportSection,
    SectionKind,
)


@dataclass
class RunSummary:
    """Lightweight summary of a single run, used in comparisons."""

    run_id: str
    model_name: str
    evaluator_type: str
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)
    num_samples: int = 0
    num_claims: int = 0

    @classmethod
    def from_report_dict(cls, report_dict: Dict[str, Any]) -> "RunSummary":
        """Build a summary from a saved report.json dict."""
        meta = report_dict.get("metadata", {})
        sections = report_dict.get("sections", [])

        # Find the metrics_table section
        metrics: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for s in sections:
            if s.get("kind") == SectionKind.METRICS_TABLE.value:
                for row in s.get("content", {}).get("rows", []):
                    metrics[row["metric"]] = row["value"]
                    counts[row["metric"]] = row.get("count", 0)
                break

        return cls(
            run_id=meta.get("run_id", "unknown"),
            model_name=meta.get("model_name", "unknown"),
            evaluator_type=meta.get("evaluator_type", "unknown"),
            timestamp=meta.get("timestamp", ""),
            metrics=metrics,
            metric_counts=counts,
            num_samples=meta.get("num_samples", 0),
            num_claims=meta.get("num_claims", 0),
        )

    @classmethod
    def from_report(cls, report: Report) -> "RunSummary":
        """Build a summary directly from a Report object."""
        return cls.from_report_dict(report.to_dict())


@dataclass
class ComparisonReportBuilder:
    """Builds comparison reports across multiple runs.

    The output is itself a ``Report`` (so existing renderers work),
    just with comparison-flavored sections.
    """

    baseline_run_id: Optional[str] = None  # Run to use as the delta reference

    def build(
        self,
        runs: List[RunSummary],
        title: str = "Comparison Report",
        description: Optional[str] = None,
    ) -> Report:
        """Build a comparison Report from multiple run summaries.

        Args:
            runs: List of RunSummary objects to compare.
            title: Title for the comparison report.
            description: Optional description.

        Returns:
            Report with comparison-specific sections.
        """
        if not runs:
            raise ValueError("Need at least one run to build a comparison.")

        baseline = self._pick_baseline(runs)

        from datetime import datetime, timezone
        metadata = ReportMetadata(
            run_id=f"comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            model_name=f"{len(runs)} models",
            evaluator_type="comparison",
            num_samples=sum(r.num_samples for r in runs),
            num_claims=sum(r.num_claims for r in runs),
            extra={
                "compared_runs": [r.run_id for r in runs],
                "baseline": baseline.run_id,
            },
        )

        report = Report(metadata=metadata)
        report.add_section(self._build_header(runs, title, description))
        report.add_section(self._build_runs_table(runs))
        report.add_section(self._build_metrics_comparison(runs, baseline))
        report.add_section(self._build_winners_section(runs))
        return report

    def build_from_files(
        self,
        report_paths: List[Union[str, Path]],
        title: str = "Comparison Report",
    ) -> Report:
        """Convenience: load report.json files directly and build a comparison."""
        runs: List[RunSummary] = []
        for path in report_paths:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            runs.append(RunSummary.from_report_dict(data))
        return self.build(runs, title=title)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _pick_baseline(self, runs: List[RunSummary]) -> RunSummary:
        if self.baseline_run_id:
            for r in runs:
                if r.run_id == self.baseline_run_id:
                    return r
        return runs[0]

    def _build_header(
        self,
        runs: List[RunSummary],
        title: str,
        description: Optional[str],
    ) -> ReportSection:
        return ReportSection(
            title=title,
            kind=SectionKind.HEADER,
            description=description,
            content={
                "run_id": f"compare_{len(runs)}_runs",
                "model_name": f"{len(runs)} runs",
                "evaluator_type": "comparison",
                "num_samples": sum(r.num_samples for r in runs),
                "num_claims": sum(r.num_claims for r in runs),
                "timestamp": "",
                "dataset_name": None,
            },
        )

    def _build_runs_table(self, runs: List[RunSummary]) -> ReportSection:
        rows = [
            {
                "run_id": r.run_id,
                "model": r.model_name,
                "evaluator": r.evaluator_type,
                "timestamp": r.timestamp,
                "samples": r.num_samples,
                "claims": r.num_claims,
            }
            for r in runs
        ]
        return ReportSection(
            title="Runs Compared",
            kind=SectionKind.METRIC_DETAIL,
            content={"value": len(runs), "count": len(runs), "details": {"runs": rows}},
        )

    def _build_metrics_comparison(
        self,
        runs: List[RunSummary],
        baseline: RunSummary,
    ) -> ReportSection:
        """Per-metric, per-run table with deltas vs baseline."""
        # Collect the full set of metric names across all runs
        all_metrics = sorted(
            {m for r in runs for m in r.metrics}
        )

        rows = []
        for metric_name in all_metrics:
            row: Dict[str, Any] = {"metric": metric_name}
            base_val = baseline.metrics.get(metric_name)
            for r in runs:
                val = r.metrics.get(metric_name)
                row[r.run_id] = val
                if r.run_id != baseline.run_id and val is not None and base_val is not None:
                    row[f"{r.run_id}_delta"] = val - base_val
            rows.append(row)

        return ReportSection(
            title=f"Metrics Comparison (baseline: {baseline.run_id})",
            kind=SectionKind.METRICS_TABLE,
            description=(
                "Each row is one metric. Columns are runs; "
                "delta columns show change vs the baseline run."
            ),
            content={
                "rows": rows,
                "baseline": baseline.run_id,
                "run_ids": [r.run_id for r in runs],
            },
        )

    def _build_winners_section(self, runs: List[RunSummary]) -> ReportSection:
        """For each metric, identify the best and worst run.

        Direction (higher-is-better vs lower-is-better) is metric-specific:
            higher-is-better: groundedness, refusal_precision, refusal_recall
            lower-is-better:  hallucination_rate, over_refusal_rate
        """
        higher_better = {"groundedness", "refusal_precision", "refusal_recall"}
        lower_better = {"hallucination_rate", "over_refusal_rate"}

        all_metrics = sorted({m for r in runs for m in r.metrics})

        winners = []
        for metric_name in all_metrics:
            scored = [
                (r.run_id, r.metrics[metric_name])
                for r in runs
                if metric_name in r.metrics
            ]
            if not scored:
                continue

            if metric_name in higher_better:
                best = max(scored, key=lambda x: x[1])
                worst = min(scored, key=lambda x: x[1])
                direction = "higher-is-better"
            elif metric_name in lower_better:
                best = min(scored, key=lambda x: x[1])
                worst = max(scored, key=lambda x: x[1])
                direction = "lower-is-better"
            else:
                # Unknown direction — report range only
                best = max(scored, key=lambda x: x[1])
                worst = min(scored, key=lambda x: x[1])
                direction = "unknown"

            winners.append({
                "metric": metric_name,
                "direction": direction,
                "best": {"run_id": best[0], "value": best[1]},
                "worst": {"run_id": worst[0], "value": worst[1]},
                "spread": best[1] - worst[1],
            })

        return ReportSection(
            title="Winners by Metric",
            kind=SectionKind.METRIC_DETAIL,
            description=(
                "For each metric, the best and worst run. Direction depends on "
                "whether higher or lower scores are preferred."
            ),
            content={
                "value": len(winners),
                "count": len(winners),
                "details": {"winners": winners},
            },
        )

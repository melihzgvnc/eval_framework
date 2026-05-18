"""Report builder.

Assembles a structured ``Report`` from metric results and sample
results. The builder is renderer-agnostic — it produces data, not
formatting.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from eval_framework.metrics.types import (
    MetricResult,
    SampleResult,
    VerificationLabel,
)
from eval_framework.reports.types import (
    Report,
    ReportMetadata,
    ReportSection,
    SectionKind,
)


@dataclass
class ReportBuilder:
    """Builds structured Report objects from evaluation outputs."""

    top_n_failures: int = 10  # Number of worst samples to include
    include_passing_samples: bool = False
    redact_sample_text: bool = False  # Replace sample text with hashes

    def build(
        self,
        metrics: List[MetricResult],
        samples: List[SampleResult],
        metadata: ReportMetadata,
    ) -> Report:
        """Build a complete report from metrics and samples.

        Args:
            metrics: List of computed metric results.
            samples: List of evaluated sample results.
            metadata: Run metadata.

        Returns:
            Fully assembled Report ready for rendering.
        """
        # Update metadata counts from samples
        metadata.num_samples = len(samples)
        metadata.num_claims = sum(s.num_claims for s in samples)

        report = Report(metadata=metadata)

        report.add_section(self._build_header(metadata))
        report.add_section(self._build_metrics_table(metrics))

        for metric in metrics:
            report.add_section(self._build_metric_detail(metric))

        report.add_section(self._build_failure_samples(samples))
        report.add_section(self._build_confidence_distribution(samples))
        report.add_section(self._build_config(metadata))

        return report

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_header(self, metadata: ReportMetadata) -> ReportSection:
        """Run-level summary header."""
        return ReportSection(
            title="Evaluation Run",
            kind=SectionKind.HEADER,
            content={
                "run_id": metadata.run_id,
                "model_name": metadata.model_name,
                "evaluator_type": metadata.evaluator_type,
                "dataset_name": metadata.dataset_name,
                "timestamp": metadata.timestamp,
                "num_samples": metadata.num_samples,
                "num_claims": metadata.num_claims,
            },
        )

    def _build_metrics_table(self, metrics: List[MetricResult]) -> ReportSection:
        """Headline metrics: name → value, one row per metric."""
        rows = [
            {
                "metric": m.name,
                "value": m.value,
                "count": m.count,
            }
            for m in metrics
        ]
        return ReportSection(
            title="Headline Metrics",
            kind=SectionKind.METRICS_TABLE,
            description="Aggregate scores across all evaluated samples.",
            content={"rows": rows},
        )

    def _build_metric_detail(self, metric: MetricResult) -> ReportSection:
        """Per-metric breakdown using MetricResult.details."""
        return ReportSection(
            title=f"Metric: {metric.name}",
            kind=SectionKind.METRIC_DETAIL,
            content={
                "name": metric.name,
                "value": metric.value,
                "count": metric.count,
                "details": metric.details,
                "metadata": metric.metadata,
            },
        )

    def _build_failure_samples(
        self, samples: List[SampleResult]
    ) -> ReportSection:
        """Top-N worst samples (most contradicted/unverifiable claims)."""
        # Score each sample by its "failure level"
        scored = []
        for sample in samples:
            if sample.num_claims == 0:
                # Refusal-only samples — score by mis-classification
                if sample.is_refusal != sample.should_refuse:
                    scored.append((1.0, sample, "refusal_mismatch"))
                continue

            n_bad = (
                len(sample.contradicted_claims)
                + len(sample.unverifiable_claims)
            )
            failure_rate = n_bad / sample.num_claims
            if failure_rate > 0 or self.include_passing_samples:
                scored.append((failure_rate, sample, "claim_failures"))

        # Sort worst first
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_n_failures]

        return ReportSection(
            title=f"Top {self.top_n_failures} Failure Samples",
            kind=SectionKind.SAMPLES,
            description=(
                "Samples ranked by failure rate. "
                "Each entry shows the source, response, and per-claim verdicts."
            ),
            content={
                "samples": [
                    self._serialize_sample(sample, failure_rate, reason)
                    for failure_rate, sample, reason in top
                ],
                "total_failures": len(scored),
            },
        )

    def _build_confidence_distribution(
        self, samples: List[SampleResult]
    ) -> ReportSection:
        """Histogram of per-claim confidence scores."""
        confidences = [
            cv.confidence
            for s in samples
            for cv in s.claim_verifications
            if cv.confidence is not None
        ]

        if not confidences:
            return ReportSection(
                title="Confidence Distribution",
                kind=SectionKind.DISTRIBUTION,
                content={"histogram": [], "stats": {}},
            )

        # Build a 10-bucket histogram
        buckets = [0] * 10
        for c in confidences:
            idx = min(int(c * 10), 9)
            buckets[idx] += 1

        histogram = [
            {
                "range": f"{i/10:.1f}–{(i+1)/10:.1f}",
                "count": count,
            }
            for i, count in enumerate(buckets)
        ]

        stats = {
            "min": min(confidences),
            "max": max(confidences),
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "stdev": (
                statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            ),
            "count": len(confidences),
        }

        return ReportSection(
            title="Confidence Distribution",
            kind=SectionKind.DISTRIBUTION,
            description="Per-claim confidence scores from the evaluator.",
            content={"histogram": histogram, "stats": stats},
        )

    def _build_config(self, metadata: ReportMetadata) -> ReportSection:
        """Run configuration snapshot for reproducibility."""
        return ReportSection(
            title="Run Configuration",
            kind=SectionKind.CONFIG,
            content={
                "framework_version": metadata.framework_version,
                "config": metadata.config,
                "extra": metadata.extra,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _serialize_sample(
        self,
        sample: SampleResult,
        failure_rate: float,
        reason: str,
    ) -> Dict[str, Any]:
        """Serialize a single sample for inclusion in the report."""
        if self.redact_sample_text:
            source = _hash_text(sample.source_text)
            response = _hash_text(sample.model_response)
            claims = [
                {
                    "claim_text": _hash_text(cv.claim_text),
                    "label": cv.label.value,
                    "confidence": cv.confidence,
                }
                for cv in sample.claim_verifications
            ]
        else:
            source = sample.source_text
            response = sample.model_response
            claims = [
                {
                    "claim_text": cv.claim_text,
                    "label": cv.label.value,
                    "confidence": cv.confidence,
                    "metadata": cv.metadata,
                }
                for cv in sample.claim_verifications
            ]

        return {
            "sample_id": sample.sample_id,
            "failure_rate": failure_rate,
            "failure_reason": reason,
            "source_text": source,
            "model_response": response,
            "is_refusal": sample.is_refusal,
            "should_refuse": sample.should_refuse,
            "claims": claims,
            "supported_count": len(sample.supported_claims),
            "contradicted_count": len(sample.contradicted_claims),
            "unverifiable_count": len(sample.unverifiable_claims),
        }


def _hash_text(text: str) -> str:
    """Hash a string for redaction."""
    import hashlib

    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"<redacted:{digest}>"

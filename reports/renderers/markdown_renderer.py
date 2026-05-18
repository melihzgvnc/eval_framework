"""Markdown renderer.

Produces a human-readable, git-friendly report. Designed to drop
cleanly into GitHub PRs and documentation.
"""

from typing import Any, Dict, List

from eval_framework.reports.renderers.base import BaseRenderer
from eval_framework.reports.types import Report, ReportSection, SectionKind


class MarkdownRenderer(BaseRenderer):
    """Render a Report as Markdown."""

    extension = ".md"

    def __init__(self, max_samples: int = 10, max_claim_text_length: int = 200):
        self.max_samples = max_samples
        self.max_claim_text_length = max_claim_text_length

    def render(self, report: Report) -> str:
        """Render the Report as a Markdown document."""
        parts: List[str] = []

        # Title
        meta = report.metadata
        parts.append(f"# Evaluation Report: {meta.model_name}")
        parts.append("")

        # Render each section
        for section in report.sections:
            parts.append(self._render_section(section))
            parts.append("")

        return "\n".join(parts).rstrip() + "\n"

    # ------------------------------------------------------------------
    # Section dispatch
    # ------------------------------------------------------------------

    def _render_section(self, section: ReportSection) -> str:
        renderers = {
            SectionKind.HEADER: self._render_header,
            SectionKind.METRICS_TABLE: self._render_metrics_table,
            SectionKind.METRIC_DETAIL: self._render_metric_detail,
            SectionKind.SAMPLES: self._render_samples,
            SectionKind.DISTRIBUTION: self._render_distribution,
            SectionKind.CONFIG: self._render_config,
            SectionKind.PROSE: self._render_prose,
        }
        renderer = renderers.get(section.kind, self._render_generic)
        return renderer(section)

    # ------------------------------------------------------------------
    # Per-kind renderers
    # ------------------------------------------------------------------

    def _render_header(self, section: ReportSection) -> str:
        c = section.content
        lines = [
            f"## {section.title}",
            "",
            f"- **Run ID**: `{c.get('run_id', 'N/A')}`",
            f"- **Model**: {c.get('model_name', 'N/A')}",
            f"- **Evaluator**: {c.get('evaluator_type', 'N/A')}",
            f"- **Dataset**: {c.get('dataset_name') or 'N/A'}",
            f"- **Timestamp**: {c.get('timestamp', 'N/A')}",
            f"- **Samples**: {c.get('num_samples', 0)}",
            f"- **Claims**: {c.get('num_claims', 0)}",
        ]
        return "\n".join(lines)

    def _render_metrics_table(self, section: ReportSection) -> str:
        rows = section.content.get("rows", [])
        lines = [f"## {section.title}", ""]
        if section.description:
            lines.append(f"_{section.description}_")
            lines.append("")

        if not rows:
            lines.append("_No metrics computed._")
            return "\n".join(lines)

        # Standard shape: {metric, value, count}
        if all("value" in r and "count" in r for r in rows):
            lines.append("| Metric | Value | Count |")
            lines.append("|--------|-------|-------|")
            for r in rows:
                lines.append(
                    f"| {r['metric']} | {r['value']:.4f} | {r['count']} |"
                )
            return "\n".join(lines)

        # Comparison shape: {metric, <run_id>: value, <run_id>_delta: delta, ...}
        # Build columns dynamically from the first row.
        columns = [k for k in rows[0].keys() if k != "metric"]
        header = "| Metric | " + " | ".join(columns) + " |"
        sep = "|--------|" + "|".join(["-------"] * len(columns)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in rows:
            cells = [r.get("metric", "")]
            for col in columns:
                val = r.get(col)
                if val is None:
                    cells.append("—")
                elif isinstance(val, float):
                    sign = "+" if "_delta" in col and val > 0 else ""
                    cells.append(f"{sign}{val:.4f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _render_metric_detail(self, section: ReportSection) -> str:
        c = section.content
        lines = [
            f"### {section.title}",
            "",
            f"**Value**: {c.get('value', 0.0):.4f} _(over {c.get('count', 0)} items)_",
            "",
        ]

        details = c.get("details", {})
        if details:
            lines.append("**Breakdown**:")
            lines.append("")
            for k, v in details.items():
                if isinstance(v, dict):
                    lines.append(f"- **{k}**: `{v}`")
                else:
                    lines.append(f"- **{k}**: {v}")

        return "\n".join(lines)

    def _render_samples(self, section: ReportSection) -> str:
        samples = section.content.get("samples", [])
        lines = [f"## {section.title}", ""]
        if section.description:
            lines.append(f"_{section.description}_")
            lines.append("")

        if not samples:
            lines.append("_No failure samples to display._")
            return "\n".join(lines)

        for i, s in enumerate(samples[: self.max_samples], start=1):
            lines.append(
                f"### {i}. Sample `{s['sample_id']}` "
                f"(failure rate: {s['failure_rate']:.2f})"
            )
            lines.append("")
            lines.append(
                f"- **Source**: {self._truncate(s.get('source_text', ''))}"
            )
            lines.append(
                f"- **Response**: {self._truncate(s.get('model_response', ''))}"
            )
            lines.append(
                f"- **Refusal**: model={s.get('is_refusal')} expected={s.get('should_refuse')}"
            )
            lines.append(
                f"- **Claim verdicts**: "
                f"{s.get('supported_count', 0)} supported, "
                f"{s.get('contradicted_count', 0)} contradicted, "
                f"{s.get('unverifiable_count', 0)} unverifiable"
            )

            claims = s.get("claims", [])
            if claims:
                lines.append("")
                lines.append("| Claim | Label | Confidence |")
                lines.append("|-------|-------|------------|")
                for claim in claims:
                    text = self._truncate(claim.get("claim_text", ""))
                    label = claim.get("label", "?")
                    conf = claim.get("confidence", 0.0)
                    lines.append(f"| {text} | {label} | {conf:.3f} |")

                # Show reasoning when available (LLM Judge)
                for claim in claims:
                    reasoning = (claim.get("metadata") or {}).get("reasoning")
                    if reasoning:
                        lines.append("")
                        lines.append(
                            f"> **Reasoning** (`{self._truncate(claim['claim_text'], 50)}`): "
                            f"{self._truncate(reasoning, 300)}"
                        )

            lines.append("")
        return "\n".join(lines)

    def _render_distribution(self, section: ReportSection) -> str:
        c = section.content
        lines = [f"## {section.title}", ""]
        if section.description:
            lines.append(f"_{section.description}_")
            lines.append("")

        stats = c.get("stats", {})
        if stats:
            lines.append(
                f"- min: {stats.get('min', 0):.3f} | "
                f"max: {stats.get('max', 0):.3f} | "
                f"mean: {stats.get('mean', 0):.3f} | "
                f"median: {stats.get('median', 0):.3f} | "
                f"stdev: {stats.get('stdev', 0):.3f} | "
                f"n: {stats.get('count', 0)}"
            )
            lines.append("")

        histogram = c.get("histogram", [])
        if histogram:
            max_count = max((h["count"] for h in histogram), default=1) or 1
            for h in histogram:
                bar_len = int((h["count"] / max_count) * 40)
                bar = "█" * bar_len
                lines.append(f"`{h['range']}` {bar} ({h['count']})")

        return "\n".join(lines)

    def _render_config(self, section: ReportSection) -> str:
        c = section.content
        lines = [f"## {section.title}", ""]
        lines.append(f"- **Framework version**: {c.get('framework_version', 'N/A')}")

        config = c.get("config", {})
        if config:
            lines.append("")
            lines.append("**Config**:")
            lines.append("```json")
            import json
            lines.append(json.dumps(config, indent=2, default=str))
            lines.append("```")

        return "\n".join(lines)

    def _render_prose(self, section: ReportSection) -> str:
        body = section.content.get("body", "")
        return f"## {section.title}\n\n{body}"

    def _render_generic(self, section: ReportSection) -> str:
        import json
        return (
            f"## {section.title}\n\n"
            f"```json\n{json.dumps(section.content, indent=2, default=str)}\n```"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str, max_len: int = None) -> str:
        max_len = max_len or self.max_claim_text_length
        if not text:
            return ""
        # Escape pipe and newlines for table safety
        clean = text.replace("\n", " ").replace("|", "\\|")
        if len(clean) > max_len:
            clean = clean[:max_len] + "…"
        return clean

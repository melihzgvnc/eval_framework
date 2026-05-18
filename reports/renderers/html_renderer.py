"""HTML renderer.

Produces a self-contained, interactive HTML report. Uses inline CSS
and vanilla JavaScript (no external dependencies) for portability —
the resulting file works offline.

Charts are rendered as inline SVG bar charts to avoid pulling in
heavy chart libraries. For richer visualizations, swap in matplotlib
or Plotly via subclass.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List

from eval_framework.reports.renderers.base import BaseRenderer
from eval_framework.reports.types import Report, ReportSection, SectionKind


class HTMLRenderer(BaseRenderer):
    """Render a Report as a self-contained HTML document."""

    extension = ".html"

    def __init__(self, max_samples: int = 25, max_claim_text_length: int = 300):
        self.max_samples = max_samples
        self.max_claim_text_length = max_claim_text_length

    def render(self, report: Report) -> str:
        meta = report.metadata
        body_parts = [self._render_section(s) for s in report.sections]
        body = "\n".join(body_parts)

        return _HTML_TEMPLATE.format(
            title=html.escape(f"Eval Report — {meta.model_name}"),
            css=_CSS,
            js=_JS,
            header=self._render_top_summary(report),
            body=body,
        )

    # ------------------------------------------------------------------
    # Top summary banner
    # ------------------------------------------------------------------

    def _render_top_summary(self, report: Report) -> str:
        meta = report.metadata
        return (
            f"<h1>{html.escape(meta.model_name)}</h1>"
            f"<p class='subtitle'>"
            f"<code>{html.escape(meta.run_id)}</code> · "
            f"{html.escape(meta.evaluator_type)} · "
            f"{html.escape(meta.timestamp)}"
            f"</p>"
        )

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
        return renderers.get(section.kind, self._render_generic)(section)

    # ------------------------------------------------------------------
    # Per-kind renderers
    # ------------------------------------------------------------------

    def _render_header(self, section: ReportSection) -> str:
        c = section.content
        rows = [
            ("Run ID", c.get("run_id")),
            ("Model", c.get("model_name")),
            ("Evaluator", c.get("evaluator_type")),
            ("Dataset", c.get("dataset_name") or "—"),
            ("Timestamp", c.get("timestamp")),
            ("Samples", c.get("num_samples", 0)),
            ("Claims", c.get("num_claims", 0)),
        ]
        rows_html = "".join(
            f"<tr><th>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        return self._wrap_section(
            section.title,
            f"<table class='kv'>{rows_html}</table>",
        )

    def _render_metrics_table(self, section: ReportSection) -> str:
        rows = section.content.get("rows", [])
        if not rows:
            return self._wrap_section(
                section.title, "<p class='empty'>No metrics computed.</p>"
            )

        # Standard shape: cards layout
        if all("value" in r and "count" in r for r in rows):
            cards = "".join(
                f"<div class='metric-card'>"
                f"<div class='metric-name'>{html.escape(r['metric'])}</div>"
                f"<div class='metric-value'>{r['value']:.4f}</div>"
                f"<div class='metric-count'>over {r['count']} items</div>"
                f"</div>"
                for r in rows
            )
            return self._wrap_section(
                section.title,
                f"<div class='metric-grid'>{cards}</div>",
                description=section.description,
            )

        # Comparison shape: render as a wide table
        columns = [k for k in rows[0].keys() if k != "metric"]
        thead = (
            "<tr><th>Metric</th>"
            + "".join(f"<th>{html.escape(c)}</th>" for c in columns)
            + "</tr>"
        )
        body_rows = []
        for r in rows:
            cells = [f"<td><strong>{html.escape(str(r.get('metric', '')))}</strong></td>"]
            for col in columns:
                val = r.get(col)
                if val is None:
                    cells.append("<td class='muted'>—</td>")
                elif isinstance(val, float):
                    if "_delta" in col:
                        sign = "+" if val > 0 else ""
                        cls = "delta-pos" if val > 0 else (
                            "delta-neg" if val < 0 else "muted"
                        )
                        cells.append(
                            f"<td class='{cls}'>{sign}{val:.4f}</td>"
                        )
                    else:
                        cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{html.escape(str(val))}</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        table = (
            f"<table class='comparison'><thead>{thead}</thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )
        return self._wrap_section(
            section.title, table, description=section.description
        )

    def _render_metric_detail(self, section: ReportSection) -> str:
        c = section.content
        details = c.get("details", {})
        rows_html = "".join(
            f"<tr><th>{html.escape(str(k))}</th>"
            f"<td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in details.items()
        )
        body = (
            f"<p><strong>Value</strong>: {c.get('value', 0):.4f} "
            f"<span class='muted'>(over {c.get('count', 0)} items)</span></p>"
            f"<table class='kv'>{rows_html}</table>"
        )
        return self._wrap_section(section.title, body, collapsible=True)

    def _render_samples(self, section: ReportSection) -> str:
        samples = section.content.get("samples", [])[: self.max_samples]
        if not samples:
            return self._wrap_section(
                section.title, "<p class='empty'>No failure samples.</p>"
            )

        items = []
        for i, s in enumerate(samples, start=1):
            claims_html = self._render_claims_table(s.get("claims", []))
            badge = self._failure_badge(s.get("failure_rate", 0))
            items.append(
                f"<details class='sample'>"
                f"<summary>"
                f"<span class='sample-index'>#{i}</span> "
                f"<code>{html.escape(s.get('sample_id', ''))}</code> "
                f"{badge}"
                f"</summary>"
                f"<div class='sample-body'>"
                f"<div class='kv-pair'><strong>Source</strong>: "
                f"{self._safe_text(s.get('source_text', ''))}</div>"
                f"<div class='kv-pair'><strong>Response</strong>: "
                f"{self._safe_text(s.get('model_response', ''))}</div>"
                f"<div class='kv-pair'><strong>Refusal</strong>: "
                f"model={s.get('is_refusal')} expected={s.get('should_refuse')}</div>"
                f"{claims_html}"
                f"</div>"
                f"</details>"
            )
        return self._wrap_section(
            section.title,
            "".join(items),
            description=section.description,
        )

    def _render_claims_table(self, claims: List[Dict[str, Any]]) -> str:
        if not claims:
            return ""
        rows = []
        reasonings = []
        for c in claims:
            label = c.get("label", "?")
            text = self._safe_text(c.get("claim_text", ""))
            conf = c.get("confidence", 0.0)
            rows.append(
                f"<tr class='label-{html.escape(label)}'>"
                f"<td>{text}</td>"
                f"<td><span class='label label-{html.escape(label)}'>"
                f"{html.escape(label)}</span></td>"
                f"<td>{conf:.3f}</td>"
                f"</tr>"
            )
            reasoning = (c.get("metadata") or {}).get("reasoning")
            if reasoning:
                reasonings.append(
                    f"<blockquote><em>{self._safe_text(c.get('claim_text', ''))}</em>: "
                    f"{self._safe_text(str(reasoning))}</blockquote>"
                )

        body = (
            f"<table class='claims'>"
            f"<thead><tr><th>Claim</th><th>Label</th><th>Confidence</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        if reasonings:
            body += f"<div class='reasonings'>{''.join(reasonings)}</div>"
        return body

    def _render_distribution(self, section: ReportSection) -> str:
        c = section.content
        stats = c.get("stats", {})
        histogram = c.get("histogram", [])

        stats_html = ""
        if stats:
            stats_html = (
                f"<p class='stats-line'>"
                f"min: {stats.get('min', 0):.3f} · "
                f"max: {stats.get('max', 0):.3f} · "
                f"mean: {stats.get('mean', 0):.3f} · "
                f"median: {stats.get('median', 0):.3f} · "
                f"stdev: {stats.get('stdev', 0):.3f} · "
                f"n: {stats.get('count', 0)}"
                f"</p>"
            )

        chart_html = self._render_bar_chart(histogram) if histogram else ""

        return self._wrap_section(
            section.title,
            stats_html + chart_html,
            description=section.description,
        )

    def _render_config(self, section: ReportSection) -> str:
        c = section.content
        body = (
            f"<p><strong>Framework version</strong>: "
            f"<code>{html.escape(str(c.get('framework_version', 'N/A')))}</code></p>"
            f"<pre><code>{html.escape(json.dumps(c.get('config', {}), indent=2, default=str))}</code></pre>"
        )
        return self._wrap_section(section.title, body, collapsible=True)

    def _render_prose(self, section: ReportSection) -> str:
        body = section.content.get("body", "")
        return self._wrap_section(
            section.title, f"<p>{html.escape(body)}</p>"
        )

    def _render_generic(self, section: ReportSection) -> str:
        body = (
            f"<pre><code>{html.escape(json.dumps(section.content, indent=2, default=str))}"
            f"</code></pre>"
        )
        return self._wrap_section(section.title, body, collapsible=True)

    # ------------------------------------------------------------------
    # Inline SVG bar chart
    # ------------------------------------------------------------------

    def _render_bar_chart(self, histogram: List[Dict[str, Any]]) -> str:
        """Inline SVG bar chart — no external dependencies."""
        if not histogram:
            return ""
        max_count = max((h["count"] for h in histogram), default=1) or 1
        n = len(histogram)
        width = 600
        height = 200
        bar_w = width / n - 4

        bars = []
        for i, h in enumerate(histogram):
            x = i * (width / n) + 2
            bar_h = (h["count"] / max_count) * (height - 30)
            y = height - 20 - bar_h
            bars.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' "
                f"width='{bar_w:.1f}' height='{bar_h:.1f}' "
                f"class='bar'/>"
                f"<text x='{x + bar_w/2:.1f}' y='{height - 4:.1f}' "
                f"class='bar-label'>{html.escape(h['range'])}</text>"
                f"<text x='{x + bar_w/2:.1f}' y='{y - 2:.1f}' "
                f"class='bar-count'>{h['count']}</text>"
            )

        return (
            f"<svg class='chart' viewBox='0 0 {width} {height}' "
            f"xmlns='http://www.w3.org/2000/svg'>{''.join(bars)}</svg>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_section(
        self,
        title: str,
        body: str,
        description: str = None,
        collapsible: bool = False,
    ) -> str:
        desc = (
            f"<p class='description'>{html.escape(description)}</p>"
            if description else ""
        )
        if collapsible:
            return (
                f"<section><details>"
                f"<summary><h2>{html.escape(title)}</h2></summary>"
                f"{desc}{body}"
                f"</details></section>"
            )
        return (
            f"<section>"
            f"<h2>{html.escape(title)}</h2>"
            f"{desc}{body}"
            f"</section>"
        )

    def _safe_text(self, text: str) -> str:
        if not text:
            return ""
        if len(text) > self.max_claim_text_length:
            text = text[: self.max_claim_text_length] + "…"
        return html.escape(text)

    def _failure_badge(self, rate: float) -> str:
        if rate >= 0.66:
            cls = "badge-high"
        elif rate >= 0.33:
            cls = "badge-mid"
        else:
            cls = "badge-low"
        return f"<span class='badge {cls}'>{rate:.2f}</span>"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<header class="top">{header}</header>
<main>
{body}
</main>
<script>{js}</script>
</body>
</html>
"""


_CSS = """
:root {
  --bg: #ffffff;
  --fg: #1f2328;
  --muted: #57606a;
  --border: #d0d7de;
  --accent: #0969da;
  --supported: #1a7f37;
  --contradicted: #cf222e;
  --unverifiable: #9a6700;
  --code-bg: #f6f8fa;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       color: var(--fg); background: var(--bg); margin: 0; line-height: 1.5; }
.top { padding: 24px 32px; border-bottom: 1px solid var(--border); background: var(--code-bg); }
.top h1 { margin: 0; font-size: 22px; }
.subtitle { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 32px; }
section { margin-bottom: 32px; }
h2 { font-size: 18px; border-bottom: 1px solid var(--border);
     padding-bottom: 6px; margin: 0 0 12px; display: inline; }
.description { color: var(--muted); font-style: italic; margin: 8px 0 12px; }
table.kv { border-collapse: collapse; }
table.kv th, table.kv td {
  text-align: left; padding: 6px 12px; border-bottom: 1px solid var(--border);
  vertical-align: top; font-size: 13px;
}
table.kv th { color: var(--muted); font-weight: 500; min-width: 140px; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 12px; }
.metric-card { border: 1px solid var(--border); border-radius: 6px;
               padding: 14px 16px; background: var(--code-bg); }
.metric-name { color: var(--muted); font-size: 12px; text-transform: uppercase;
               letter-spacing: 0.5px; }
.metric-value { font-size: 28px; font-weight: 600; margin: 4px 0 2px; }
.metric-count { color: var(--muted); font-size: 12px; }
.muted { color: var(--muted); }
table.claims { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
table.claims th, table.claims td {
  text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border);
}
.label { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 500; text-transform: uppercase; }
.label-supported { color: var(--supported); background: #dafbe1; }
.label-contradicted { color: var(--contradicted); background: #ffebe9; }
.label-unverifiable { color: var(--unverifiable); background: #fff8c5; }
.sample { border: 1px solid var(--border); border-radius: 6px;
          margin-bottom: 8px; padding: 10px 14px; }
.sample summary { cursor: pointer; font-weight: 500; list-style: none; }
.sample summary::-webkit-details-marker { display: none; }
.sample-index { color: var(--muted); margin-right: 8px; }
.sample-body { margin-top: 12px; padding-left: 12px;
               border-left: 2px solid var(--border); }
.kv-pair { margin: 6px 0; font-size: 13px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; margin-left: 8px; }
.badge-high { background: #ffebe9; color: var(--contradicted); }
.badge-mid { background: #fff8c5; color: var(--unverifiable); }
.badge-low { background: #dafbe1; color: var(--supported); }
.reasonings blockquote { margin: 8px 0; padding: 6px 12px;
                         border-left: 3px solid var(--accent);
                         background: var(--code-bg); font-size: 13px; }
pre { background: var(--code-bg); padding: 12px; border-radius: 6px;
      overflow-x: auto; font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, monospace; }
.empty { color: var(--muted); font-style: italic; }
.stats-line { font-family: ui-monospace, monospace; font-size: 12px;
              color: var(--muted); }
.chart { width: 100%; max-width: 600px; height: auto; margin-top: 8px; }
.chart .bar { fill: var(--accent); }
.chart .bar-label, .chart .bar-count {
  font-size: 10px; fill: var(--muted); text-anchor: middle;
  font-family: ui-monospace, monospace;
}
table.comparison { width: 100%; border-collapse: collapse; font-size: 13px; }
table.comparison th, table.comparison td {
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border);
}
table.comparison th { color: var(--muted); font-weight: 500; }
.delta-pos { color: var(--accent); font-weight: 500; }
.delta-neg { color: var(--accent); font-weight: 500; }
details summary h2 { cursor: pointer; }
"""


_JS = """
// Toggle all collapsible sections via 'e' key
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'e') {
    document.querySelectorAll('details').forEach((d) => d.open = !d.open);
  }
});
"""

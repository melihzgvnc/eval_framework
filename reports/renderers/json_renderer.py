"""JSON renderer.

Produces a complete, machine-readable serialization of the Report.
This is the source-of-truth format and should always be emitted.
"""

import json

from eval_framework.reports.renderers.base import BaseRenderer
from eval_framework.reports.types import Report


class JSONRenderer(BaseRenderer):
    """Render a Report as JSON."""

    extension = ".json"

    def __init__(self, indent: int = 2, sort_keys: bool = False):
        self.indent = indent
        self.sort_keys = sort_keys

    def render(self, report: Report) -> str:
        """Serialize the Report to a JSON string."""
        return json.dumps(
            report.to_dict(),
            indent=self.indent,
            sort_keys=self.sort_keys,
            default=str,  # fallback for non-JSON-native types
        )

"""JSONL dataset loader.

Reads files where each line is a valid JSON object. This is the most
common format for ML evaluation datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_framework.datasets.loaders.base import BaseLoader


class JSONLLoader(BaseLoader):
    """Load datasets from JSONL (JSON Lines) files."""

    format_name = "jsonl"

    def _parse_file(
        self, path: Path
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse a JSONL file.

        Skips blank lines. Raises on malformed JSON lines with context
        about which line failed.
        """
        rows: List[Dict[str, Any]] = []
        columns_set: Dict[str, None] = {}  # ordered set via dict

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {path}: {exc}"
                    ) from exc

                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Line {line_num} of {path} is not a JSON object "
                        f"(got {type(obj).__name__})"
                    )

                rows.append(obj)
                for key in obj:
                    columns_set[key] = None

        columns = list(columns_set.keys())
        return rows, columns

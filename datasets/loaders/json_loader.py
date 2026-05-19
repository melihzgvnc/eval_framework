"""JSON dataset loader.

Reads a single JSON file containing a top-level array of objects,
or an object with a configurable key pointing to the array.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_framework.datasets.loaders.base import BaseLoader


class JSONLoader(BaseLoader):
    """Load datasets from a single JSON file."""

    format_name = "json"

    def __init__(self, data_key: Optional[str] = None, **kwargs):
        """Initialize the JSON loader.

        Args:
            data_key: If the JSON root is an object, this key points to
                the array of samples. Common values: "data", "samples",
                "examples". If None, the root must be an array, or the
                loader will try common keys automatically.
            **kwargs: Passed to BaseLoader.
        """
        super().__init__(**kwargs)
        self.data_key = data_key

    def _parse_file(
        self, path: Path
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse a JSON file."""
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        rows = self._extract_rows(data, path)

        # Collect columns
        columns_set: Dict[str, None] = {}
        for row in rows:
            for key in row:
                columns_set[key] = None

        return rows, list(columns_set.keys())

    def _extract_rows(
        self, data: Any, path: Path
    ) -> List[Dict[str, Any]]:
        """Extract the list of row dicts from the parsed JSON."""
        # Direct array at root
        if isinstance(data, list):
            return self._validate_rows(data, path)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON array or object at root of {path}, "
                f"got {type(data).__name__}"
            )

        # Explicit data_key
        if self.data_key:
            if self.data_key not in data:
                raise ValueError(
                    f"Key '{self.data_key}' not found in {path}. "
                    f"Available keys: {sorted(data.keys())}"
                )
            return self._validate_rows(data[self.data_key], path)

        # Auto-detect common keys
        common_keys = ["data", "samples", "examples", "items", "rows", "records"]
        for key in common_keys:
            if key in data and isinstance(data[key], list):
                return self._validate_rows(data[key], path)

        # Last resort: if there's exactly one key with a list value, use it
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if len(list_keys) == 1:
            return self._validate_rows(data[list_keys[0]], path)

        raise ValueError(
            f"Cannot find sample array in {path}. "
            f"Set data_key to one of: {sorted(data.keys())}"
        )

    @staticmethod
    def _validate_rows(
        rows: Any, path: Path
    ) -> List[Dict[str, Any]]:
        """Ensure rows is a list of dicts."""
        if not isinstance(rows, list):
            raise ValueError(
                f"Expected array of objects in {path}, got {type(rows).__name__}"
            )
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Item {i} in {path} is not a JSON object "
                    f"(got {type(row).__name__})"
                )
        return rows

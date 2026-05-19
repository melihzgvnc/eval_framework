"""CSV/TSV dataset loader.

Reads tabular data using Python's stdlib csv module. Supports both
comma-separated and tab-separated formats.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_framework.datasets.loaders.base import BaseLoader


class CSVLoader(BaseLoader):
    """Load datasets from CSV or TSV files."""

    format_name = "csv"

    def __init__(
        self,
        delimiter: Optional[str] = None,
        encoding: str = "utf-8",
        **kwargs,
    ):
        """Initialize the CSV loader.

        Args:
            delimiter: Column delimiter. If None, auto-detected from
                file extension (.tsv → tab, else comma).
            encoding: File encoding.
            **kwargs: Passed to BaseLoader.
        """
        super().__init__(**kwargs)
        self.delimiter = delimiter
        self.encoding = encoding

    def _parse_file(
        self, path: Path
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse a CSV/TSV file."""
        delimiter = self.delimiter
        if delimiter is None:
            delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","

        rows: List[Dict[str, Any]] = []

        with path.open("r", encoding=self.encoding, newline="") as f:
            # Sniff for header
            reader = csv.DictReader(f, delimiter=delimiter)
            columns = reader.fieldnames or []

            for line_num, row in enumerate(reader, start=2):
                # csv.DictReader can produce None keys for extra columns
                cleaned = {
                    k: v for k, v in row.items()
                    if k is not None and v is not None
                }
                rows.append(cleaned)

        if not columns:
            raise ValueError(f"No header row found in {path}")

        return rows, list(columns)

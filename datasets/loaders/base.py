"""Base loader abstraction.

All format-specific loaders inherit from BaseLoader. The base class
handles column mapping, validation, and the common load interface.
Subclasses only implement raw row parsing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from eval_framework.datasets.schema import ColumnMapping
from eval_framework.datasets.types import DatasetInfo, EvalSample
from eval_framework.datasets.validation import ValidationResult, validate_samples


class BaseLoader(ABC):
    """Abstract base for dataset loaders.

    Subclasses implement ``_parse_file`` to return raw rows and column
    names. The base class handles mapping, validation, and metadata.
    """

    #: File format identifier (e.g., "jsonl", "json", "csv").
    format_name: str = ""

    def __init__(
        self,
        column_mapping: Optional[ColumnMapping] = None,
        auto_detect_columns: bool = True,
        validate: bool = True,
    ):
        """Initialize the loader.

        Args:
            column_mapping: Explicit column mapping. If None and
                ``auto_detect_columns`` is True, mapping is inferred.
            auto_detect_columns: Whether to auto-detect column mapping
                when no explicit mapping is provided.
            validate: Whether to validate samples after loading.
        """
        self.column_mapping = column_mapping
        self.auto_detect_columns = auto_detect_columns
        self.validate = validate

    def load(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
    ) -> Tuple[List[EvalSample], DatasetInfo, Optional[ValidationResult]]:
        """Load a dataset from a file.

        Args:
            path: Path to the dataset file.
            name: Optional dataset name (defaults to filename stem).

        Returns:
            Tuple of (samples, dataset_info, validation_result).
            validation_result is None if validation is disabled.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file cannot be parsed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        dataset_name = name or path.stem

        # Parse raw rows
        rows, columns = self._parse_file(path)

        # Resolve column mapping
        mapping = self.column_mapping
        if mapping is None and self.auto_detect_columns:
            mapping = ColumnMapping.auto_detect(columns)
        elif mapping is None:
            mapping = ColumnMapping()

        # Map rows to EvalSamples
        samples = [
            EvalSample(**mapping.resolve(row, i))
            for i, row in enumerate(rows)
        ]

        # Build dataset info
        info = DatasetInfo(
            name=dataset_name,
            path=str(path),
            format=self.format_name,
            num_samples=len(samples),
            columns=columns,
        )

        # Validate
        validation = None
        if self.validate:
            validation = validate_samples(samples)

        return samples, info, validation

    @abstractmethod
    def _parse_file(
        self, path: Path
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Parse the file into raw rows and column names.

        Args:
            path: Path to the file.

        Returns:
            Tuple of (list_of_row_dicts, list_of_column_names).

        Raises:
            ValueError: If the file cannot be parsed.
        """
        pass

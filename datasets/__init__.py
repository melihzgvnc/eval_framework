"""Dataset loading and preprocessing.

Provides loaders for common file formats (JSONL, JSON, CSV/TSV) with
automatic column mapping and validation. Datasets are expected to
contain pre-generated model outputs for evaluation.

Typical usage::

    from eval_framework.datasets import load_dataset

    samples, info, validation = load_dataset("data/eval_set.jsonl")
    if not validation.is_valid:
        print(validation.summary())
    else:
        # samples is List[EvalSample], ready for evaluation
        ...

Custom column mapping::

    from eval_framework.datasets import load_dataset, ColumnMapping

    mapping = ColumnMapping(
        source_text="ground_truth",
        model_response="llm_output",
        query="user_prompt",
    )
    samples, info, _ = load_dataset("data/custom.csv", column_mapping=mapping)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .loaders import BaseLoader, CSVLoader, JSONLoader, JSONLLoader
from .schema import ColumnMapping
from .types import DatasetInfo, EvalSample
from .validation import ValidationError, ValidationResult, validate_samples


def load_dataset(
    path: Union[str, Path],
    column_mapping: Optional[ColumnMapping] = None,
    name: Optional[str] = None,
    validate: bool = True,
    data_key: Optional[str] = None,
) -> Tuple[List[EvalSample], DatasetInfo, Optional[ValidationResult]]:
    """Load a dataset with automatic format detection.

    Infers the file format from the extension and delegates to the
    appropriate loader.

    Args:
        path: Path to the dataset file.
        column_mapping: Explicit column mapping. If None, columns are
            auto-detected from common naming conventions.
        name: Optional dataset name (defaults to filename stem).
        validate: Whether to validate samples after loading.
        data_key: For JSON files, the key containing the sample array.

    Returns:
        Tuple of (samples, dataset_info, validation_result).
        validation_result is None if validation is disabled.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the format is unsupported or the file is malformed.
    """
    path = Path(path)
    ext = path.suffix.lower()

    loader: BaseLoader
    if ext == ".jsonl":
        loader = JSONLLoader(column_mapping=column_mapping, validate=validate)
    elif ext == ".json":
        loader = JSONLoader(
            column_mapping=column_mapping, validate=validate, data_key=data_key
        )
    elif ext in (".csv", ".tsv", ".tab"):
        loader = CSVLoader(column_mapping=column_mapping, validate=validate)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: .jsonl, .json, .csv, .tsv"
        )

    return loader.load(path, name=name)


__all__ = [
    # Convenience function
    "load_dataset",
    # Types
    "EvalSample",
    "DatasetInfo",
    "ColumnMapping",
    # Validation
    "validate_samples",
    "ValidationResult",
    "ValidationError",
    # Loaders
    "BaseLoader",
    "JSONLLoader",
    "JSONLoader",
    "CSVLoader",
]

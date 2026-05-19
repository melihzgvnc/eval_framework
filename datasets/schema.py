"""Column mapping and field resolution.

Translates arbitrary dataset column names into the framework's
canonical field names. Supports explicit mapping and auto-detection
of common naming conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# Common aliases for each canonical field. Used by auto-detection.
_SOURCE_ALIASES: Set[str] = {
    "source_text", "source", "context", "reference", "ground_truth",
    "groundtruth", "premise", "document", "passage", "evidence",
    "gold", "gold_text", "expected",
}

_RESPONSE_ALIASES: Set[str] = {
    "model_response", "response", "answer", "output", "generation",
    "generated", "prediction", "model_output", "completion",
    "hypothesis", "text",
}

_QUERY_ALIASES: Set[str] = {
    "query", "question", "prompt", "input", "instruction", "request",
    "user_input", "user_query",
}

_ID_ALIASES: Set[str] = {
    "sample_id", "id", "idx", "index", "example_id", "uid", "uuid",
    "row_id", "item_id",
}

_SHOULD_REFUSE_ALIASES: Set[str] = {
    "should_refuse", "refuse", "expected_refusal", "is_harmful",
    "is_unsafe", "unsafe", "harmful", "refusal_expected",
}


@dataclass
class ColumnMapping:
    """Maps dataset column names to framework field names.

    Provide explicit mappings for non-standard column names, or rely
    on ``auto_detect()`` to infer from common aliases.
    """

    sample_id: Optional[str] = None
    source_text: Optional[str] = None
    model_response: Optional[str] = None
    query: Optional[str] = None
    should_refuse: Optional[str] = None

    # Columns to include as metadata (all unmapped columns by default)
    metadata_columns: Optional[List[str]] = None

    def resolve(self, row: Dict[str, Any], row_index: int) -> Dict[str, Any]:
        """Apply the mapping to a single raw row.

        Args:
            row: Raw dictionary from the data file.
            row_index: Positional index (used as fallback sample_id).

        Returns:
            Dictionary with canonical field names ready for EvalSample.
        """
        resolved: Dict[str, Any] = {}

        # Required fields
        resolved["source_text"] = self._get_field(row, self.source_text, "")
        resolved["model_response"] = self._get_field(row, self.model_response, "")

        # Optional fields
        resolved["sample_id"] = str(
            self._get_field(row, self.sample_id, str(row_index))
        )
        resolved["query"] = self._get_field(row, self.query, None)

        # Boolean coercion for should_refuse
        raw_refuse = self._get_field(row, self.should_refuse, False)
        resolved["should_refuse"] = _coerce_bool(raw_refuse)

        # Metadata: everything not already mapped
        mapped_cols = {
            self.sample_id, self.source_text, self.model_response,
            self.query, self.should_refuse,
        }
        mapped_cols.discard(None)

        if self.metadata_columns is not None:
            meta_keys = self.metadata_columns
        else:
            meta_keys = [k for k in row if k not in mapped_cols]

        resolved["metadata"] = {k: row[k] for k in meta_keys if k in row}

        return resolved

    @classmethod
    def auto_detect(cls, columns: List[str]) -> "ColumnMapping":
        """Infer a ColumnMapping from a list of column names.

        Matches columns against known aliases for each field. Falls back
        to None (which means the field won't be populated) if no match.

        Args:
            columns: List of column names from the dataset.

        Returns:
            ColumnMapping with detected field assignments.
        """
        col_set = set(columns)
        col_lower_map = {c.lower(): c for c in columns}

        def _find(aliases: Set[str]) -> Optional[str]:
            # Exact match first
            for alias in aliases:
                if alias in col_set:
                    return alias
            # Case-insensitive fallback
            for alias in aliases:
                if alias.lower() in col_lower_map:
                    return col_lower_map[alias.lower()]
            return None

        return cls(
            sample_id=_find(_ID_ALIASES),
            source_text=_find(_SOURCE_ALIASES),
            model_response=_find(_RESPONSE_ALIASES),
            query=_find(_QUERY_ALIASES),
            should_refuse=_find(_SHOULD_REFUSE_ALIASES),
        )

    @staticmethod
    def _get_field(row: Dict[str, Any], key: Optional[str], default: Any) -> Any:
        if key is None:
            return default
        return row.get(key, default)


def _coerce_bool(value: Any) -> bool:
    """Coerce various representations to bool.

    Handles: True/False, 1/0, "true"/"false", "yes"/"no", "1"/"0".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "t", "y"}
    return bool(value)

"""Cache key generation.

Produces deterministic, namespaced keys from arbitrary structured
inputs. Inputs are JSON-canonicalized (sorted keys, no whitespace)
before hashing so semantically identical inputs always produce the
same key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


# Length of the hex digest portion of the key. 32 chars = 128 bits is
# more than sufficient to avoid collisions in practice while keeping
# keys reasonable for SQLite indexing.
_DIGEST_LENGTH = 32


def make_key(namespace: str, **inputs: Any) -> str:
    """Build a deterministic cache key from a namespace and inputs.

    Args:
        namespace: Logical category, e.g. ``"claims"``, ``"nli"``, ``"judge"``.
            Becomes the key prefix.
        **inputs: Arbitrary keyword inputs that affect the cached value.
            Must be JSON-serializable. Any change to any input produces
            a different key.

    Returns:
        Key in the form ``"<namespace>:<sha256_prefix>"``.

    Raises:
        ValueError: If namespace is empty or inputs are not serializable.
    """
    if not namespace:
        raise ValueError("namespace must be a non-empty string")

    try:
        canonical = json.dumps(
            inputs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cache inputs must be JSON-serializable: {exc}") from exc

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{namespace}:{digest}"


def parse_namespace(key: str) -> str:
    """Extract the namespace prefix from a key.

    Args:
        key: A key produced by ``make_key``.

    Returns:
        The namespace portion (everything before the first colon).
    """
    if ":" not in key:
        return ""
    return key.split(":", 1)[0]


def _json_default(obj: Any) -> Any:
    """Fallback for non-JSON-native types in cache keys.

    Enums become their value; objects with ``to_dict`` use it; otherwise
    fall back to ``repr`` so the key is still deterministic for that
    object type.
    """
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return repr(obj)

"""JSON schemas for structured claim decomposition output."""

from typing import Any, Dict

# Schema enforced on the LLM's structured decomposition output.
DECOMPOSITION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "description": "List of atomic claims extracted from the text",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": (
                            "A single atomic, self-contained factual statement"
                        ),
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of claim",
                        "enum": [
                            "factual",
                            "opinion",
                            "conditional",
                            "comparative",
                            "causal",
                        ],
                    },
                    "source_sentence": {
                        "type": "string",
                        "description": (
                            "The original sentence this claim was derived from"
                        ),
                    },
                },
                "required": ["claim", "type"],
            },
        },
    },
    "required": ["claims"],
}

"""JSON schemas describing structured judge responses."""

from typing import Any, Dict

# Schema enforced on the LLM judge's structured output.
JUDGMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Overall score from 0-100",
            "minimum": 0,
            "maximum": 100,
        },
        "reasoning": {
            "type": "string",
            "description": "Detailed reasoning for the score",
        },
        "sub_scores": {
            "type": "object",
            "description": "Optional breakdown scores by dimension",
            "additionalProperties": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the judgment (0-1)",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": ["score", "reasoning"],
}

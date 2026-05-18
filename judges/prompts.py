"""Prompt templates for LLM-as-a-Judge evaluation.

Each rubric is a string template with named placeholders. The exact set of
placeholders depends on the rubric (e.g. factuality uses ``context`` and
``answer``; helpfulness uses ``question`` and ``answer``).
"""

SYSTEM_MESSAGE: str = (
    "You are an expert evaluator. Always return your evaluation as a valid "
    "JSON object that conforms to the provided schema. Be objective, "
    "thorough, and consistent in your scoring."
)

FACTUALITY: str = (
    "Evaluate the factuality of this answer based on the provided context.\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "SCORING GUIDELINES:\n"
    "- 0-20: Completely false/made up, contradicts context\n"
    "- 21-40: Mostly false with minor truthful elements\n"
    "- 41-60: Partially true with significant errors/omissions\n"
    "- 61-80: Mostly true with minor inaccuracies\n"
    "- 81-100: Completely accurate and well-supported by context\n\n"
    "Return a JSON object with your evaluation."
)

HELPFULNESS: str = (
    "Evaluate how helpful this answer is for the user's question.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "SCORING GUIDELINES:\n"
    "- 0-20: Not helpful at all, irrelevant or misleading\n"
    "- 21-40: Slightly helpful but incomplete or vague\n"
    "- 41-60: Somewhat helpful but has significant issues\n"
    "- 61-80: Helpful and mostly addresses the question\n"
    "- 81-100: Extremely helpful, complete, and insightful\n\n"
    "Return a JSON object with your evaluation."
)

COHERENCE: str = (
    "Evaluate the coherence and clarity of this answer.\n\n"
    "ANSWER:\n{answer}\n\n"
    "SCORING GUIDELINES:\n"
    "- 0-20: Incoherent, confusing, unreadable\n"
    "- 21-40: Poorly organized, hard to follow\n"
    "- 41-60: Somewhat clear but with organizational issues\n"
    "- 61-80: Clear and well-organized\n"
    "- 81-100: Exceptionally clear, logical, and easy to follow\n\n"
    "Return a JSON object with your evaluation."
)

COMPREHENSIVENESS: str = (
    "Evaluate how comprehensive and thorough this answer is.\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:\n{answer}\n\n"
    "SCORING GUIDELINES:\n"
    "- 0-20: Very superficial, misses key points\n"
    "- 21-40: Covers some aspects but misses important details\n"
    "- 41-60: Moderately comprehensive with gaps\n"
    "- 61-80: Comprehensive covering most important aspects\n"
    "- 81-100: Exceptionally thorough and complete\n\n"
    "Return a JSON object with your evaluation."
)

# Registry of available rubrics keyed by criteria name.
RUBRICS: dict[str, str] = {
    "factuality": FACTUALITY,
    "helpfulness": HELPFULNESS,
    "coherence": COHERENCE,
    "comprehensiveness": COMPREHENSIVENESS,
}

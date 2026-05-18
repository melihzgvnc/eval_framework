"""Prompt templates for LLM-based claim decomposition.

These prompts instruct the LLM to break a text into atomic,
independently verifiable claims.
"""

SYSTEM_MESSAGE: str = (
    "You are an expert at analyzing text and decomposing it into atomic claims. "
    "An atomic claim is a single, self-contained factual statement that can be "
    "independently verified as true or false. Be precise and exhaustive."
)

DECOMPOSITION_PROMPT: str = (
    "Decompose the following text into a list of atomic claims. Each claim should be:\n"
    "1. Self-contained: understandable without the other claims\n"
    "2. Atomic: expressing exactly one piece of information\n"
    "3. Faithful: preserving the meaning of the original text\n"
    "4. De-contextualized: replacing pronouns and references with explicit mentions\n\n"
    "TEXT:\n{text}\n\n"
    "Extract every factual claim made in the text. If the text contains opinions, "
    "conditional statements, or comparisons, represent them as claims that attribute "
    "the opinion/condition/comparison to the text.\n\n"
    "Return a JSON object with your decomposition."
)

DECOMPOSITION_WITH_CONTEXT_PROMPT: str = (
    "Decompose the following text into a list of atomic claims. Each claim should be:\n"
    "1. Self-contained: understandable without the other claims\n"
    "2. Atomic: expressing exactly one piece of information\n"
    "3. Faithful: preserving the meaning of the original text\n"
    "4. De-contextualized: replacing pronouns and references with explicit mentions\n\n"
    "CONTEXT (for reference resolution only):\n{context}\n\n"
    "TEXT TO DECOMPOSE:\n{text}\n\n"
    "Use the context to resolve any ambiguous references (pronouns, \"it\", \"they\", etc.) "
    "in the text, but only extract claims from the TEXT, not from the context.\n\n"
    "Return a JSON object with your decomposition."
)

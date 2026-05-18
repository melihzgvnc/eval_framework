"""Report renderers.

Each renderer consumes a Report and produces output in a specific
format (JSON, Markdown, HTML, etc.).
"""

from .base import BaseRenderer
from .json_renderer import JSONRenderer
from .markdown_renderer import MarkdownRenderer

__all__ = [
    "BaseRenderer",
    "JSONRenderer",
    "MarkdownRenderer",
]


def __getattr__(name):
    """Lazy import for HTMLRenderer to avoid optional dependency at import time."""
    if name == "HTMLRenderer":
        from .html_renderer import HTMLRenderer
        return HTMLRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

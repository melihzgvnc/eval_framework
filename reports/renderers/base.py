"""Base renderer abstraction.

Renderers consume a Report and produce a string (or bytes) in some
target format. Each renderer is independent — adding a new format
means adding a new subclass.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from eval_framework.reports.types import Report


class BaseRenderer(ABC):
    """Abstract base for report renderers."""

    #: File extension for the output format (e.g., ".json", ".md").
    extension: str = ""

    @abstractmethod
    def render(self, report: Report) -> str:
        """Render a Report into the target format.

        Args:
            report: The Report to render.

        Returns:
            String representation in the renderer's format.
        """
        pass

    def render_to_file(
        self, report: Report, path: Union[str, Path]
    ) -> Path:
        """Render and write the report to disk.

        Args:
            report: The Report to render.
            path: Output file path.

        Returns:
            Path to the written file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(report), encoding="utf-8")
        return path

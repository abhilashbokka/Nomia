"""Shared exception types used across the pipeline.

Kept in their own module (rather than defined in extract.py/classify.py/pipeline.py) to avoid
circular imports between the modules that need to raise and catch them.
"""

from __future__ import annotations


class NomiaError(Exception):
    """Base class for all Nomia-specific errors."""


class ExtractionError(NomiaError):
    """Raised when a file's signals (image/PDF content) cannot be extracted."""


class CorruptFileError(ExtractionError):
    """The file exists and is readable at the OS level, but its content is not a valid
    image/PDF (truncated, malformed, zero-byte, etc.)."""


class EncryptedPDFError(ExtractionError):
    """The PDF is password-protected and could not be opened with an empty password."""


class UnsupportedFormatError(ExtractionError):
    """The file's extension/content is not an image or PDF that Nomia knows how to render."""


class ClassificationError(NomiaError):
    """Raised for classification failures that aren't simply 'model returned bad JSON'
    (that case is handled via ClassificationOutcome.route == 'failed', not an exception)."""


class ModelNotAvailableError(ClassificationError):
    """The configured Ollama model is not pulled / not reachable."""

    def __init__(self, model: str, message: str | None = None):
        self.model = model
        super().__init__(message or f"Model '{model}' is not available. Run: ollama pull {model}")


class NamingError(NomiaError):
    """Raised for naming/template resolution failures."""


class OrganizerError(NomiaError):
    """Raised for undo-journal / apply / undo failures that are not per-file (e.g. a
    corrupted journal database)."""

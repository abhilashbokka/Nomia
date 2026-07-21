"""Optional macOS-only OCR tier using Apple's on-device Vision framework, via the `ocrmac`
package. Capability-gated per CLAUDE.md's cross-platform rule: any OS-specific signal is an
optional enhancement behind a capability check, with graceful degradation everywhere else -
non-macOS platforms and machines without the optional dependency installed simply get an
empty result and fall back to vision-model-only classification, exactly as before this module
existed.
"""

from __future__ import annotations

import logging
import sys

from PIL import Image

logger = logging.getLogger(__name__)

_text_from_image = None
if sys.platform == "darwin":
    try:
        from ocrmac.ocrmac import text_from_image as _text_from_image
    except ImportError:
        _text_from_image = None


def is_available() -> bool:
    return _text_from_image is not None


def ocr_image(image: Image.Image) -> str:
    """Best-effort on-device OCR. Returns "" on any failure or when unavailable - OCR is a
    context-enrichment optimization here, never a hard requirement of the pipeline."""
    if _text_from_image is None:
        return ""
    try:
        lines = _text_from_image(image, recognition_level="accurate", detail=False)
    except Exception as exc:  # noqa: BLE001 - OCR must never crash the batch
        logger.debug("OCR failed: %s", exc)
        return ""
    return "\n".join(lines)

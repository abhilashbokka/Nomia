"""Heuristic quality scoring for extracted/OCR'd text - decides whether text is clean and
substantial enough to let the classifier skip the vision model's image input entirely.
Conceptually adapted from the autorename-pdf reference implementation's assess_text_quality():
combines raw length, alphanumeric density, and average-word-length sanity into one score in
[0, 1], rather than trusting text presence alone (a single stray character or OCR noise line
is "text" but not usable text).
"""

from __future__ import annotations

TEXT_QUALITY_THRESHOLD = 0.3
_LENGTH_SATURATION_CHARS = 200


def assess_text_quality(text: str | None) -> float:
    if not text:
        return 0.0
    stripped = text.strip()
    if not stripped:
        return 0.0

    words = stripped.split()
    if not words:
        return 0.0

    length_score = min(1.0, len(stripped) / _LENGTH_SATURATION_CHARS)
    alnum_ratio = sum(1 for c in stripped if c.isalnum()) / len(stripped)
    avg_word_len = sum(len(w) for w in words) / len(words)
    # Real prose/document text usually averages 2-12 chars/word; wildly outside that range
    # signals OCR noise or a garbled, space-free run-on rather than genuinely readable text.
    word_len_score = 1.0 if 2 <= avg_word_len <= 12 else 0.3

    # length_score gates (multiplies) the other two rather than being summed alongside them -
    # otherwise a single short, clean word like "Hi" scores highly on alnum-ratio/word-length
    # alone despite being exactly the sparse, single-line case this function must catch (the
    # same problem flagged with the old synthetic handwritten_note fixture).
    return round(length_score * (alnum_ratio * 0.6 + word_len_score * 0.4), 3)

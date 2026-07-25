"""Keyword evidence for the fast classification path: matches per-category keyword phrases
(from the user-editable taxonomy) against extracted text (PDF text layer or on-device OCR),
and derives a filename-worthy description slug from that text when the vision model is skipped.

Pure functions, no I/O, no model dependencies — this module works identically whether or not
the optional SigLIP/torch dependencies are installed.
"""

from __future__ import annotations

import re
from functools import lru_cache

from nomia.config import CategoryDef
from nomia.naming import slugify

# Only the first few lines of a document are title candidates; a merchant name, letterhead, or
# form title virtually always appears at the top of the extracted text.
_TITLE_CANDIDATE_LINES = 10
_MAX_TITLE_WORDS = 8
_MAX_DESCRIPTION_WORDS = 4
# Filler words dropped when a category key is used as a description suffix
# ("diagram_or_chart" -> "diagram-chart", not "diagram-or-chart").
_CATEGORY_FILLER_WORDS = {"or", "of", "and", "the", "a"}


@lru_cache(maxsize=1024)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary, case-insensitive pattern for a keyword phrase. Whitespace between the
    phrase's words matches any run of whitespace (including a line break in OCR output), so
    "beginning balance" still matches when OCR splits it across lines."""
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)


def match_keywords(text: str | None, taxonomy: list[CategoryDef]) -> dict[str, list[str]]:
    """Returns {category_key: [distinct matched phrases]} for every category with at least one
    hit. Categories with no keywords configured (photo, screenshot, ...) simply never appear —
    absence of keyword evidence is itself informative and handled by the caller's fusion step."""
    if not text or not text.strip():
        return {}
    matches: dict[str, list[str]] = {}
    for cat in taxonomy:
        hit = [kw for kw in cat.keywords if kw and _phrase_pattern(kw).search(text)]
        if hit:
            matches[cat.key] = hit
    return matches


def _category_slug(category_key: str) -> str:
    slug, _ = slugify(category_key.replace("_", " "))
    words = [w for w in slug.split("-") if w and w not in _CATEGORY_FILLER_WORDS]
    return "-".join(words) or "file"


def _best_title_line(text: str) -> str | None:
    """Picks the most title-like line from the top of the document: early, mostly alphabetic,
    not dominated by digits (order numbers, dates, barcodes), and short enough to be a heading
    rather than a sentence of body text."""
    best: str | None = None
    best_score = 0.0
    seen = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        seen += 1
        if seen > _TITLE_CANDIDATE_LINES:
            break
        if len(line) < 3 or len(line.split()) > _MAX_TITLE_WORDS:
            continue
        total = len(line)
        alpha = sum(c.isalpha() for c in line)
        digits = sum(c.isdigit() for c in line)
        if alpha / total < 0.5 or digits / total > 0.4:
            continue
        position_weight = 1.0 / (1.0 + 0.25 * (seen - 1))
        # Letterheads and store names are typically ALL CAPS — a mild bonus, not a requirement.
        caps_bonus = 1.2 if line.isupper() and alpha >= 4 else 1.0
        score = (alpha / total) * position_weight * caps_bonus
        if score > best_score:
            best, best_score = line, score
    return best


def derive_description(text: str | None, category_key: str) -> str:
    """The {description} token for fast-path classifications (where no vision model generates
    one): a slug from the document's most title-like line, suffixed with the category unless
    that would be redundant. Falls back to the bare category slug when there is no usable text
    (photos, handwriting OCR noise, empty scans)."""
    cat_slug = _category_slug(category_key)
    if not text or not text.strip():
        return cat_slug

    line = _best_title_line(text)
    if line is None:
        return cat_slug

    line_slug, _ = slugify(line)
    words = [w for w in line_slug.split("-") if w][:_MAX_DESCRIPTION_WORDS]
    if not words:
        return cat_slug

    cat_words = set(cat_slug.split("-"))
    if cat_words & set(words):
        return "-".join(words)
    return "-".join([*words, cat_slug])

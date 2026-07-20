"""Naming template engine: token resolution, slugify, copy-ordering (the {index} numbering
scheme), and collision-safe path resolution. Pure functions wherever possible — no disk
mutation happens in this module, only path computation. See CLAUDE.md for the copy-ordering
rule this module implements.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from nomia.config import NomiaConfig
from nomia.scanner import FileRecord, effective_creation_date

logger = logging.getLogger(__name__)

UNSORTED_FOLDER = "_Unsorted"
OTHER_FOLDER = "_Other"
DUMP_FOLDER = "_dump"

_ILLEGAL_CHARS_RE = re.compile(r'[/\\:*?"<>|]')
_TOKEN_RE = re.compile(r"\{(\w+)\}")
_SEPARATOR_CHARS = "-_. "


# --------------------------------------------------------------------------------------------
# Slugify / sanitize
# --------------------------------------------------------------------------------------------

def sanitize_original(name: str) -> str:
    """Strips only OS-illegal characters; preserves case, spacing, and Unicode. Distinct from
    slugify() - this backs the {original} token and _Unsorted//_Other//_dump/ filenames, where
    the point is to keep the file recognizable, not to produce a clean slug."""
    cleaned = _ILLEGAL_CHARS_RE.sub("", name)
    cleaned = cleaned.strip().rstrip(".")  # trailing dots/spaces are illegal on Windows
    return cleaned or "file"


def slugify(text: str, *, max_len: int = 60) -> tuple[str, list[str]]:
    """Lowercase, hyphenated, ASCII-safe slug for the {description}/{subcategory} tokens.
    Non-ASCII characters are transliterated via Unicode NFKD normalization (e.g. 'café' ->
    'cafe') using only the standard library - never silently dropped without a log entry."""
    log: list[str] = []
    original = text.strip()
    lowered = original.lower()

    normalized = unicodedata.normalize("NFKD", lowered)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    if any(ord(ch) > 127 for ch in lowered) and ascii_text.strip():
        log.append(f"transliterated non-ASCII text {original!r} -> {ascii_text!r}")
    elif any(ord(ch) > 127 for ch in lowered) and not ascii_text.strip():
        log.append(f"text {original!r} had no ASCII-safe representation; using fallback")

    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    slug = re.sub(r"-+", "-", slug).strip("-")

    if not slug:
        slug = "untitled"
        log.append(f"'{original}' produced an empty slug; used fallback '{slug}'")

    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
        log.append(f"truncated slug to {max_len} characters")

    return slug, log


def truncate_filename(stem: str, ext: str, max_bytes: int = 255) -> str:
    """UTF-8-byte-safe truncation: never cuts a multi-byte character in half, which a naive
    character-count truncation could do for non-ASCII stems (see the {original} token, which
    deliberately isn't ASCII-transliterated)."""
    ext_bytes = ext.encode("utf-8")
    budget = max_bytes - len(ext_bytes)
    if budget <= 0:
        return (stem[:1] if stem else "f") + ext

    stem_bytes = stem.encode("utf-8")
    if len(stem_bytes) <= budget:
        return stem + ext

    truncated = stem_bytes[:budget]
    while truncated and (truncated[-1] & 0b1100_0000) == 0b1000_0000:
        truncated = truncated[:-1]
    return truncated.decode("utf-8", errors="ignore") + ext


# --------------------------------------------------------------------------------------------
# Template token resolution
# --------------------------------------------------------------------------------------------

@dataclass
class NamingContext:
    category: str
    subcategory: str | None
    description: str
    original_stem: str
    index_str: str | None
    date: datetime | None
    confidence: float
    location: str | None = None


def _token_values(ctx: NamingContext) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "category": ctx.category or None,
        "subcategory": ctx.subcategory or None,
        "description": ctx.description or None,
        "original": ctx.original_stem or None,
        "index": ctx.index_str,
        "confidence": f"{ctx.confidence:.2f}",
        "location": ctx.location or None,
    }
    if ctx.date is not None:
        values["yyyy"] = f"{ctx.date.year:04d}"
        values["mm"] = f"{ctx.date.month:02d}"
        values["dd"] = f"{ctx.date.day:02d}"
        values["date"] = ctx.date.strftime("%Y-%m-%d")
    else:
        values["yyyy"] = values["mm"] = values["dd"] = values["date"] = None
    return values


def _merge_literal(pending: str, new: str) -> str:
    """Joins two literal fragments that became adjacent because a token between them was
    missing. If both fragments are pure separator runs at their touching edge, keep only the
    longer of the two runs there instead of concatenating both in full - this is what turns
    "{yyyy}-{mm}-{dd}_{description}" (all three date tokens missing) into a single dropped
    separator rather than a run of three. Any non-separator literal text (an intentional
    multi-character literal like the "__" in "{original}__{category}", or custom trailing
    text in a user template) is never touched - only genuinely adjacent separator-only runs
    get collapsed."""
    if not pending or not new:
        return pending + new

    trailing = len(pending) - len(pending.rstrip(_SEPARATOR_CHARS))
    leading = len(new) - len(new.lstrip(_SEPARATOR_CHARS))
    if trailing and leading:
        keep = max(trailing, leading)
        pending_core = pending[: len(pending) - trailing]
        new_core = new[leading:]
        separator_run = (pending[len(pending) - trailing :] if trailing >= leading else new[:leading])[:keep]
        return pending_core + separator_run + new_core
    return pending + new


def _resolve_segment(segment: str, values: dict[str, str | None]) -> str:
    """Walks the segment as alternating literal/token/literal/... parts, emitting a literal
    "pending" separator only when real content follows it (never a leading separator before
    the first resolved value, never a bare trailing separator left by an omitted final token).
    A missing token contributes nothing, but its surrounding literals still merge via
    _merge_literal so a run of several missing tokens collapses to at most one separator."""
    parts = _TOKEN_RE.split(segment)
    output: list[str] = []
    pending = ""

    for i, part in enumerate(parts):
        if i % 2 == 0:
            pending = _merge_literal(pending, part)
            continue

        if part not in values:
            logger.warning("Naming template references unknown token '{%s}' - treating it as empty.", part)
        value = values.get(part)
        if value:
            if output:
                output.append(pending)
            output.append(value)
            pending = ""
        # else: missing token contributes nothing; pending already carries its neighbors.

    if output and pending.strip(_SEPARATOR_CHARS):
        output.append(pending)

    return "".join(output)


def resolve_template(template: str, ctx: NamingContext) -> str:
    """Resolves a full naming template into a destination-relative path (no extension, no
    leading/trailing slash). Splits on the literal '/' for folder segments so templates are
    portable regardless of the host OS path separator; the caller joins the result with
    Path(...) for the actual filesystem operation."""
    values = _token_values(ctx)
    segments = [_resolve_segment(seg, values) for seg in template.split("/")]
    return "/".join(seg for seg in segments if seg)


# --------------------------------------------------------------------------------------------
# Destination collision tracking
# --------------------------------------------------------------------------------------------

class DestinationIndex:
    """Tracks every relative path claimed at the destination - seeded from a real directory
    listing at plan time, then updated in-memory as each planned item reserves a name, so two
    items planned within the *same* run can't collide with each other either, not just against
    pre-existing files."""

    def __init__(self, destination_root: Path):
        self._destination_root = destination_root
        self._taken: set[Path] = set()
        if destination_root.exists():
            for path in destination_root.rglob("*"):
                if path.is_file():
                    try:
                        self._taken.add(path.relative_to(destination_root))
                    except ValueError:
                        continue

    def is_taken(self, rel_path: Path) -> bool:
        return rel_path in self._taken

    def reserve(self, rel_path: Path) -> None:
        self._taken.add(rel_path)


def resolve_collision(rel_path: Path, dest_index: DestinationIndex) -> Path:
    """Never overwrites: if rel_path is already claimed (on disk or by an earlier item in this
    same run), appends a mechanical, non-semantic suffix (__2, __3, ...) before the extension -
    deliberately distinct in form from the semantic {index} naming token, since a collision here
    usually means the chosen template doesn't disambiguate copies on its own."""
    if not dest_index.is_taken(rel_path):
        dest_index.reserve(rel_path)
        return rel_path

    parent, stem, ext = rel_path.parent, rel_path.stem, rel_path.suffix
    counter = 2
    while True:
        candidate = parent / f"{stem}__{counter}{ext}"
        if not dest_index.is_taken(candidate):
            dest_index.reserve(candidate)
            logger.warning("collision_resolved: '%s' was already taken; using '%s' instead", rel_path, candidate)
            return candidate
        counter += 1


# --------------------------------------------------------------------------------------------
# Special buckets: _Unsorted/, _Other/, _dump/ bypass the naming template entirely
# --------------------------------------------------------------------------------------------

def unsorted_relative_path(original_filename: str) -> Path:
    return Path(UNSORTED_FOLDER) / sanitize_original(original_filename)


def other_relative_path(original_filename: str) -> Path:
    return Path(OTHER_FOLDER) / sanitize_original(original_filename)


def dump_relative_path(original_filename: str) -> Path:
    return Path(DUMP_FOLDER) / sanitize_original(original_filename)


# --------------------------------------------------------------------------------------------
# Copy-ordering + full naming plan for auto/review-routed items
# --------------------------------------------------------------------------------------------

@dataclass
class NamingCandidate:
    """Everything naming.py needs for one file, decoupled from pipeline.py's richer PlannedItem
    so this module has no dependency on the orchestration layer."""
    record: FileRecord
    category_key: str
    subcategory: str | None
    raw_description: str
    confidence: float
    location: str | None = None


@dataclass
class NamedItem:
    candidate: NamingCandidate
    dest_relative_path: Path
    naming_index: int | None
    naming_date_source: str
    transform_log: list[str] = field(default_factory=list)


def _category_folder_value(category_key: str, cfg: NomiaConfig) -> str:
    category = cfg.category_by_key(category_key)
    value = category.destination_subfolder if category is not None else category_key
    return sanitize_original(value) or sanitize_original(category_key)


def _build_base_context(candidate: NamingCandidate, cfg: NomiaConfig) -> tuple[NamingContext, list[str], str]:
    description_slug, desc_log = slugify(candidate.raw_description)
    subcategory_slug = None
    if candidate.subcategory:
        subcategory_slug, sub_log = slugify(candidate.subcategory)
        desc_log = desc_log + sub_log

    date, date_source = effective_creation_date(candidate.record)
    original_stem = sanitize_original(candidate.record.path.stem)
    category_value = _category_folder_value(candidate.category_key, cfg)

    ctx = NamingContext(
        category=category_value,
        subcategory=subcategory_slug,
        description=description_slug,
        original_stem=original_stem,
        index_str=None,
        date=date,
        confidence=candidate.confidence,
        location=candidate.location,
    )
    return ctx, desc_log, date_source


def plan_organized_names(
    candidates: list[NamingCandidate],
    cfg: NomiaConfig,
    dest_index: DestinationIndex,
) -> list[NamedItem]:
    """The core naming pass for auto/review-routed files: renders each candidate's template,
    groups same-rendered-base candidates for {index} disambiguation (sorted ascending by
    effective creation date, tie-broken by normalized path string for stability across
    re-runs), then resolves every result against the shared DestinationIndex so nothing here
    can ever collide with anything else planned in this run or already on disk."""
    template = cfg.active_naming_template()

    prepared: list[tuple[NamingCandidate, NamingContext, list[str], str]] = []
    for candidate in candidates:
        ctx, transform_log, date_source = _build_base_context(candidate, cfg)
        prepared.append((candidate, ctx, transform_log, date_source))

    base_keys = [resolve_template(template, ctx) for _candidate, ctx, _log, _src in prepared]

    groups: dict[str, list[int]] = {}
    for i, key in enumerate(base_keys):
        groups.setdefault(key, []).append(i)

    index_str_by_position: dict[int, str | None] = {}
    for key, positions in groups.items():
        if len(positions) == 1:
            index_str_by_position[positions[0]] = None
            continue

        def _sort_key(pos: int) -> tuple[int, float, str]:
            _candidate, ctx, _log, _src = prepared[pos]
            has_date = 0 if ctx.date is not None else 1
            timestamp = ctx.date.timestamp() if ctx.date is not None else float("inf")
            path_str = str(prepared[pos][0].record.path).lower()
            return (has_date, timestamp, path_str)

        ordered_positions = sorted(positions, key=_sort_key)
        width = max(2, len(str(len(ordered_positions))))
        for rank, pos in enumerate(ordered_positions, start=1):
            index_str_by_position[pos] = str(rank).zfill(width)
            if "{index}" not in template:
                logger.info(
                    "Naming template has no {index} token but %d files share the base name '%s'; "
                    "they will be disambiguated via the collision suffix instead.", len(ordered_positions), key,
                )

    results: list[NamedItem] = []
    # Process in the same deterministic order used for grouping/tie-breaking, so collision
    # suffix assignment (__2, __3, ...) is stable across re-runs too.
    order = sorted(range(len(prepared)), key=lambda i: str(prepared[i][0].record.path).lower())
    for pos in order:
        candidate, ctx, transform_log, date_source = prepared[pos]
        ctx.index_str = index_str_by_position[pos]
        rendered = resolve_template(template, ctx)
        ext = candidate.record.path.suffix
        rel_path = Path(rendered) if rendered else Path(sanitize_original(candidate.record.path.stem))
        stem = rel_path.name
        truncated_name = truncate_filename(stem, ext, cfg.max_filename_bytes)
        full_rel_path = rel_path.parent / truncated_name if rel_path.parent != Path(".") else Path(truncated_name)

        resolved_path = resolve_collision(full_rel_path, dest_index)
        results.append(
            NamedItem(
                candidate=candidate,
                dest_relative_path=resolved_path,
                naming_index=int(ctx.index_str) if ctx.index_str else None,
                naming_date_source=date_source,
                transform_log=transform_log,
            )
        )

    return results

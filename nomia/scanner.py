"""Walks source folders, hashes every file, and groups byte-identical duplicates.

Each file is read once here (for hashing) — extract.py reads it again only for content
signals (EXIF/PDF render), never twice for the same purpose. Duplicate detection at this
stage is purely by content (SHA-256); grouping same-name-different-content "copies" for the
{index} numbering scheme is naming.py's job, not this module's.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_JUNK_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}

# Matches YYYY-MM-DD, YYYY_MM_DD, or YYYYMMDD anywhere in a filename stem (e.g. IMG_20260720,
# Screenshot 2026-07-20, scan_2026_07_20_001).
_FILENAME_DATE_RE = re.compile(r"(20\d{2}|19\d{2})[-_]?(\d{2})[-_]?(\d{2})")


@dataclass(frozen=True)
class FileRecord:
    path: Path
    size: int
    sha256: str
    created_at: datetime | None
    modified_at: datetime | None
    ext: str
    source_root: Path
    discovery_seq: int
    is_duplicate_of: Path | None = None


def walk_sources(
    source_folders: list[Path],
    *,
    follow_symlinks: bool = False,
    exclude_dirs: list[Path] | None = None,
) -> Iterator[tuple[Path, Path]]:
    """Yields (file_path, source_root) pairs. `exclude_dirs` lets callers (pipeline.py) keep a
    destination_root that happens to be nested inside a source folder from being re-scanned as
    if it were new input."""
    resolved_excludes = [Path(p).resolve() for p in (exclude_dirs or [])]

    for source in source_folders:
        root = Path(source).expanduser().resolve()
        if not root.is_dir():
            logger.warning("Source folder does not exist or is not a directory, skipping: %s", root)
            continue

        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            current = Path(dirpath).resolve()
            if any(current == ex or ex in current.parents for ex in resolved_excludes):
                dirnames[:] = []
                continue
            # Prune excluded dirs before os.walk descends into them.
            dirnames[:] = [
                d for d in dirnames
                if not any((current / d).resolve() == ex or ex in (current / d).resolve().parents for ex in resolved_excludes)
            ]
            for filename in sorted(filenames):
                if filename in _JUNK_FILENAMES:
                    continue
                yield (Path(dirpath) / filename, root)


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 — never loads a whole large file into memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_creation_time(path: Path) -> datetime | None:
    """macOS exposes true creation time via st_birthtime. Windows' st_ctime is creation time
    (not "change time" as on Unix). Most Linux filesystems/kernels don't expose creation time
    through os.stat() at all, so this legitimately returns None there — naming.py's fallback
    chain is what makes that safe."""
    try:
        stat = path.stat()
    except OSError:
        return None

    system = platform.system()
    if system == "Windows":
        return datetime.fromtimestamp(stat.st_ctime)

    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime:
        return datetime.fromtimestamp(birthtime)
    return None


def parse_date_from_filename(name: str) -> datetime | None:
    match = _FILENAME_DATE_RE.search(name)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def effective_creation_date(record: FileRecord) -> tuple[datetime | None, str]:
    """The fallback chain from CLAUDE.md: OS creation -> OS modified -> filename-parsed date ->
    discovery order. Returns (date_or_None, tier_used) so callers can log which tier fired.
    A None date signals "discovery order" (no real date signal available at all) — callers must
    still produce a deterministic order in that case (naming.py ties back to the normalized
    relative path string, not this function)."""
    if record.created_at is not None:
        return record.created_at, "os_creation"
    if record.modified_at is not None:
        return record.modified_at, "os_modified"
    parsed = parse_date_from_filename(record.path.name)
    if parsed is not None:
        return parsed, "filename_parsed"
    return None, "discovery_order"


def scan(
    source_folders: list[Path],
    *,
    exclude_dirs: list[Path] | None = None,
) -> list[FileRecord]:
    """Walks every source folder and produces one FileRecord per file, including hash-duplicates
    (annotated via dedupe_by_hash, not dropped here)."""
    records: list[FileRecord] = []
    seq = 0
    for path, root in walk_sources(source_folders, exclude_dirs=exclude_dirs):
        try:
            stat = path.stat()
        except OSError as exc:
            logger.warning("Could not stat %s (%s); skipping.", path, exc)
            continue

        try:
            digest = hash_file(path)
        except OSError as exc:
            logger.warning("Could not read %s to hash it (%s); skipping.", path, exc)
            continue

        record = FileRecord(
            path=path,
            size=stat.st_size,
            sha256=digest,
            created_at=get_creation_time(path),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            ext=path.suffix.lower(),
            source_root=root,
            discovery_seq=seq,
        )
        records.append(record)
        seq += 1

    logger.info("Scanned %d files across %d source folder(s).", len(records), len(source_folders))
    return records


def dedupe_by_hash(records: list[FileRecord]) -> list[FileRecord]:
    """Groups records by SHA-256. Within a group of true byte-identical duplicates, the
    "keeper" is whichever file has the earliest effective_creation_date (using the same
    fallback chain naming.py uses for copy-ordering, with the same path-string tie-break so
    the choice is stable across re-runs). Every other record in the group is annotated with
    is_duplicate_of pointing at the keeper — records are never dropped, so the pipeline can log
    every duplicate explicitly rather than silently discarding them."""
    by_hash: dict[str, list[FileRecord]] = {}
    for record in records:
        by_hash.setdefault(record.sha256, []).append(record)

    result: list[FileRecord] = []
    for group in by_hash.values():
        if len(group) == 1:
            result.append(group[0])
            continue

        def sort_key(r: FileRecord) -> tuple[int, float, str]:
            date, _tier = effective_creation_date(r)
            timestamp = date.timestamp() if date is not None else float("inf")
            has_date = 0 if date is not None else 1
            return (has_date, timestamp, str(r.path).lower())

        ordered = sorted(group, key=sort_key)
        keeper = ordered[0]
        result.append(keeper)
        for dup in ordered[1:]:
            result.append(replace(dup, is_duplicate_of=keeper.path))
            logger.info("Duplicate content detected: %s is a byte-identical duplicate of %s", dup.path, keeper.path)

    return result

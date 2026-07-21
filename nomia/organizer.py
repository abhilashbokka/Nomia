"""The undo journal (SQLite) plus apply/undo/verify/resume logic. This is where the five
non-negotiable invariants from CLAUDE.md are actually enforced in code: never overwrite, never
delete-before-verify, log every action, dry-run first (nothing here mutates disk until
apply_plan is explicitly called), and every applied run is fully undoable.

server.py and cli.py both call the functions in this module directly - neither has its own
copy of this logic.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nomia.config import NomiaConfig
from nomia.errors import OrganizerError
from nomia.scanner import hash_file

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  source_folders TEXT NOT NULL,
  destination_root TEXT NOT NULL,
  naming_template TEXT NOT NULL,
  model_used TEXT NOT NULL,
  thresholds_json TEXT NOT NULL,
  config_snapshot_json TEXT NOT NULL,
  report_path TEXT,
  totals_json TEXT,
  verification_json TEXT
);

CREATE TABLE IF NOT EXISTS items (
  item_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  source_path TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  source_size INTEGER NOT NULL,
  dest_path TEXT,
  dest_sha256 TEXT,
  dump_path TEXT,
  dump_sha256 TEXT,
  source_preserved INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  route TEXT NOT NULL,
  user_decision TEXT NOT NULL DEFAULT 'pending',
  name_override TEXT,
  category TEXT,
  subcategory TEXT,
  description TEXT,
  confidence REAL,
  reason TEXT,
  raw_model_response TEXT,
  duplicate_of TEXT,
  naming_index INTEGER,
  naming_date_source TEXT,
  error TEXT,
  warning TEXT,
  created_at TEXT NOT NULL,
  applied_at TEXT,
  undone_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_run ON items(run_id);
CREATE INDEX IF NOT EXISTS idx_items_dest_path ON items(dest_path);
CREATE INDEX IF NOT EXISTS idx_items_source_path ON items(source_path);

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id INTEGER NOT NULL REFERENCES items(item_id),
  run_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  from_status TEXT,
  to_status TEXT,
  detail TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ItemRecord:
    item_id: int
    run_id: str
    source_path: str
    source_sha256: str
    source_size: int
    dest_path: str | None
    dest_sha256: str | None
    dump_path: str | None
    dump_sha256: str | None
    source_preserved: bool
    status: str
    route: str
    user_decision: str
    name_override: str | None
    category: str | None
    subcategory: str | None
    description: str | None
    confidence: float | None
    reason: str | None
    raw_model_response: str | None
    duplicate_of: str | None
    naming_index: int | None
    naming_date_source: str | None
    error: str | None
    warning: str | None
    created_at: str
    applied_at: str | None
    undone_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ItemRecord":
        data = dict(row)
        data["source_preserved"] = bool(data["source_preserved"])
        return cls(**data)

    def effective_dest_name(self) -> str | None:
        """The name actually used for the applied copy - name_override takes precedence over
        the originally-planned dest_path if the user edited it during review."""
        if self.name_override:
            return self.name_override
        return Path(self.dest_path).name if self.dest_path else None


@dataclass
class RunRecord:
    run_id: str
    status: str
    started_at: str
    finished_at: str | None
    source_folders: list[str]
    destination_root: str
    naming_template: str
    model_used: str
    thresholds_json: dict
    config_snapshot_json: dict
    report_path: str | None
    totals_json: dict
    verification_json: dict | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RunRecord":
        data = dict(row)
        data["source_folders"] = json.loads(data["source_folders"])
        data["thresholds_json"] = json.loads(data["thresholds_json"])
        data["config_snapshot_json"] = json.loads(data["config_snapshot_json"])
        data["totals_json"] = json.loads(data["totals_json"]) if data["totals_json"] else {}
        data["verification_json"] = json.loads(data["verification_json"]) if data["verification_json"] else None
        return cls(**data)


class VerificationReport(BaseModel):
    scanned_count: int
    accounted_count: int
    counts_by_status: dict[str, int]
    hash_matches: int
    hash_mismatches: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.scanned_count == self.accounted_count and not self.hash_mismatches


class ApplyResult(BaseModel):
    run_id: str
    applied: int
    failed: int
    skipped: int
    verification: VerificationReport


class UndoResult(BaseModel):
    run_id: str
    undone: int
    skipped: int
    details: list[dict[str, Any]]


class ResumeResult(BaseModel):
    run_id: str
    finalized: int
    failed: int


class UndoJournal:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- runs -------------------------------------------------------------------------------

    def start_run(
        self,
        *,
        source_folders: list[str],
        destination_root: str,
        naming_template: str,
        model_used: str,
        thresholds: dict,
        config_snapshot: dict,
        run_id: str | None = None,
    ) -> str:
        """A caller (server.py) that needs to know the run_id *before* the slow scan/classify
        work begins - to register a progress-polling job under that same id - can pass one in
        explicitly; otherwise a fresh one is generated here."""
        run_id = run_id or uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, status, started_at, source_folders, destination_root, "
                "naming_template, model_used, thresholds_json, config_snapshot_json, totals_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, "planning", _now(), json.dumps(source_folders), destination_root,
                    naming_template, model_used, json.dumps(thresholds), json.dumps(config_snapshot),
                    json.dumps({}),
                ),
            )
            self._conn.commit()
        return run_id

    def set_run_totals(self, run_id: str, totals: dict) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET totals_json = ? WHERE run_id = ?", (json.dumps(totals), run_id))
            self._conn.commit()

    def set_run_status(self, run_id: str, status: str, *, finished: bool = False) -> None:
        with self._lock:
            if finished:
                self._conn.execute(
                    "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?", (status, _now(), run_id)
                )
            else:
                self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))
            self._conn.commit()

    def set_run_report_path(self, run_id: str, report_path: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET report_path = ? WHERE run_id = ?", (report_path, run_id))
            self._conn.commit()

    def set_run_verification(self, run_id: str, report: VerificationReport) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET verification_json = ? WHERE run_id = ?",
                (report.model_dump_json(), run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.from_row(row) if row else None

    def get_last_applied_run_id(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM runs WHERE status IN ('applied', 'applied_with_verification_errors', "
                "'partially_applied') ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
        return row["run_id"] if row else None

    def find_interrupted_runs(self) -> list[str]:
        """Runs left at 'applying' by an unclean shutdown - candidates for resume_crashed_run."""
        with self._lock:
            rows = self._conn.execute("SELECT run_id FROM runs WHERE status = 'applying'").fetchall()
        return [row["run_id"] for row in rows]

    # --- items --------------------------------------------------------------------------------

    def record_planned(
        self,
        run_id: str,
        *,
        source_path: str,
        source_sha256: str,
        source_size: int,
        dest_path: str | None,
        dump_path: str | None,
        route: str,
        category: str | None = None,
        subcategory: str | None = None,
        description: str | None = None,
        confidence: float | None = None,
        reason: str | None = None,
        raw_model_response: str | None = None,
        duplicate_of: str | None = None,
        naming_index: int | None = None,
        naming_date_source: str | None = None,
        error: str | None = None,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO items (run_id, source_path, source_sha256, source_size, dest_path, "
                "dump_path, status, route, category, subcategory, description, confidence, reason, "
                "raw_model_response, duplicate_of, naming_index, naming_date_source, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, source_path, source_sha256, source_size, dest_path, dump_path, route,
                    category, subcategory, description, confidence, reason, raw_model_response,
                    duplicate_of, naming_index, naming_date_source, error, _now(),
                ),
            )
            self._conn.commit()
            return cursor.lastrowid

    def get_items(self, run_id: str, *, status: str | None = None) -> list[ItemRecord]:
        with self._lock:
            if status is not None:
                rows = self._conn.execute(
                    "SELECT * FROM items WHERE run_id = ? AND status = ? ORDER BY item_id", (run_id, status)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM items WHERE run_id = ? ORDER BY item_id", (run_id,)
                ).fetchall()
        return [ItemRecord.from_row(row) for row in rows]

    def get_item(self, item_id: int) -> ItemRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
        return ItemRecord.from_row(row) if row else None

    def find_applied_by_source_path(self, source_path: str) -> ItemRecord | None:
        """Backs idempotency: has this exact source path already been organized by a previous
        run? Only relevant when preserve_source left the file in place to be re-discovered by a
        later scan - a normal move would make the source path disappear entirely."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM items WHERE source_path = ? AND status = 'applied' "
                "ORDER BY applied_at DESC LIMIT 1",
                (source_path,),
            ).fetchone()
        return ItemRecord.from_row(row) if row else None

    def _set_item_status(self, item_id: int, status: str, **fields: Any) -> None:
        with self._lock:
            row = self._conn.execute("SELECT status FROM items WHERE item_id = ?", (item_id,)).fetchone()
            from_status = row["status"] if row else None

            set_clauses = ["status = ?"]
            values: list[Any] = [status]
            for key, value in fields.items():
                set_clauses.append(f"{key} = ?")
                values.append(value)
            values.append(item_id)
            self._conn.execute(f"UPDATE items SET {', '.join(set_clauses)} WHERE item_id = ?", values)

            self._conn.execute(
                "INSERT INTO events (item_id, run_id, ts, from_status, to_status, detail) "
                "VALUES (?, (SELECT run_id FROM items WHERE item_id = ?), ?, ?, ?, ?)",
                (item_id, item_id, _now(), from_status, status, json.dumps(fields, default=str)),
            )
            self._conn.commit()

    def mark_in_progress(self, item_id: int) -> None:
        self._set_item_status(item_id, "in_progress")

    def mark_applied(
        self,
        item_id: int,
        *,
        dest_relative_path: str,
        dest_sha256: str,
        dump_relative_path: str | None = None,
        dump_sha256: str | None = None,
        source_preserved: bool = False,
        warning: str | None = None,
    ) -> None:
        """dest_path/dump_path are always stored relative to the run's destination_root, both
        before and after applying - never as absolute paths. This call updates them to the
        *final* relative path actually used (which may differ from the originally-planned one
        if the user's review-stage name_override changed the filename), so a later undo/verify
        resolves against where the file truly ended up."""
        self._set_item_status(
            item_id, "applied",
            dest_path=dest_relative_path, dest_sha256=dest_sha256,
            dump_path=dump_relative_path, dump_sha256=dump_sha256,
            source_preserved=int(source_preserved), warning=warning, applied_at=_now(),
        )

    def mark_failed(self, item_id: int, error: str) -> None:
        self._set_item_status(item_id, "failed", error=error)

    def mark_skipped(self, item_id: int, reason: str) -> None:
        self._set_item_status(item_id, "skipped", error=reason)

    def mark_undone(self, item_id: int) -> None:
        self._set_item_status(item_id, "undone", undone_at=_now())

    def mark_undo_skipped(self, item_id: int, reason: str) -> None:
        self._set_item_status(item_id, "undo_skipped_modified", error=reason)

    def set_user_decision(self, item_id: int, *, user_decision: str | None = None, name_override: str | None = None) -> None:
        fields: dict[str, Any] = {}
        if user_decision is not None:
            fields["user_decision"] = user_decision
        if name_override is not None:
            fields["name_override"] = name_override
        if not fields:
            return
        with self._lock:
            set_clauses = ", ".join(f"{k} = ?" for k in fields)
            self._conn.execute(f"UPDATE items SET {set_clauses} WHERE item_id = ?", (*fields.values(), item_id))
            self._conn.commit()


# --------------------------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------------------------

def _should_include_in_apply(item: ItemRecord) -> bool:
    # "unsorted" is not "do nothing" - it's an automatic filing action into a fixed, safe
    # bucket, just flagged for visibility (low confidence, corrupt/encrypted/unreadable file,
    # or a failed classification). "other" is the same idea for swept non-image/PDF files.
    # "left_untouched" (non-image/PDF files when sweeping is disabled) and the skip_* /
    # error routes are recorded for the report but never touch disk.
    if item.route in ("skip_duplicate", "skip_already_organized", "left_untouched", "error"):
        return False
    if item.route == "review":
        return item.user_decision == "confirmed"
    if item.user_decision == "skipped":
        return False
    return item.route in ("auto", "unsorted", "other")


def _copy_and_verify(source: Path, dest: Path, expected_sha256: str) -> str:
    """Copies to a same-directory temp file, verifies its hash against the source, then
    atomically renames into place - but only if dest still doesn't exist. Naming.py already
    resolves collisions at plan time, but this is a deliberate defense-in-depth check: a
    plan is reviewed before Apply, and disk state can change in between (a concurrent run, or
    a file the user created at that exact path in the meantime). Invariant #1 is "never
    overwrite - never", so it is enforced again here, not just trusted from an earlier stage."""
    if dest.exists():
        raise OrganizerError(
            f"Refusing to overwrite: '{dest}' already exists (it did not when this run was planned)."
        )

    tmp = dest.with_name(f"{dest.name}.nomia-tmp-{uuid.uuid4().hex[:8]}")
    try:
        shutil.copy2(source, tmp)
        actual = hash_file(tmp)
        if actual != expected_sha256:
            raise OrganizerError(
                f"Hash mismatch copying {source} -> {dest}: expected {expected_sha256}, got {actual}"
            )
        if dest.exists():  # re-check immediately before the final rename closes the race window
            raise OrganizerError(
                f"Refusing to overwrite: '{dest}' was created concurrently while copying."
            )
        os.replace(tmp, dest)
        return actual
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _remove_source_with_retry(source: Path) -> None:
    last_exc: OSError | None = None
    for attempt in range(2):
        try:
            source.unlink()
            return
        except OSError as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.5)
    raise OrganizerError(f"Could not remove source file {source} after retry: {last_exc}")


def _apply_single_item(item: ItemRecord, cfg: NomiaConfig, destination_root: Path, journal: UndoJournal) -> None:
    source = Path(item.source_path)
    dest_name = item.effective_dest_name()
    dest_rel = Path(item.dest_path).parent / dest_name if dest_name else Path(item.dest_path)
    dest = destination_root / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    dest_sha256 = _copy_and_verify(source, dest, item.source_sha256)

    dump_rel_str: str | None = None
    dump_sha256: str | None = None
    if item.dump_path:
        dump_rel = Path(item.dump_path)
        dump_dest = destination_root / dump_rel
        dump_dest.parent.mkdir(parents=True, exist_ok=True)
        dump_sha256 = _copy_and_verify(source, dump_dest, item.source_sha256)
        dump_rel_str = str(dump_rel)

    warning = None
    if not cfg.preserve_source:
        try:
            _remove_source_with_retry(source)
        except OrganizerError as exc:
            warning = str(exc)
            logger.warning("Applied '%s' successfully but could not remove the source file: %s", source, exc)

    journal.mark_applied(
        item.item_id, dest_relative_path=str(dest_rel), dest_sha256=dest_sha256,
        dump_relative_path=dump_rel_str, dump_sha256=dump_sha256,
        source_preserved=cfg.preserve_source, warning=warning,
    )


def apply_plan(run_id: str, cfg: NomiaConfig, journal: UndoJournal) -> ApplyResult:
    run = journal.get_run(run_id)
    if run is None:
        raise OrganizerError(f"No such run: {run_id}")

    destination_root = Path(run.destination_root)
    journal.set_run_status(run_id, "applying")

    items = journal.get_items(run_id, status="planned")
    applied = failed = skipped = 0

    for item in items:
        if not _should_include_in_apply(item):
            journal.mark_skipped(item.item_id, "not_included_for_apply")
            skipped += 1
            continue

        journal.mark_in_progress(item.item_id)
        try:
            _apply_single_item(item, cfg, destination_root, journal)
            applied += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must never abort the whole batch
            logger.error("Failed to apply item %s (%s): %s", item.item_id, item.source_path, exc)
            journal.mark_failed(item.item_id, str(exc))
            failed += 1

    verification = verify_run(run_id, journal)

    if failed == 0 and verification.ok:
        final_status = "applied"
    elif applied == 0:
        final_status = "failed"
    elif not verification.ok:
        final_status = "applied_with_verification_errors"
    else:
        final_status = "partially_applied"

    journal.set_run_status(run_id, final_status, finished=True)
    return ApplyResult(run_id=run_id, applied=applied, failed=failed, skipped=skipped, verification=verification)


# --------------------------------------------------------------------------------------------
# Undo
# --------------------------------------------------------------------------------------------

def _undo_single_item(item: ItemRecord, destination_root: Path, journal: UndoJournal) -> tuple[bool, str | None]:
    if not item.dest_path:
        return False, "dest_missing"
    dest = destination_root / item.dest_path
    if not dest.exists():
        journal.mark_undo_skipped(item.item_id, "dest_missing")
        return False, "dest_missing"

    if hash_file(dest) != item.dest_sha256:
        journal.mark_undo_skipped(item.item_id, "dest_modified_since_apply")
        return False, "dest_modified_since_apply"

    dump_path = destination_root / item.dump_path if item.dump_path else None

    if item.source_preserved:
        dest.unlink(missing_ok=True)
        if dump_path and dump_path.exists():
            dump_path.unlink(missing_ok=True)
        journal.mark_undone(item.item_id)
        return True, None

    source = Path(item.source_path)
    if source.exists():
        journal.mark_undo_skipped(item.item_id, "source_path_now_occupied")
        return False, "source_path_now_occupied"

    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dest), str(source))
    if hash_file(source) != item.dest_sha256:
        journal.mark_undo_skipped(item.item_id, "verification_failed_after_move_back")
        return False, "verification_failed_after_move_back"

    if dump_path and dump_path.exists():
        dump_path.unlink(missing_ok=True)

    journal.mark_undone(item.item_id)
    return True, None


def undo_run(run_id: str, journal: UndoJournal) -> UndoResult:
    run = journal.get_run(run_id)
    if run is None:
        raise OrganizerError(f"No such run: {run_id}")
    destination_root = Path(run.destination_root)

    items = journal.get_items(run_id, status="applied")
    items_sorted = sorted(items, key=lambda i: i.applied_at or "", reverse=True)

    undone = skipped = 0
    details: list[dict[str, Any]] = []
    for item in items_sorted:
        ok, reason = _undo_single_item(item, destination_root, journal)
        if ok:
            undone += 1
        else:
            skipped += 1
            details.append({"item_id": item.item_id, "source_path": item.source_path, "reason": reason})

    journal.set_run_status(run_id, "undone" if skipped == 0 else "partially_undone", finished=True)
    return UndoResult(run_id=run_id, undone=undone, skipped=skipped, details=details)


# --------------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------------

def verify_run(run_id: str, journal: UndoJournal) -> VerificationReport:
    """Count reconciliation (every scanned file accounted for by final status) and hash
    reconciliation (destination and dump bytes re-read from disk right now and compared
    against the source hash recorded at scan time). Runs automatically at the end of
    apply_plan(); also independently callable to re-check a run later."""
    run = journal.get_run(run_id)
    if run is None:
        raise OrganizerError(f"No such run: {run_id}")
    destination_root = Path(run.destination_root)

    items = journal.get_items(run_id)
    scanned_count = run.totals_json.get("scanned_count", len(items))

    counts_by_status: dict[str, int] = {}
    for item in items:
        counts_by_status[item.status] = counts_by_status.get(item.status, 0) + 1
    accounted_count = sum(counts_by_status.values())

    hash_matches = 0
    mismatches: list[dict[str, Any]] = []
    for item in items:
        if item.status != "applied":
            continue
        problems: list[tuple[str, str | None]] = []

        if item.dest_path:
            dest = destination_root / item.dest_path
            if not dest.exists():
                problems.append(("dest_missing", None))
            else:
                actual = hash_file(dest)
                if actual != item.source_sha256:
                    problems.append(("dest_hash_mismatch", actual))

        if item.dump_path:
            dump_p = destination_root / item.dump_path
            if not dump_p.exists():
                problems.append(("dump_missing", None))
            else:
                actual = hash_file(dump_p)
                if actual != item.source_sha256:
                    problems.append(("dump_hash_mismatch", actual))

        if problems:
            for kind, actual in problems:
                mismatches.append({
                    "item_id": item.item_id, "source_path": item.source_path,
                    "dest_path": item.dest_path, "dump_path": item.dump_path,
                    "kind": kind, "expected": item.source_sha256, "actual": actual,
                })
        else:
            hash_matches += 1

    report = VerificationReport(
        scanned_count=scanned_count, accounted_count=accounted_count,
        counts_by_status=counts_by_status, hash_matches=hash_matches, hash_mismatches=mismatches,
    )
    journal.set_run_verification(run_id, report)
    if mismatches:
        logger.error("Verification found %d hash mismatch(es) for run %s", len(mismatches), run_id)
    if scanned_count != accounted_count:
        logger.error(
            "Verification count mismatch for run %s: scanned %d, accounted for %d",
            run_id, scanned_count, accounted_count,
        )

    # verify_run is independently re-runnable (e.g. `nomia verify <run_id>` later, to catch
    # bit-rot) - apply_plan() sets the run's final status itself right after calling this, but
    # a standalone re-check that finds a *new* problem still needs to be reflected in the run's
    # stored status, not just in verification_json. Only downgrades a plain "applied" run; a
    # status that already reflects a more specific problem (partially_applied, failed, ...) is
    # left alone rather than papered over.
    if not report.ok and run.status == "applied":
        journal.set_run_status(run_id, "applied_with_verification_errors")

    return report


# --------------------------------------------------------------------------------------------
# Crash resume
# --------------------------------------------------------------------------------------------

def resume_crashed_run(run_id: str, journal: UndoJournal) -> ResumeResult:
    """Finds items stuck at 'in_progress' after an unclean shutdown, checks filesystem reality
    (dest exists and hashes correctly -> finalize as applied; otherwise -> failed, and the
    source is guaranteed intact since it is only ever removed after a verified destination
    write). Also sweeps orphaned .nomia-tmp-* staging files anywhere under the destination."""
    run = journal.get_run(run_id)
    if run is None:
        raise OrganizerError(f"No such run: {run_id}")

    destination_root = Path(run.destination_root)
    stuck_items = journal.get_items(run_id, status="in_progress")
    finalized = failed = 0

    for item in stuck_items:
        dest_abs = destination_root / item.dest_path if item.dest_path else None
        dest_ok = dest_abs is not None and dest_abs.exists() and hash_file(dest_abs) == item.source_sha256
        dump_ok = True
        if item.dump_path:
            dump_abs = destination_root / item.dump_path
            dump_ok = dump_abs.exists() and hash_file(dump_abs) == item.source_sha256

        if dest_ok and dump_ok:
            journal.mark_applied(
                item.item_id, dest_relative_path=item.dest_path, dest_sha256=item.source_sha256,
                dump_relative_path=item.dump_path, dump_sha256=(item.source_sha256 if item.dump_path else None),
                source_preserved=item.source_preserved,
            )
            finalized += 1
        else:
            journal.mark_failed(item.item_id, "interrupted_before_verified_write")
            failed += 1

    if destination_root.exists():
        for tmp_file in destination_root.rglob("*.nomia-tmp-*"):
            try:
                tmp_file.unlink()
            except OSError:
                pass

    journal.set_run_status(run_id, "partially_applied" if failed else "applied")
    return ResumeResult(run_id=run_id, finalized=finalized, failed=failed)

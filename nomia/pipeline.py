"""Orchestrates scan -> extract (parallel) -> classify (queued against one Ollama worker) ->
naming into a dry-run Plan. Nothing in this module mutates disk - it only writes 'planned' rows
to the undo journal. organizer.apply_plan() is a separate, explicit step (see CLAUDE.md's
dry-run-first invariant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

from nomia.classify import ClassificationOutcome, classify_file, check_model_available
from nomia.config import NomiaConfig
from nomia.errors import ModelNotAvailableError
from nomia.extract import extract_all
from nomia.naming import (
    DestinationIndex,
    NamingCandidate,
    dump_relative_path,
    other_relative_path,
    plan_organized_names,
    resolve_collision,
    unsorted_relative_path,
)
from nomia.organizer import UndoJournal
from nomia.scanner import FileRecord, dedupe_by_hash, scan

logger = logging.getLogger(__name__)

Route = Literal[
    "auto", "review", "unsorted", "other", "left_untouched", "skip_duplicate", "skip_already_organized",
]
Stage = Literal["scanning", "extracting", "classifying", "naming", "done"]


class ProgressEvent(BaseModel):
    stage: Stage
    done: int
    total: int


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class PlannedItem:
    record: FileRecord
    route: Route
    dest_relative_path: Path | None
    dump_relative_path: Path | None
    category: str | None = None
    subcategory: str | None = None
    description: str | None = None
    confidence: float | None = None
    reason: str | None = None
    raw_model_response: str | None = None
    naming_index: int | None = None
    naming_date_source: str | None = None
    error: str | None = None


@dataclass
class Plan:
    run_id: str
    items: list[PlannedItem]
    summary: dict[str, int]


def _notify(cb: ProgressCallback | None, stage: Stage, done: int, total: int) -> None:
    if cb is not None:
        cb(ProgressEvent(stage=stage, done=done, total=total))


def _stable_sort_key(item: PlannedItem) -> str:
    return str(item.record.path).lower()


def build_plan(
    cfg: NomiaConfig,
    journal: UndoJournal,
    *,
    max_extract_workers: int = 8,
    progress_cb: ProgressCallback | None = None,
    run_id: str | None = None,
) -> Plan:
    """`run_id`: pass a pre-generated id when the caller needs to know it before this
    (potentially slow) call returns - e.g. server.py registers a progress-polling job under
    that id before kicking this off in a background thread. Left as None for the CLI, which
    only needs the id once the whole call has finished anyway."""
    if not cfg.source_folders:
        raise ValueError("No source folders configured.")
    if not cfg.destination_root:
        raise ValueError("No destination folder configured.")

    source_folders = [Path(p) for p in cfg.source_folders]
    destination_root = Path(cfg.destination_root)

    run_id = journal.start_run(
        source_folders=[str(p) for p in source_folders],
        destination_root=str(destination_root),
        naming_template=cfg.active_naming_template(),
        model_used=cfg.model.active_model,
        thresholds=cfg.thresholds.model_dump(),
        config_snapshot=cfg.model_dump(mode="json"),
        run_id=run_id,
    )

    try:
        return _build_plan_inner(
            run_id, cfg, journal, source_folders, destination_root,
            max_extract_workers=max_extract_workers, progress_cb=progress_cb,
        )
    except Exception:
        # Nothing has touched disk yet at this stage (dry-run only), so there is nothing to
        # roll back - but the run row should reflect reality rather than being left stuck at
        # "planning" forever.
        journal.set_run_status(run_id, "failed", finished=True)
        raise


def _build_plan_inner(
    run_id: str,
    cfg: NomiaConfig,
    journal: UndoJournal,
    source_folders: list[Path],
    destination_root: Path,
    *,
    max_extract_workers: int,
    progress_cb: ProgressCallback | None,
) -> Plan:
    _notify(progress_cb, "scanning", 0, 0)
    records = scan(source_folders, exclude_dirs=[destination_root])
    records = dedupe_by_hash(records)
    journal.set_run_totals(run_id, {"scanned_count": len(records)})
    _notify(progress_cb, "scanning", len(records), len(records))

    dest_index = DestinationIndex(destination_root)
    items: list[PlannedItem] = []
    to_process: list[FileRecord] = []

    for record in records:
        if record.is_duplicate_of is not None:
            items.append(PlannedItem(
                record=record, route="skip_duplicate", dest_relative_path=None, dump_relative_path=None,
                error=f"duplicate_of:{record.is_duplicate_of}",
            ))
            continue

        prior = journal.find_applied_by_source_path(str(record.path))
        if prior is not None:
            items.append(PlannedItem(
                record=record, route="skip_already_organized",
                dest_relative_path=Path(prior.dest_path) if prior.dest_path else None,
                dump_relative_path=None, error="already_organized_in_a_previous_run",
            ))
            continue

        to_process.append(record)

    # Fail fast: if there's anything left to classify, confirm the model is actually available
    # before spending time on extraction (which can be slow for many/large PDFs) only to
    # discover the model problem on file 50 of 200.
    if to_process and not check_model_available(cfg.model.active_model, host=cfg.model.ollama_host):
        raise ModelNotAvailableError(cfg.model.active_model)

    _notify(progress_cb, "extracting", 0, len(to_process))
    signals_list = extract_all(to_process, cfg, max_workers=max_extract_workers)
    signals_by_path = {s.path: s for s in signals_list}
    _notify(progress_cb, "extracting", len(to_process), len(to_process))

    naming_candidates: list[NamingCandidate] = []
    outcome_by_path: dict[Path, ClassificationOutcome] = {}
    special_items: list[PlannedItem] = []

    for i, record in enumerate(to_process):
        _notify(progress_cb, "classifying", i, len(to_process))
        signals = signals_by_path.get(record.path)

        if signals is None:
            special_items.append(PlannedItem(
                record=record, route="unsorted", dest_relative_path=None, dump_relative_path=None,
                error="extraction_missing",
            ))
            continue

        if signals.media_type == "unsupported":
            route: Route = "other" if cfg.sweep_other_files else "left_untouched"
            special_items.append(PlannedItem(record=record, route=route, dest_relative_path=None, dump_relative_path=None))
            continue

        if signals.error is not None:
            special_items.append(PlannedItem(
                record=record, route="unsorted", dest_relative_path=None, dump_relative_path=None,
                error=signals.error,
            ))
            continue

        outcome = classify_file(signals, cfg)

        # "unsorted" covers two distinct cases that both bypass the naming template and land in
        # _Unsorted/ with the sanitized original filename: a low-confidence-but-valid
        # classification (outcome.result is set, confidence just didn't clear the threshold),
        # and a failed classification (bad/unparseable model output, outcome.result is None).
        # Only "auto" and "review" actually go through the naming template.
        if outcome.route in ("failed", "unsorted") or outcome.result is None:
            special_items.append(PlannedItem(
                record=record, route="unsorted", dest_relative_path=None, dump_relative_path=None,
                error=outcome.error, raw_model_response=outcome.raw_response,
                category=(outcome.result.category if outcome.result else None),
                subcategory=(outcome.result.subcategory if outcome.result else None),
                description=(outcome.result.description if outcome.result else None),
                confidence=(outcome.result.confidence if outcome.result else None),
                reason=(outcome.result.reason if outcome.result else None),
            ))
            continue

        outcome_by_path[record.path] = outcome
        naming_candidates.append(NamingCandidate(
            record=record, category_key=outcome.result.category, subcategory=outcome.result.subcategory,
            raw_description=outcome.result.description, confidence=outcome.result.confidence,
        ))

    _notify(progress_cb, "classifying", len(to_process), len(to_process))

    _notify(progress_cb, "naming", 0, len(naming_candidates))
    named_items = plan_organized_names(naming_candidates, cfg, dest_index)
    for named in named_items:
        record = named.candidate.record
        outcome = outcome_by_path[record.path]
        result = outcome.result
        dump_rel = resolve_collision(dump_relative_path(record.path.name), dest_index) if cfg.keep_dump_copies else None
        items.append(PlannedItem(
            record=record, route=outcome.route, dest_relative_path=named.dest_relative_path,
            dump_relative_path=dump_rel, category=result.category, subcategory=result.subcategory,
            description=result.description, confidence=result.confidence, reason=result.reason,
            raw_model_response=outcome.raw_response, naming_index=named.naming_index,
            naming_date_source=named.naming_date_source,
        ))
    _notify(progress_cb, "naming", len(named_items), len(named_items))

    # Special-bucket items (unsorted/other) still need collision-safe destination paths, resolved
    # against the same shared DestinationIndex so nothing anywhere in this run can collide with
    # anything else. Processed in a stable, path-sorted order so suffix assignment (__2, __3, ...)
    # is deterministic across re-runs, matching the guarantee naming.py provides internally.
    for item in sorted(special_items, key=_stable_sort_key):
        if item.route == "unsorted":
            item.dest_relative_path = resolve_collision(unsorted_relative_path(item.record.path.name), dest_index)
        elif item.route == "other":
            item.dest_relative_path = resolve_collision(other_relative_path(item.record.path.name), dest_index)
        if item.route in ("unsorted", "other") and cfg.keep_dump_copies:
            item.dump_relative_path = resolve_collision(dump_relative_path(item.record.path.name), dest_index)
        items.append(item)

    for item in items:
        journal.record_planned(
            run_id,
            source_path=str(item.record.path),
            source_sha256=item.record.sha256,
            source_size=item.record.size,
            dest_path=str(item.dest_relative_path) if item.dest_relative_path else None,
            dump_path=str(item.dump_relative_path) if item.dump_relative_path else None,
            route=item.route,
            category=item.category,
            subcategory=item.subcategory,
            description=item.description,
            confidence=item.confidence,
            reason=item.reason,
            raw_model_response=item.raw_model_response,
            duplicate_of=str(item.record.is_duplicate_of) if item.record.is_duplicate_of else None,
            naming_index=item.naming_index,
            naming_date_source=item.naming_date_source,
            error=item.error,
        )

    summary: dict[str, int] = {}
    for item in items:
        summary[item.route] = summary.get(item.route, 0) + 1

    journal.set_run_status(run_id, "planned")
    _notify(progress_cb, "done", len(items), len(items))
    logger.info("Plan %s built: %s", run_id, summary)
    return Plan(run_id=run_id, items=items, summary=summary)

"""FastAPI wrapper over pipeline.py/organizer.py/report.py - no business logic lives here, only
request/response shaping and async job orchestration. Serves the static web/ UI at "/" and the
API under "/api/*" from the same process, so no CORS setup is needed (this is a single-user
local app, not a multi-tenant service - see CLAUDE.md).
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nomia.classify import ModelsStatusReport, get_models_status
from nomia.config import NomiaConfig, load_config, save_config
from nomia.errors import ModelNotAvailableError, NomiaError
from nomia.extract import extract_signals
from nomia.logging_setup import configure_logging
from nomia.naming import NamingContext, resolve_template
from nomia.organizer import (
    ApplyResult,
    ItemRecord,
    UndoJournal,
    UndoResult,
    VerificationReport,
    apply_plan as organizer_apply_plan,
    resume_crashed_run,
    undo_run as organizer_undo_run,
    verify_run as organizer_verify_run,
)
from nomia.paths import journal_db_path, reports_dir, thumbnails_dir
from nomia.pipeline import Plan, ProgressEvent, build_plan
from nomia.report import generate_report
from nomia.scanner import FileRecord

logger = logging.getLogger(__name__)

load_dotenv()
configure_logging()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Nomia")

_journal: UndoJournal | None = None
_journal_lock = threading.Lock()


def get_journal() -> UndoJournal:
    global _journal
    with _journal_lock:
        if _journal is None:
            _journal = UndoJournal(journal_db_path())
        return _journal


# --------------------------------------------------------------------------------------------
# In-process job registry for the async /api/scan and /api/apply endpoints. Explicit non-goal:
# no multi-run concurrency control beyond this - single-user local app, not a service (CLAUDE.md).
# --------------------------------------------------------------------------------------------

@dataclass
class JobState:
    kind: Literal["scan", "apply"]
    status: Literal["running", "done", "error"] = "running"
    stage: str | None = None
    done: int = 0
    total: int = 0
    error: str | None = None
    result: Any = None


_jobs: dict[str, JobState] = {}
_jobs_lock = threading.Lock()


def _set_job(run_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(run_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def _run_scan_job(run_id: str, cfg: NomiaConfig) -> None:
    journal = get_journal()

    def _progress(event: ProgressEvent) -> None:
        _set_job(run_id, stage=event.stage, done=event.done, total=event.total)

    try:
        plan = build_plan(cfg, journal, progress_cb=_progress, run_id=run_id)
        _set_job(run_id, status="done", result=plan.summary)
    except ModelNotAvailableError as exc:
        logger.error("Scan job %s failed: model not available (%s)", run_id, exc)
        _set_job(run_id, status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI rather than crashing the server
        logger.exception("Scan job %s failed: %s", run_id, exc)
        _set_job(run_id, status="error", error=str(exc))


def _run_apply_job(run_id: str) -> None:
    journal = get_journal()
    try:
        run = journal.get_run(run_id)
        if run is None:
            raise NomiaError(f"No such run: {run_id}")
        # Apply against the exact config that was active when this run was *planned*, not
        # whatever the live config happens to be now - the naming/dest paths and dump/preserve
        # toggles were all computed against that snapshot, so applying must stay consistent
        # with what the user actually previewed.
        cfg = NomiaConfig.model_validate(run.config_snapshot_json)
        result = organizer_apply_plan(run_id, cfg, journal)

        report_path = reports_dir() / f"run_{run_id}.xlsx"
        generate_report(run_id, journal, report_path)

        _set_job(run_id, status="done", result=result.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Apply job %s failed: %s", run_id, exc)
        _set_job(run_id, status="error", error=str(exc))


# --------------------------------------------------------------------------------------------
# DTOs
# --------------------------------------------------------------------------------------------

class PlannedItemDTO(BaseModel):
    item_id: int
    source_path: str
    thumbnail_url: str | None
    category: str | None
    subcategory: str | None
    description: str | None
    confidence: float | None
    route: str
    status: str
    reason: str | None
    proposed_name: str | None
    proposed_dest_path: str | None
    user_decision: str
    naming_index: int | None
    duplicate_of: str | None
    error: str | None
    warning: str | None


def _to_dto(item: ItemRecord) -> PlannedItemDTO:
    proposed_name = item.name_override or (Path(item.dest_path).name if item.dest_path else None)
    return PlannedItemDTO(
        item_id=item.item_id,
        source_path=item.source_path,
        thumbnail_url=f"/api/thumbnail/{item.item_id}",
        category=item.category,
        subcategory=item.subcategory,
        description=item.description,
        confidence=item.confidence,
        route=item.route,
        status=item.status,
        reason=item.reason,
        proposed_name=proposed_name,
        proposed_dest_path=item.dest_path,
        user_decision=item.user_decision,
        naming_index=item.naming_index,
        duplicate_of=item.duplicate_of,
        error=item.error,
        warning=item.warning,
    )


class PreviewResponse(BaseModel):
    run_id: str
    items: list[PlannedItemDTO]
    summary: dict[str, int]


class ScanRequest(BaseModel):
    source_folders: list[str] | None = None
    destination_root: str | None = None


class JobStatusResponse(BaseModel):
    run_id: str
    kind: str
    status: str
    stage: str | None = None
    done: int = 0
    total: int = 0
    error: str | None = None
    result: Any = None


class PatchItemRequest(BaseModel):
    user_decision: str | None = None
    name_override: str | None = None


class BulkPatchRequest(BaseModel):
    item_ids: list[int]
    user_decision: str


class ApplyRequest(BaseModel):
    confirm: bool = False


class ValidatePathRequest(BaseModel):
    path: str


class NamingPreviewRequest(BaseModel):
    template: str


# --------------------------------------------------------------------------------------------
# Health / models / config
# --------------------------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    cfg = load_config()
    status = get_models_status(cfg)
    return {"status": "ok", "ollama_reachable": status.ollama_reachable}


@app.get("/api/models/status", response_model=ModelsStatusReport)
def models_status() -> ModelsStatusReport:
    cfg = load_config()
    return get_models_status(cfg)


@app.get("/api/config", response_model=NomiaConfig)
def get_config() -> NomiaConfig:
    return load_config()


@app.put("/api/config", response_model=NomiaConfig)
def put_config(cfg: NomiaConfig) -> NomiaConfig:
    save_config(cfg)
    return cfg


@app.post("/api/validate-path")
def validate_path(req: ValidatePathRequest) -> dict:
    path = Path(req.path).expanduser()
    return {"exists": path.exists(), "is_dir": path.is_dir()}


@app.post("/api/naming/preview")
def naming_preview(req: NamingPreviewRequest) -> dict:
    """Renders the user's chosen template against a fixed example context, using the real
    naming.resolve_template - so the UI's live preview can never drift from actual behavior."""
    example_ctx = NamingContext(
        category="receipt", subcategory="grocery", description="costco-grocery-receipt",
        original_stem="scan001", index_str="01", date=datetime.now(), confidence=0.91,
    )
    try:
        rendered = resolve_template(req.template, example_ctx)
    except Exception as exc:  # noqa: BLE001 - a malformed custom template must not 500 the UI
        return {"example_filename": None, "error": str(exc)}
    return {"example_filename": f"{rendered}.pdf" if rendered else None}


# --------------------------------------------------------------------------------------------
# Scan (dry-run plan)
# --------------------------------------------------------------------------------------------

@app.post("/api/scan")
def start_scan(req: ScanRequest) -> dict:
    cfg = load_config()
    if req.source_folders is not None:
        cfg.source_folders = req.source_folders
    if req.destination_root is not None:
        cfg.destination_root = req.destination_root
    if req.source_folders is not None or req.destination_root is not None:
        save_config(cfg)

    if not cfg.source_folders or not cfg.destination_root:
        raise HTTPException(400, "Source folders and a destination root must be configured before scanning.")

    run_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[run_id] = JobState(kind="scan")

    thread = threading.Thread(target=_run_scan_job, args=(run_id, cfg), daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "planning"}


@app.get("/api/scan/{run_id}/status", response_model=JobStatusResponse)
def scan_status(run_id: str) -> JobStatusResponse:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None or job.kind != "scan":
        raise HTTPException(404, f"No scan job found for run_id {run_id}")
    return JobStatusResponse(run_id=run_id, kind=job.kind, status=job.status, stage=job.stage, done=job.done, total=job.total, error=job.error, result=job.result)


@app.get("/api/preview/{run_id}", response_model=PreviewResponse)
def get_preview(run_id: str) -> PreviewResponse:
    journal = get_journal()
    run = journal.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No such run: {run_id}")
    items = journal.get_items(run_id)
    summary: dict[str, int] = {}
    for item in items:
        summary[item.route] = summary.get(item.route, 0) + 1
    return PreviewResponse(run_id=run_id, items=[_to_dto(i) for i in items], summary=summary)


@app.get("/api/thumbnail/{item_id}")
def get_thumbnail(item_id: int) -> Response:
    journal = get_journal()
    item = journal.get_item(item_id)
    if item is None:
        raise HTTPException(404, "No such item")

    cache_path = thumbnails_dir() / f"{item_id}.jpg"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/jpeg")

    run = journal.get_run(item.run_id)
    source_path = Path(item.source_path)
    if not source_path.exists() and run is not None and item.dest_path:
        source_path = Path(run.destination_root) / item.dest_path
    if not source_path.exists():
        raise HTTPException(404, "Source file no longer available for a thumbnail")

    cfg = load_config()
    record = FileRecord(
        path=source_path, size=source_path.stat().st_size, sha256=item.source_sha256,
        created_at=None, modified_at=datetime.now(), ext=source_path.suffix.lower(),
        source_root=source_path.parent, discovery_seq=0,
    )
    signals = extract_signals(record, cfg)
    if signals.render_png is None:
        raise HTTPException(404, "Could not render a thumbnail for this file")

    try:
        from PIL import Image
        import io as _io

        with Image.open(_io.BytesIO(signals.render_png)) as image:
            image.thumbnail((200, 200))
            buf = _io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=80)
            thumbnail_bytes = buf.getvalue()
    except Exception as exc:  # noqa: BLE001 - a thumbnail is a nice-to-have, never fatal
        logger.warning("Could not build thumbnail for item %s: %s", item_id, exc)
        raise HTTPException(404, "Could not build a thumbnail for this file")

    thumbnails_dir().mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(thumbnail_bytes)
    return Response(content=thumbnail_bytes, media_type="image/jpeg")


@app.patch("/api/preview/{run_id}/items/{item_id}", response_model=PlannedItemDTO)
def patch_item(run_id: str, item_id: int, req: PatchItemRequest) -> PlannedItemDTO:
    journal = get_journal()
    item = journal.get_item(item_id)
    if item is None or item.run_id != run_id:
        raise HTTPException(404, "No such item for this run")
    journal.set_user_decision(item_id, user_decision=req.user_decision, name_override=req.name_override)
    return _to_dto(journal.get_item(item_id))


@app.patch("/api/preview/{run_id}/items:bulk")
def patch_items_bulk(run_id: str, req: BulkPatchRequest) -> dict:
    journal = get_journal()
    updated = 0
    for item_id in req.item_ids:
        item = journal.get_item(item_id)
        if item is not None and item.run_id == run_id:
            journal.set_user_decision(item_id, user_decision=req.user_decision)
            updated += 1
    return {"updated": updated}


# --------------------------------------------------------------------------------------------
# Apply / verify / undo / report
# --------------------------------------------------------------------------------------------

@app.post("/api/apply/{run_id}")
def start_apply(run_id: str, req: ApplyRequest) -> dict:
    if not req.confirm:
        raise HTTPException(400, "Apply requires confirm: true.")
    journal = get_journal()
    if journal.get_run(run_id) is None:
        raise HTTPException(404, f"No such run: {run_id}")

    with _jobs_lock:
        _jobs[run_id] = JobState(kind="apply")

    thread = threading.Thread(target=_run_apply_job, args=(run_id,), daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "applying"}


@app.get("/api/apply/{run_id}/status", response_model=JobStatusResponse)
def apply_status(run_id: str) -> JobStatusResponse:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None or job.kind != "apply":
        raise HTTPException(404, f"No apply job found for run_id {run_id}")

    result = job.result
    if result is not None:
        journal = get_journal()
        run = journal.get_run(run_id)
        if run is not None:
            result = {**result, "report_path": run.report_path, "verification": run.verification_json}
    return JobStatusResponse(run_id=run_id, kind=job.kind, status=job.status, stage=job.stage, done=job.done, total=job.total, error=job.error, result=result)


@app.post("/api/verify/{run_id}", response_model=VerificationReport)
def verify(run_id: str) -> VerificationReport:
    journal = get_journal()
    if journal.get_run(run_id) is None:
        raise HTTPException(404, f"No such run: {run_id}")
    return organizer_verify_run(run_id, journal)


@app.post("/api/resume/{run_id}")
def resume(run_id: str) -> dict:
    journal = get_journal()
    if journal.get_run(run_id) is None:
        raise HTTPException(404, f"No such run: {run_id}")
    result = resume_crashed_run(run_id, journal)
    return result.model_dump()


@app.get("/api/report/{run_id}")
def get_report(run_id: str) -> FileResponse:
    journal = get_journal()
    run = journal.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No such run: {run_id}")

    report_path = Path(run.report_path) if run.report_path else reports_dir() / f"run_{run_id}.xlsx"
    if not report_path.exists():
        generate_report(run_id, journal, report_path)

    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=report_path.name,
    )


@app.post("/api/undo/{run_id}", response_model=UndoResult)
def undo(run_id: str) -> UndoResult:
    journal = get_journal()
    if journal.get_run(run_id) is None:
        raise HTTPException(404, f"No such run: {run_id}")
    return organizer_undo_run(run_id, journal)


@app.get("/api/runs/last-applied")
def last_applied() -> dict:
    journal = get_journal()
    return {"run_id": journal.get_last_applied_run_id()}


# --------------------------------------------------------------------------------------------
# Static web UI - mounted last so it never shadows the /api/* routes above.
# --------------------------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

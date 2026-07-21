import hashlib
from pathlib import Path

import pytest

from nomia.config import NomiaConfig
from nomia.organizer import (
    UndoJournal,
    apply_plan,
    resume_crashed_run,
    undo_run,
    verify_run,
)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_source(tmp_path: Path, name: str, content: bytes = b"file-content") -> Path:
    src_dir = tmp_path / "source"
    src_dir.mkdir(exist_ok=True)
    path = src_dir / name
    path.write_bytes(content)
    return path


def _journal(tmp_path: Path) -> UndoJournal:
    return UndoJournal(tmp_path / "journal.sqlite3")


def _plan_one_item(
    journal: UndoJournal,
    tmp_path: Path,
    *,
    source_name: str = "receipt.pdf",
    content: bytes = b"file-content",
    dest_rel: str = "receipt/receipt_2024-01-01.pdf",
    dump_rel: str | None = "_dump/receipt.pdf",
    route: str = "auto",
    user_decision: str | None = None,
) -> tuple[str, int, Path, Path]:
    source = _make_source(tmp_path, source_name, content)
    dest_root = tmp_path / "dest"
    run_id = journal.start_run(
        source_folders=[str(tmp_path / "source")], destination_root=str(dest_root),
        naming_template="{category}_{yyyy}-{mm}-{dd}_{index}", model_used="moondream",
        thresholds={"auto_min": 0.8, "review_min": 0.5}, config_snapshot={},
    )
    item_id = journal.record_planned(
        run_id, source_path=str(source), source_sha256=_hash(content), source_size=len(content),
        dest_path=dest_rel, dump_path=dump_rel, route=route, category="receipt",
        description="receipt", confidence=0.9,
    )
    if user_decision:
        journal.set_user_decision(item_id, user_decision=user_decision)
    journal.set_run_totals(run_id, {"scanned_count": 1})
    return run_id, item_id, source, dest_root


# --------------------------------------------------------------------------------------------
# apply_plan
# --------------------------------------------------------------------------------------------

def test_apply_moves_file_and_removes_source(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)

    result = apply_plan(run_id, NomiaConfig(), journal)

    assert result.applied == 1
    assert result.failed == 0
    assert not source.exists()  # default behavior: source removed after verified copy
    dest_file = dest_root / "receipt" / "receipt_2024-01-01.pdf"
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"file-content"
    dump_file = dest_root / "_dump" / "receipt.pdf"
    assert dump_file.exists()
    assert dump_file.read_bytes() == b"file-content"

    item = journal.get_item(item_id)
    assert item.status == "applied"
    assert item.dest_sha256 == _hash(b"file-content")
    assert item.dump_sha256 == _hash(b"file-content")


def test_preserve_source_never_removes_source_file(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)

    cfg = NomiaConfig(preserve_source=True)
    result = apply_plan(run_id, cfg, journal)

    assert result.applied == 1
    assert source.exists()  # never removed
    assert source.read_bytes() == b"file-content"
    assert (dest_root / "receipt" / "receipt_2024-01-01.pdf").exists()

    item = journal.get_item(item_id)
    assert item.source_preserved is True


def test_apply_never_overwrites_preexisting_dest_file(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)

    # Simulate a file that appeared at the exact planned destination after planning finished.
    dest_file = dest_root / "receipt" / "receipt_2024-01-01.pdf"
    dest_file.parent.mkdir(parents=True)
    dest_file.write_bytes(b"someone-elses-content")

    result = apply_plan(run_id, NomiaConfig(), journal)

    assert result.applied == 0
    assert result.failed == 1
    assert dest_file.read_bytes() == b"someone-elses-content"  # untouched
    assert source.exists()  # source never removed since the copy never succeeded
    item = journal.get_item(item_id)
    assert item.status == "failed"


def test_review_route_requires_explicit_confirmation(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path, route="review")

    result = apply_plan(run_id, NomiaConfig(), journal)

    assert result.applied == 0
    assert result.skipped == 1
    assert source.exists()
    assert journal.get_item(item_id).status == "skipped"


def test_review_route_applies_once_confirmed(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path, route="review", user_decision="confirmed")

    result = apply_plan(run_id, NomiaConfig(), journal)

    assert result.applied == 1
    assert not source.exists()


def test_user_skip_excludes_even_an_auto_routed_item(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path, route="auto", user_decision="skipped")

    result = apply_plan(run_id, NomiaConfig(), journal)

    assert result.applied == 0
    assert result.skipped == 1
    assert source.exists()


def test_unsorted_route_is_applied_automatically(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(
        journal, tmp_path, dest_rel="_Unsorted/receipt.pdf", dump_rel=None, route="unsorted",
    )

    result = apply_plan(run_id, NomiaConfig(), journal)

    assert result.applied == 1
    assert (dest_root / "_Unsorted" / "receipt.pdf").exists()


# --------------------------------------------------------------------------------------------
# verify_run
# --------------------------------------------------------------------------------------------

def test_verify_run_clean_apply_has_no_mismatches(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)
    apply_plan(run_id, NomiaConfig(), journal)

    report = verify_run(run_id, journal)

    assert report.ok
    assert report.hash_matches == 1
    assert report.hash_mismatches == []
    assert report.scanned_count == report.accounted_count == 1


def test_verify_run_detects_corrupted_destination_file(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)
    apply_plan(run_id, NomiaConfig(), journal)

    dest_file = dest_root / "receipt" / "receipt_2024-01-01.pdf"
    dest_file.write_bytes(b"corrupted-after-the-fact")

    report = verify_run(run_id, journal)

    assert not report.ok
    assert len(report.hash_mismatches) == 1
    assert report.hash_mismatches[0]["kind"] == "dest_hash_mismatch"

    run = journal.get_run(run_id)
    assert run.status == "applied_with_verification_errors"


# --------------------------------------------------------------------------------------------
# undo_run
# --------------------------------------------------------------------------------------------

def test_undo_restores_source_and_removes_dest_and_dump(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)
    apply_plan(run_id, NomiaConfig(), journal)

    result = undo_run(run_id, journal)

    assert result.undone == 1
    assert result.skipped == 0
    assert source.exists()
    assert source.read_bytes() == b"file-content"
    assert not (dest_root / "receipt" / "receipt_2024-01-01.pdf").exists()
    assert not (dest_root / "_dump" / "receipt.pdf").exists()
    assert journal.get_item(item_id).status == "undone"


def test_undo_skips_and_flags_modified_destination(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)
    apply_plan(run_id, NomiaConfig(), journal)

    dest_file = dest_root / "receipt" / "receipt_2024-01-01.pdf"
    dest_file.write_bytes(b"user-edited-this-file-since-apply")

    result = undo_run(run_id, journal)

    assert result.undone == 0
    assert result.skipped == 1
    assert not source.exists()  # never restored - we don't know if the edited content matters
    assert dest_file.read_bytes() == b"user-edited-this-file-since-apply"  # untouched
    item = journal.get_item(item_id)
    assert item.status == "undo_skipped_modified"


def test_undo_with_preserved_source_only_removes_copies(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)
    apply_plan(run_id, NomiaConfig(preserve_source=True), journal)
    assert source.exists()

    result = undo_run(run_id, journal)

    assert result.undone == 1
    assert source.exists()  # was never touched, and undo doesn't touch it either
    assert source.read_bytes() == b"file-content"
    assert not (dest_root / "receipt" / "receipt_2024-01-01.pdf").exists()


# --------------------------------------------------------------------------------------------
# resume_crashed_run
# --------------------------------------------------------------------------------------------

def test_resume_finalizes_item_whose_copy_actually_succeeded(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path, dump_rel=None)

    # Simulate a crash right after the destination file was written and verified, but before
    # the journal was updated to 'applied' (i.e. still sitting at 'in_progress').
    dest_file = dest_root / "receipt" / "receipt_2024-01-01.pdf"
    dest_file.parent.mkdir(parents=True)
    dest_file.write_bytes(b"file-content")
    journal.mark_in_progress(item_id)

    result = resume_crashed_run(run_id, journal)

    assert result.finalized == 1
    assert result.failed == 0
    assert journal.get_item(item_id).status == "applied"


def test_resume_marks_failed_when_copy_never_completed(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path, dump_rel=None)

    # Simulate a crash before the destination file was ever written.
    journal.mark_in_progress(item_id)

    result = resume_crashed_run(run_id, journal)

    assert result.finalized == 0
    assert result.failed == 1
    assert source.exists()  # source is guaranteed intact - it's only removed after a verified write
    assert journal.get_item(item_id).status == "failed"


def test_resume_sweeps_orphaned_temp_files(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path, dump_rel=None)
    journal.mark_in_progress(item_id)

    orphan = dest_root / "receipt" / "receipt_2024-01-01.pdf.nomia-tmp-deadbeef"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"partial")

    resume_crashed_run(run_id, journal)

    assert not orphan.exists()


def test_find_interrupted_runs_and_last_applied(tmp_path):
    journal = _journal(tmp_path)
    run_id, item_id, source, dest_root = _plan_one_item(journal, tmp_path)

    assert journal.get_last_applied_run_id() is None
    apply_plan(run_id, NomiaConfig(), journal)
    assert journal.get_last_applied_run_id() == run_id

    journal.set_run_status(run_id, "applying")  # simulate a crash mid-apply on a second run
    assert run_id in journal.find_interrupted_runs()

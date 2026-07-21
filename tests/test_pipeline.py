from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from nomia.config import NomiaConfig
from nomia.errors import ModelNotAvailableError
from nomia.organizer import UndoJournal
from nomia.pipeline import build_plan


def _cfg(tmp_path: Path, **overrides) -> NomiaConfig:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir(exist_ok=True)
    cfg = NomiaConfig(source_folders=[str(source)], destination_root=str(dest))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _journal(tmp_path: Path) -> UndoJournal:
    return UndoJournal(tmp_path / "journal.sqlite3")


def _fake_chat_response(category="receipt", description="costco-receipt", confidence=0.9):
    content = (
        f'{{"category": "{category}", "subcategory": null, "description": "{description}", '
        f'"reason": "looks like a receipt", "confidence": {confidence}}}'
    )
    return SimpleNamespace(message=SimpleNamespace(content=content))


def test_build_plan_classifies_and_names_a_simple_image(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    (Path(cfg.source_folders[0]) / "photo.jpg").write_bytes(b"")
    Image.new("RGB", (20, 20)).save(Path(cfg.source_folders[0]) / "photo.jpg")

    mocker.patch("nomia.classify.check_model_available", return_value=True)
    mocker.patch("ollama.Client.chat", return_value=_fake_chat_response())

    journal = _journal(tmp_path)
    plan = build_plan(cfg, journal)

    assert plan.summary.get("auto") == 1
    item = plan.items[0]
    assert item.category == "receipt"
    assert item.dest_relative_path is not None
    assert str(item.dest_relative_path).endswith(".jpg")

    # keep_dump_copies defaults to True.
    assert item.dump_relative_path is not None

    journal_items = journal.get_items(plan.run_id)
    assert len(journal_items) == 1
    assert journal_items[0].status == "planned"


def test_build_plan_routes_low_confidence_to_unsorted(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    Image.new("RGB", (20, 20)).save(Path(cfg.source_folders[0]) / "ambiguous.jpg")

    mocker.patch("nomia.classify.check_model_available", return_value=True)
    mocker.patch("ollama.Client.chat", return_value=_fake_chat_response(confidence=0.2))

    plan = build_plan(cfg, _journal(tmp_path))

    assert plan.summary.get("unsorted") == 1
    assert str(plan.items[0].dest_relative_path).startswith("_Unsorted")


def test_build_plan_detects_duplicate_content(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    src = Path(cfg.source_folders[0])
    (src / "a.txt").write_bytes(b"identical-content")
    (src / "b.txt").write_bytes(b"identical-content")

    plan = build_plan(cfg, _journal(tmp_path))

    routes = sorted(item.route for item in plan.items)
    # Both are unsupported (.txt) by default, but one is also a content-duplicate of the other.
    assert routes.count("skip_duplicate") == 1


def test_build_plan_leaves_unsupported_files_untouched_by_default(tmp_path):
    cfg = _cfg(tmp_path)
    (Path(cfg.source_folders[0]) / "notes.txt").write_text("hello")

    plan = build_plan(cfg, _journal(tmp_path))

    assert plan.summary.get("left_untouched") == 1
    assert plan.items[0].dest_relative_path is None


def test_build_plan_sweeps_unsupported_files_when_enabled(tmp_path):
    cfg = _cfg(tmp_path, sweep_other_files=True)
    (Path(cfg.source_folders[0]) / "notes.txt").write_text("hello")

    plan = build_plan(cfg, _journal(tmp_path))

    assert plan.summary.get("other") == 1
    assert str(plan.items[0].dest_relative_path).startswith("_Other")


def test_build_plan_routes_corrupt_file_to_unsorted_with_error(tmp_path):
    cfg = _cfg(tmp_path)
    (Path(cfg.source_folders[0]) / "broken.jpg").write_bytes(b"not a real jpeg")

    plan = build_plan(cfg, _journal(tmp_path))

    assert plan.summary.get("unsorted") == 1
    assert plan.items[0].error == "corrupt"


def test_build_plan_raises_when_model_unavailable(tmp_path, mocker):
    cfg = _cfg(tmp_path)
    Image.new("RGB", (10, 10)).save(Path(cfg.source_folders[0]) / "photo.jpg")
    mocker.patch("nomia.pipeline.check_model_available", return_value=False)

    journal = _journal(tmp_path)
    try:
        build_plan(cfg, journal)
        assert False, "expected ModelNotAvailableError"
    except ModelNotAvailableError:
        pass

    runs = journal.find_interrupted_runs()  # shouldn't be stuck at 'applying', just confirm it exists
    run_id = next(iter(journal._conn.execute("SELECT run_id FROM runs").fetchall()))["run_id"]
    assert journal.get_run(run_id).status == "failed"


def test_idempotency_skips_previously_organized_source_path(tmp_path, mocker):
    cfg = _cfg(tmp_path, preserve_source=True)
    photo = Path(cfg.source_folders[0]) / "photo.jpg"
    Image.new("RGB", (10, 10)).save(photo)

    mocker.patch("nomia.classify.check_model_available", return_value=True)
    mocker.patch("ollama.Client.chat", return_value=_fake_chat_response())

    journal = _journal(tmp_path)
    first_plan = build_plan(cfg, journal)
    assert first_plan.summary.get("auto") == 1

    # Simulate that first_plan's item was actually applied (preserve_source left photo.jpg in place).
    item_id = journal.get_items(first_plan.run_id)[0].item_id
    journal.mark_applied(item_id, dest_relative_path="receipt/whatever.jpg", dest_sha256="x")

    second_plan = build_plan(cfg, journal)
    assert second_plan.summary.get("skip_already_organized") == 1
    # No Ollama call should have been needed for the second plan's only file.

from datetime import datetime
from pathlib import Path

from nomia.scanner import (
    FileRecord,
    dedupe_by_hash,
    effective_creation_date,
    hash_file,
    parse_date_from_filename,
    scan,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_scan_finds_all_files_and_hashes_them(tmp_path):
    _write(tmp_path / "a.txt", b"hello")
    _write(tmp_path / "sub" / "b.txt", b"world")

    records = scan([tmp_path])

    assert len(records) == 2
    names = {r.path.name for r in records}
    assert names == {"a.txt", "b.txt"}
    for r in records:
        assert r.sha256 == hash_file(r.path)
        assert r.is_duplicate_of is None


def test_scan_skips_os_junk_files(tmp_path):
    _write(tmp_path / ".DS_Store", b"junk")
    _write(tmp_path / "real.txt", b"content")

    records = scan([tmp_path])

    assert [r.path.name for r in records] == ["real.txt"]


def test_scan_skips_missing_source_folder(tmp_path):
    missing = tmp_path / "does_not_exist"
    records = scan([missing])
    assert records == []


def test_scan_exclude_dirs_prunes_destination_root(tmp_path):
    dest = tmp_path / "Organized"
    _write(dest / "already_placed.pdf", b"x")
    _write(tmp_path / "incoming.pdf", b"y")

    records = scan([tmp_path], exclude_dirs=[dest])

    assert [r.path.name for r in records] == ["incoming.pdf"]


def test_dedupe_by_hash_groups_identical_content(tmp_path):
    _write(tmp_path / "invoice.pdf", b"same-bytes")
    _write(tmp_path / "invoice (1).pdf", b"same-bytes")
    _write(tmp_path / "unique.pdf", b"different-bytes")

    records = scan([tmp_path])
    deduped = dedupe_by_hash(records)

    by_name = {r.path.name: r for r in deduped}
    assert by_name["unique.pdf"].is_duplicate_of is None
    # Exactly one of the two identical-content files should be flagged as a duplicate of the other.
    dup_flags = [by_name["invoice.pdf"].is_duplicate_of, by_name["invoice (1).pdf"].is_duplicate_of]
    assert dup_flags.count(None) == 1
    assert dup_flags.count(None) + sum(1 for d in dup_flags if d is not None) == 2


def test_dedupe_is_stable_across_reruns(tmp_path):
    _write(tmp_path / "a.pdf", b"dup")
    _write(tmp_path / "b.pdf", b"dup")

    first = {r.path.name: r.is_duplicate_of for r in dedupe_by_hash(scan([tmp_path]))}
    second = {r.path.name: r.is_duplicate_of for r in dedupe_by_hash(scan([tmp_path]))}

    assert first == second


def test_parse_date_from_filename_variants():
    assert parse_date_from_filename("IMG_20260720_receipt.jpg") == datetime(2026, 7, 20)
    assert parse_date_from_filename("Screenshot 2026-07-20 at noon.png") == datetime(2026, 7, 20)
    assert parse_date_from_filename("no_date_here.jpg") is None
    assert parse_date_from_filename("plan_9999-99-99.jpg") is None  # invalid calendar date


def test_effective_creation_date_fallback_chain(tmp_path):
    path = tmp_path / "IMG_20200101.jpg"
    path.write_bytes(b"x")

    # Tier 1: OS creation time present.
    record = FileRecord(
        path=path, size=1, sha256="x", created_at=datetime(2019, 1, 1),
        modified_at=datetime(2021, 1, 1), ext=".jpg", source_root=tmp_path, discovery_seq=0,
    )
    date, tier = effective_creation_date(record)
    assert (date, tier) == (datetime(2019, 1, 1), "os_creation")

    # Tier 2: no creation time, falls back to modified time.
    record2 = record.__class__(**{**record.__dict__, "created_at": None})
    date2, tier2 = effective_creation_date(record2)
    assert (date2, tier2) == (datetime(2021, 1, 1), "os_modified")

    # Tier 3: neither OS timestamp available, falls back to filename-parsed date.
    record3 = record.__class__(**{**record.__dict__, "created_at": None, "modified_at": None})
    date3, tier3 = effective_creation_date(record3)
    assert (date3, tier3) == (datetime(2020, 1, 1), "filename_parsed")

    # Tier 4: nothing available at all -> discovery order (None date).
    no_date_path = tmp_path / "no_date_here.jpg"
    record4 = record.__class__(**{**record.__dict__, "path": no_date_path, "created_at": None, "modified_at": None})
    date4, tier4 = effective_creation_date(record4)
    assert (date4, tier4) == (None, "discovery_order")

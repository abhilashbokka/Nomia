from datetime import datetime
from pathlib import Path

from nomia.config import NomiaConfig
from nomia.naming import (
    DestinationIndex,
    NamingCandidate,
    NamingContext,
    dump_relative_path,
    other_relative_path,
    plan_organized_names,
    resolve_collision,
    resolve_template,
    sanitize_original,
    slugify,
    truncate_filename,
    unsorted_relative_path,
)
from nomia.scanner import FileRecord


def _record(name: str, *, created_at=None, modified_at=None, path=None) -> FileRecord:
    p = path or Path(f"/source/{name}")
    return FileRecord(
        path=p, size=10, sha256="deadbeef", created_at=created_at,
        modified_at=modified_at or datetime(2024, 1, 1), ext=p.suffix.lower(),
        source_root=Path("/source"), discovery_seq=0,
    )


# --------------------------------------------------------------------------------------------
# slugify / sanitize_original
# --------------------------------------------------------------------------------------------

def test_slugify_basic():
    slug, log = slugify("Costco Grocery Receipt")
    assert slug == "costco-grocery-receipt"
    assert log == []


def test_slugify_transliterates_and_logs_non_ascii():
    slug, log = slugify("café résumé")
    assert slug == "cafe-resume"
    assert len(log) == 1


def test_slugify_truncates_long_text():
    slug, log = slugify("word " * 30, max_len=20)
    assert len(slug) <= 20
    assert any("truncated" in entry for entry in log)


def test_slugify_empty_input_falls_back():
    slug, log = slugify("!!!")
    assert slug == "untitled"
    assert log  # logged the fallback


def test_sanitize_original_strips_illegal_chars_preserves_case():
    assert sanitize_original('My:File*Name?.pdf') == "MyFileName.pdf"
    assert sanitize_original("Café Photo") == "Café Photo"  # Unicode preserved, not slugified


def test_truncate_filename_is_utf8_byte_safe():
    stem = "café" * 100  # multi-byte UTF-8 characters
    result = truncate_filename(stem, ".jpg", max_bytes=20)
    # Must decode cleanly - a naive byte slice could cut a multi-byte character in half.
    assert result.encode("utf-8")  # no UnicodeDecodeError
    assert len(result.encode("utf-8")) <= 20


# --------------------------------------------------------------------------------------------
# resolve_template
# --------------------------------------------------------------------------------------------

def test_resolve_template_category_date_index():
    ctx = NamingContext(category="receipt", subcategory=None, description="costco-receipt",
                         original_stem="scan001", index_str="01", date=datetime(2026, 7, 20), confidence=0.9)
    assert resolve_template("{category}_{yyyy}-{mm}-{dd}_{index}", ctx) == "receipt_2026-07-20_01"


def test_resolve_template_foldered_by_category_year():
    ctx = NamingContext(category="receipt", subcategory=None, description="costco-receipt",
                         original_stem="scan001", index_str=None, date=datetime(2026, 7, 20), confidence=0.9)
    assert resolve_template("{category}/{yyyy}/{description}", ctx) == "receipt/2026/costco-receipt"


def test_resolve_template_missing_date_drops_token_and_separator():
    ctx = NamingContext(category="receipt", subcategory=None, description="costco-receipt",
                         original_stem="scan001", index_str=None, date=None, confidence=0.9)
    result = resolve_template("{yyyy}-{mm}-{dd}_{description}", ctx)
    assert result == "costco-receipt"
    assert "None" not in result


def test_resolve_template_missing_subcategory_drops_cleanly():
    ctx = NamingContext(category="receipt", subcategory=None, description="costco-receipt",
                         original_stem="scan001", index_str=None, date=None, confidence=0.9)
    result = resolve_template("{category}_{subcategory}_{description}", ctx)
    assert result == "receipt_costco-receipt"


def test_resolve_template_keep_original_tag_category():
    ctx = NamingContext(category="receipt", subcategory=None, description="ignored",
                         original_stem="scan001", index_str=None, date=None, confidence=0.9)
    assert resolve_template("{original}__{category}", ctx) == "scan001__receipt"


# --------------------------------------------------------------------------------------------
# DestinationIndex / resolve_collision
# --------------------------------------------------------------------------------------------

def test_resolve_collision_no_conflict_passes_through(tmp_path):
    index = DestinationIndex(tmp_path)
    result = resolve_collision(Path("receipt_2026-07-20.pdf"), index)
    assert result == Path("receipt_2026-07-20.pdf")


def test_resolve_collision_seeds_from_existing_disk_files(tmp_path):
    (tmp_path / "receipt").mkdir()
    (tmp_path / "receipt" / "existing.pdf").write_bytes(b"x")

    index = DestinationIndex(tmp_path)
    result = resolve_collision(Path("receipt/existing.pdf"), index)
    assert result == Path("receipt/existing__2.pdf")


def test_resolve_collision_within_same_run(tmp_path):
    index = DestinationIndex(tmp_path)
    first = resolve_collision(Path("photo.jpg"), index)
    second = resolve_collision(Path("photo.jpg"), index)
    third = resolve_collision(Path("photo.jpg"), index)
    assert (first, second, third) == (Path("photo.jpg"), Path("photo__2.jpg"), Path("photo__3.jpg"))
    # Never overwrites: all three must be distinct paths.
    assert len({first, second, third}) == 3


# --------------------------------------------------------------------------------------------
# Copy-ordering / plan_organized_names
# --------------------------------------------------------------------------------------------

def test_copy_ordering_assigns_sequential_index_by_creation_date_ascending(tmp_path):
    cfg = NomiaConfig()
    cfg.naming_preset_key = "category_date_index"  # "{category}_{yyyy}-{mm}-{dd}_{index}"

    # Three "copies" scanned on the same calendar day (so they render to the same {yyyy}-{mm}-{dd}
    # base and must collide/group), but at different times, out of chronological order on disk.
    records = [
        _record("invoice copy.pdf", created_at=datetime(2024, 1, 1, 20, 0), path=Path("/source/invoice copy.pdf")),
        _record("invoice.pdf", created_at=datetime(2024, 1, 1, 8, 0), path=Path("/source/invoice.pdf")),
        _record("invoice (1).pdf", created_at=datetime(2024, 1, 1, 14, 0), path=Path("/source/invoice (1).pdf")),
    ]
    candidates = [
        NamingCandidate(record=r, category_key="invoice", subcategory=None, raw_description="invoice", confidence=0.9)
        for r in records
    ]

    dest_index = DestinationIndex(tmp_path)
    named = plan_organized_names(candidates, cfg, dest_index)

    by_source = {item.candidate.record.path.name: item for item in named}
    assert by_source["invoice.pdf"].naming_index == 1           # 08:00, earliest
    assert by_source["invoice (1).pdf"].naming_index == 2        # 14:00, middle
    assert by_source["invoice copy.pdf"].naming_index == 3       # 20:00, latest
    assert str(by_source["invoice.pdf"].dest_relative_path).endswith("_01.pdf")
    assert str(by_source["invoice copy.pdf"].dest_relative_path).endswith("_03.pdf")
    # All three share the same {yyyy}-{mm}-{dd} base, proving they genuinely collided/grouped.
    assert all(str(item.dest_relative_path).startswith("invoice_2024-01-01_") for item in by_source.values())


def test_copy_ordering_is_stable_across_reruns(tmp_path):
    cfg = NomiaConfig()
    records = [
        _record("a.pdf", created_at=datetime(2024, 1, 1), path=Path("/source/a.pdf")),
        _record("b.pdf", created_at=datetime(2024, 1, 1), path=Path("/source/b.pdf")),  # same date, tie-break on path
    ]
    candidates = [
        NamingCandidate(record=r, category_key="invoice", subcategory=None, raw_description="invoice", confidence=0.9)
        for r in records
    ]

    first = plan_organized_names(candidates, cfg, DestinationIndex(tmp_path))
    second = plan_organized_names(candidates, cfg, DestinationIndex(tmp_path))

    first_map = {i.candidate.record.path.name: i.naming_index for i in first}
    second_map = {i.candidate.record.path.name: i.naming_index for i in second}
    assert first_map == second_map


def test_single_file_group_gets_no_index(tmp_path):
    cfg = NomiaConfig()
    record = _record("unique.pdf", created_at=datetime(2024, 1, 1), path=Path("/source/unique.pdf"))
    candidate = NamingCandidate(record=record, category_key="invoice", subcategory=None, raw_description="unique-invoice", confidence=0.9)

    named = plan_organized_names([candidate], cfg, DestinationIndex(tmp_path))
    assert named[0].naming_index is None


def test_template_without_index_token_falls_back_to_collision_suffix(tmp_path):
    cfg = NomiaConfig()
    cfg.naming_preset_key = "foldered_category_year"  # "{category}/{yyyy}/{description}" - no {index}

    records = [
        _record("invoice.pdf", created_at=datetime(2024, 1, 1), path=Path("/source/invoice.pdf")),
        _record("invoice (1).pdf", created_at=datetime(2024, 2, 2), path=Path("/source/invoice (1).pdf")),
    ]
    candidates = [
        NamingCandidate(record=r, category_key="invoice", subcategory=None, raw_description="invoice", confidence=0.9)
        for r in records
    ]

    named = plan_organized_names(candidates, cfg, DestinationIndex(tmp_path))
    paths = {str(item.dest_relative_path) for item in named}
    assert len(paths) == 2  # never collide/overwrite even though the template alone can't disambiguate


def test_naming_never_produces_duplicate_paths_across_many_collisions(tmp_path):
    cfg = NomiaConfig()
    records = [
        _record(f"scan{i}.pdf", created_at=datetime(2024, 1, i + 1), path=Path(f"/source/scan{i}.pdf"))
        for i in range(5)
    ]
    candidates = [
        NamingCandidate(record=r, category_key="receipt", subcategory=None, raw_description="same-receipt", confidence=0.9)
        for r in records
    ]
    named = plan_organized_names(candidates, cfg, DestinationIndex(tmp_path))
    paths = [str(item.dest_relative_path) for item in named]
    assert len(paths) == len(set(paths))


# --------------------------------------------------------------------------------------------
# Special buckets
# --------------------------------------------------------------------------------------------

def test_special_bucket_paths_bypass_naming_template():
    assert unsorted_relative_path("weird:name?.pdf") == Path("_Unsorted/weirdname.pdf")
    assert other_relative_path("archive.zip") == Path("_Other/archive.zip")
    assert dump_relative_path("IMG_0001.jpg") == Path("_dump/IMG_0001.jpg")

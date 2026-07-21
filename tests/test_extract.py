import fitz
import pytest
from PIL import Image

from nomia.config import NomiaConfig
from nomia.extract import extract_all, extract_signals
from nomia.scanner import scan
from nomia.text_quality import TEXT_QUALITY_THRESHOLD


def _record_for(tmp_path, filename: str):
    records = scan([tmp_path])
    return next(r for r in records if r.path.name == filename)


def test_zero_byte_file_is_flagged(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    record = _record_for(tmp_path, "empty.jpg")
    signals = extract_signals(record, NomiaConfig())
    assert signals.error == "zero_byte"


def test_unsupported_extension_is_flagged(tmp_path):
    (tmp_path / "notes.txt").write_text("just some notes")
    record = _record_for(tmp_path, "notes.txt")
    signals = extract_signals(record, NomiaConfig())
    assert signals.error == "unsupported_format"
    assert signals.media_type == "unsupported"


def test_corrupt_image_is_flagged(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"this is not a real jpeg file at all")
    record = _record_for(tmp_path, "broken.jpg")
    signals = extract_signals(record, NomiaConfig())
    assert signals.error == "corrupt"


def test_valid_image_produces_render_and_exif(tmp_path):
    path = tmp_path / "photo.jpg"
    img = Image.new("RGB", (40, 20), color="blue")
    exif = img.getexif()
    exif[274] = 6  # Orientation
    img.save(path, exif=exif)

    record = _record_for(tmp_path, "photo.jpg")
    signals = extract_signals(record, NomiaConfig())

    assert signals.error is None
    assert signals.media_type == "image"
    assert signals.render_png is not None
    assert signals.exif_orientation == 6
    # exif_transpose(90deg rotation for orientation 6) swaps width/height.
    assert (signals.width, signals.height) == (20, 40)


def test_animated_gif_uses_first_frame_only(tmp_path):
    path = tmp_path / "anim.gif"
    frames = [Image.new("RGB", (10, 10), color=c) for c in ["red", "green", "blue"]]
    frames[0].save(path, save_all=True, append_images=frames[1:])

    record = _record_for(tmp_path, "anim.gif")
    signals = extract_signals(record, NomiaConfig())

    assert signals.error is None
    assert signals.render_png is not None


def test_valid_pdf_renders_first_page(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello Nomia")
    doc.save(path)
    doc.close()

    record = _record_for(tmp_path, "doc.pdf")
    signals = extract_signals(record, NomiaConfig())

    assert signals.error is None
    assert signals.media_type == "pdf"
    assert signals.render_png is not None
    assert signals.pdf_page_count == 1
    assert signals.extra_render_pngs == []


def test_pdf_pages_to_render_is_capped_at_two(tmp_path):
    path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}")
    doc.save(path)
    doc.close()

    record = _record_for(tmp_path, "multi.pdf")
    cfg = NomiaConfig(pdf_pages_to_render=2)
    signals = extract_signals(record, cfg)

    assert signals.pdf_page_count == 5
    assert len(signals.all_render_pngs()) == 2


def test_encrypted_pdf_is_flagged(tmp_path):
    path = tmp_path / "secret.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="hunter2", owner_pw="hunter2")
    doc.close()

    record = _record_for(tmp_path, "secret.pdf")
    signals = extract_signals(record, NomiaConfig())

    assert signals.error == "encrypted"


def test_corrupt_pdf_is_flagged(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not actually a valid pdf structure")
    record = _record_for(tmp_path, "broken.pdf")
    signals = extract_signals(record, NomiaConfig())
    assert signals.error == "corrupt"


def test_pdf_with_real_text_layer_is_detected_as_pdf_layer_source(tmp_path):
    path = tmp_path / "invoice.pdf"
    doc = fitz.open()
    page = doc.new_page()
    long_text = (
        "INVOICE\nAcme Corporation\n123 Main Street\nBill To: John Doe\n"
        "Total Due: $542.00\nThank you for your business."
    )
    page.insert_text((72, 72), long_text)
    doc.save(path)
    doc.close()

    record = _record_for(tmp_path, "invoice.pdf")
    signals = extract_signals(record, NomiaConfig())

    assert signals.text_source == "pdf_layer"
    assert signals.text_quality >= TEXT_QUALITY_THRESHOLD
    assert "Acme Corporation" in signals.extracted_text


def test_pdf_with_sparse_text_scores_below_threshold(tmp_path):
    path = tmp_path / "sparse.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hi")
    doc.save(path)
    doc.close()

    record = _record_for(tmp_path, "sparse.pdf")
    signals = extract_signals(record, NomiaConfig())

    assert signals.text_quality < TEXT_QUALITY_THRESHOLD


def test_pdf_with_no_text_layer_has_no_pdf_layer_source(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()  # no text inserted at all
    doc.save(path)
    doc.close()

    record = _record_for(tmp_path, "blank.pdf")
    signals = extract_signals(record, NomiaConfig())

    assert signals.error is None
    assert signals.extracted_text is None
    assert signals.text_quality == 0.0


def test_image_without_ocr_available_has_no_text_source(tmp_path):
    # ocrmac isn't installed in this environment - confirms the capability-gated OCR tier
    # degrades to exactly the pre-existing (no-text-signal) behavior rather than erroring.
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (40, 20), color="blue").save(path)

    record = _record_for(tmp_path, "photo.jpg")
    signals = extract_signals(record, NomiaConfig())

    assert signals.error is None
    assert signals.text_source is None
    assert signals.text_quality == 0.0
    assert signals.extracted_text is None


def test_extract_all_never_raises_on_mixed_batch(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    (tmp_path / "broken.jpg").write_bytes(b"garbage")
    (tmp_path / "notes.txt").write_text("hi")
    img_path = tmp_path / "good.png"
    Image.new("RGB", (10, 10)).save(img_path)

    records = scan([tmp_path])
    results = extract_all(records, NomiaConfig(), max_workers=4)

    assert len(results) == 4
    errors = {r.original_filename: r.error for r in results}
    assert errors["empty.jpg"] == "zero_byte"
    assert errors["broken.jpg"] == "corrupt"
    assert errors["notes.txt"] == "unsupported_format"
    assert errors["good.png"] is None

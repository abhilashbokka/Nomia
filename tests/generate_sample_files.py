"""Generates tests/sample_files/ and labels.json.

Not part of the app itself - a one-off dev tool kept in the repo so the labeled benchmark set
is reproducible/expandable rather than being a mystery pile of committed binaries. All content
is synthetic/mock (drawn shapes and placeholder text), including for id_document/bank_statement/
medical - never real personal data, per CLAUDE.md.

Run with: uv run python tests/generate_sample_files.py
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import fitz
import pillow_heif
from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent / "sample_files"
LABELS_PATH = OUT_DIR / "labels.json"

labels: dict[str, dict] = {}


def _set_mtime(path: Path, days_ago: float) -> None:
    ts = time.time() - days_ago * 86400
    os.utime(path, (ts, ts))


def _text_image(lines: list[str], size=(400, 560), bg="white", fg="black") -> Image.Image:
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=fg)
        y += 22
    return img


def add_image(name: str, image: Image.Image, *, category: str, subcategory: str | None = None, days_ago: float = 0) -> None:
    path = OUT_DIR / name
    image.save(path)
    if days_ago:
        _set_mtime(path, days_ago)
    labels[name] = {"category": category, "subcategory": subcategory}


def add_pdf(name: str, lines: list[str], *, category: str, subcategory: str | None = None, days_ago: float = 0) -> None:
    path = OUT_DIR / name
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    doc.save(path)
    doc.close()
    if days_ago:
        _set_mtime(path, days_ago)
    labels[name] = {"category": category, "subcategory": subcategory}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in OUT_DIR.glob("*"):
        if existing.is_file() and existing.name != "README.md":
            existing.unlink()
    labels.clear()

    # --- one example per starter category -----------------------------------------------------

    add_image("receipt_costco.jpg", _text_image([
        "COSTCO WHOLESALE", "1234 Market St", "",
        "Organic Bananas        3.99", "Almond Milk            4.49",
        "Paper Towels           8.99", "", "SUBTOTAL              17.47",
        "TAX                    1.40", "TOTAL                 18.87", "",
        "THANK YOU FOR SHOPPING",
    ]), category="receipt", subcategory="grocery")

    add_pdf("invoice_acme.pdf", [
        "ACME CONSULTING LLC", "INVOICE #10234", "Bill To: Example Customer", "",
        "Service: Website redesign ......... $2,400.00",
        "Service: Hosting (annual) ......... $180.00", "",
        "TOTAL DUE: $2,580.00", "Due date: 30 days",
    ], category="invoice")

    add_image("mock_id_card.jpg", _text_image([
        "*** SAMPLE MOCK ID - NOT A REAL DOCUMENT ***", "",
        "STATE OF EXAMPLE", "DRIVER LICENSE", "",
        "NAME: JANE SAMPLE", "DOB: 01/01/1990", "LICENSE #: X0000000",
        "EXP: 01/01/2030",
    ], size=(400, 260)), category="id_document", subcategory="drivers_license")

    add_pdf("mock_bank_statement.pdf", [
        "*** SAMPLE MOCK BANK STATEMENT - NOT REAL ***", "",
        "First Example Bank - Checking Account", "Statement Period: 01/01 - 01/31",
        "", "Beginning Balance:        $1,000.00",
        "Deposits:                 $2,500.00", "Withdrawals:              $1,800.00",
        "Ending Balance:           $1,700.00",
    ], category="bank_statement")

    add_pdf("mock_medical_report.pdf", [
        "*** SAMPLE MOCK MEDICAL RECORD - NOT REAL ***", "",
        "Example Clinic - Visit Summary", "Patient: John Sample", "Date: 01/15",
        "", "Vitals: BP 118/76, HR 68", "Assessment: Routine checkup, no concerns.",
    ], category="medical")

    add_image("screenshot_app.png", _screenshot_mock(), category="screenshot")

    add_image("photo_landscape.jpg", _landscape_mock(), category="photo")

    add_image("chart_sales.png", _bar_chart_mock(), category="diagram_or_chart")

    add_image("handwritten_note.png", _handwriting_mock(), category="handwritten_note")

    add_pdf("contract_nda.pdf", [
        "MUTUAL NON-DISCLOSURE AGREEMENT (SAMPLE)", "",
        "This agreement is entered into between Party A and Party B.",
        "1. Confidential Information ...", "2. Term ...", "",
        "Signature: _____________________   Date: ___________",
    ], category="contract_or_form")

    add_image("ambiguous_other.png", Image.new("RGB", (300, 300), color=(180, 180, 180)), category="other")

    # --- edge cases -----------------------------------------------------------------------------

    (OUT_DIR / "zero_byte.jpg").write_bytes(b"")
    (OUT_DIR / "corrupt.jpg").write_bytes(b"this is not a valid jpeg file at all, just garbage bytes")

    encrypted = fitz.open()
    encrypted.new_page()
    encrypted.save(OUT_DIR / "encrypted_statement.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="hunter2", owner_pw="hunter2")
    encrypted.close()

    multipage = [_text_image([f"Fax page {i + 1} of 3"], size=(300, 300)) for i in range(3)]
    multipage[0].save(OUT_DIR / "multipage_fax.tiff", save_all=True, append_images=multipage[1:])

    frames = [Image.new("RGB", (120, 120), color=c) for c in ["red", "green", "blue"]]
    frames[0].save(OUT_DIR / "animated.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)

    heic_source = _text_image(["HEIC sample receipt", "Generated for testing", "pillow-heif encode"], size=(300, 200))
    try:
        pillow_heif.from_pillow(heic_source).save(OUT_DIR / "receipt_heic_sample.heic")
        labels["receipt_heic_sample.heic"] = {"category": "receipt", "subcategory": None}
    except Exception as exc:  # noqa: BLE001 - HEIC encoding support varies by platform/libheif build
        print(f"Could not generate a synthetic HEIC sample ({exc}); skipping. "
              f"A real HEIC file from an iPhone is a better test anyway - add one manually if needed.")

    # --- duplicate / copy-ordering fixtures ------------------------------------------------------

    # Near-duplicate copies (same logical receipt, different bytes/timestamps) - exercises
    # naming.py's copy-ordering {index} assignment, NOT scanner.py's hash-based dedupe.
    base_lines = ["WALGREENS #4521", "Receipt", "Cough Drops    4.99", "TOTAL          4.99"]
    add_image("drugstore_receipt.jpg", _text_image(base_lines + [" "]), category="receipt", days_ago=3)
    add_image("drugstore_receipt (1).jpg", _text_image(base_lines + ["  "]), category="receipt", days_ago=2)
    add_image("drugstore_receipt copy.jpg", _text_image(base_lines + ["   "]), category="receipt", days_ago=1)

    # True byte-identical duplicate pair - exercises scanner.py's SHA-256 dedupe.
    dup_content = (OUT_DIR / "receipt_costco.jpg").read_bytes()
    (OUT_DIR / "receipt_costco_exact_copy.jpg").write_bytes(dup_content)
    # Deliberately not added to labels.json: dedupe should route it to skip_duplicate before it
    # ever reaches classification, so it isn't part of the accuracy benchmark.

    LABELS_PATH.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(list(OUT_DIR.glob('*')))} files to {OUT_DIR}")
    print(f"Wrote {len(labels)} labels to {LABELS_PATH}")


def _screenshot_mock() -> Image.Image:
    img = Image.new("RGB", (500, 320), color=(245, 245, 247))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 500, 36], fill=(230, 230, 232))
    draw.text((12, 10), "Settings", fill=(30, 30, 30))
    for i in range(4):
        y = 50 + i * 50
        draw.rectangle([20, y, 480, y + 36], outline=(210, 210, 212))
        draw.text((32, y + 10), f"Option {i + 1}", fill=(60, 60, 60))
    return img


def _landscape_mock() -> Image.Image:
    img = Image.new("RGB", (500, 320), color=(135, 206, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 220, 500, 320], fill=(34, 139, 34))
    draw.ellipse([380, 30, 460, 110], fill=(255, 236, 139))
    return img


def _bar_chart_mock() -> Image.Image:
    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)
    values = [80, 140, 60, 200, 110]
    x = 30
    for v in values:
        draw.rectangle([x, 260 - v, x + 40, 260], fill=(66, 133, 244))
        x += 60
    draw.line([20, 260, 380, 260], fill="black")
    return img


def _handwriting_mock() -> Image.Image:
    img = Image.new("RGB", (400, 300), color=(255, 255, 250))
    draw = ImageDraw.Draw(img)
    draw.line([40, 60, 120, 40, 180, 80, 240, 30, 320, 70], fill="black", width=2, joint="curve")
    draw.line([40, 120, 100, 140, 160, 100, 220, 150, 300, 110], fill="black", width=2, joint="curve")
    draw.line([40, 180, 90, 160, 150, 200, 210, 170, 280, 210], fill="black", width=2, joint="curve")
    return img


if __name__ == "__main__":
    main()

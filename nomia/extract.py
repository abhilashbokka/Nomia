"""Extracts the per-file signals that feed the classifier: a rendered PNG (for images and PDF
pages alike, so classify.py has a single code path), EXIF metadata, PDF page info, and cheap
path-based context. Every error case in here degrades to `ExtractedSignals.error` rather than
raising — one bad file must never crash the batch (see CLAUDE.md).
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.ExifTags import IFD

from nomia import ocr
from nomia.config import NomiaConfig
from nomia.scanner import FileRecord
from nomia.text_quality import TEXT_QUALITY_THRESHOLD, assess_text_quality

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
PDF_EXTENSIONS = {".pdf"}

# Keeps vision-model calls fast and memory use bounded; not specified by the product brief, a
# deliberate engineering default. 1024px on the long edge is ample for a coarse classifier.
MAX_RENDER_DIM = 1024
PDF_RENDER_DPI = 150

MediaType = Literal["image", "pdf", "unsupported"]
# "zero_byte" | "corrupt" | "encrypted" | "unsupported_format" | "unreadable"
ExtractError = str


@dataclass
class ExtractedSignals:
    path: Path
    media_type: MediaType
    render_png: bytes | None
    width: int | None
    height: int | None
    exif_datetime: datetime | None
    exif_orientation: int | None
    gps_lat: float | None
    gps_lon: float | None
    pdf_page_count: int | None
    path_context: str
    original_filename: str
    error: ExtractError | None = None
    extra_render_pngs: list[bytes] = field(default_factory=list)
    extracted_text: str | None = None
    text_quality: float = 0.0
    text_source: Literal["pdf_layer", "ocr", None] = None

    def all_render_pngs(self) -> list[bytes]:
        """All rendered pages/frames to send to the vision model, primary page first."""
        if self.render_png is None:
            return []
        return [self.render_png, *self.extra_render_pngs]


def _media_type_for(ext: str) -> MediaType:
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    return "unsupported"


def _path_context(record: FileRecord) -> str:
    try:
        rel_parent = record.path.parent.relative_to(record.source_root)
    except ValueError:
        rel_parent = record.path.parent
    parts = [p for p in rel_parent.parts if p not in (".",)]
    return " / ".join(parts)


def _downscale(image: Image.Image, max_dim: int = MAX_RENDER_DIM) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
    scale = max_dim / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _to_png_bytes(image: Image.Image) -> bytes:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _dms_to_decimal(dms, ref) -> float | None:
    try:
        degrees, minutes, seconds = (float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    value = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        value = -value
    return value


def _extract_exif(image: Image.Image) -> tuple[datetime | None, int | None, float | None, float | None]:
    exif_dt: datetime | None = None
    orientation: int | None = None
    lat: float | None = None
    lon: float | None = None
    try:
        exif = image.getexif()
        if not exif:
            return None, None, None, None

        orientation = exif.get(274)

        dt_raw = exif.get(306)  # DateTime
        try:
            exif_ifd = exif.get_ifd(IFD.Exif)
            dt_raw = exif_ifd.get(36867, dt_raw)  # DateTimeOriginal takes precedence
        except Exception:  # noqa: BLE001 - IFD may be absent/malformed; DateTime fallback is fine
            pass
        if dt_raw:
            try:
                exif_dt = datetime.strptime(str(dt_raw), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                exif_dt = None

        try:
            gps_ifd = exif.get_ifd(IFD.GPSInfo)
            if gps_ifd and 1 in gps_ifd and 2 in gps_ifd and 3 in gps_ifd and 4 in gps_ifd:
                lat = _dms_to_decimal(gps_ifd[2], gps_ifd[1])
                lon = _dms_to_decimal(gps_ifd[4], gps_ifd[3])
        except Exception:  # noqa: BLE001 - GPS is optional signal, never fatal
            pass
    except Exception as exc:  # noqa: BLE001 - EXIF is best-effort metadata, never fatal
        logger.debug("EXIF extraction failed: %s", exc)
    return exif_dt, orientation, lat, lon


def _extract_image(record: FileRecord) -> ExtractedSignals:
    base = dict(
        path=record.path, media_type="image", render_png=None, width=None, height=None,
        exif_datetime=None, exif_orientation=None, gps_lat=None, gps_lon=None,
        pdf_page_count=None, path_context=_path_context(record), original_filename=record.path.name,
    )
    try:
        with Image.open(record.path) as image:
            image.seek(0)  # first frame only for animated GIF / multi-frame images
            exif_dt, orientation, lat, lon = _extract_exif(image)
            oriented = ImageOps.exif_transpose(image) or image
            width, height = oriented.size
            render = _to_png_bytes(_downscale(oriented))
            ocr_text = ocr.ocr_image(oriented) if ocr.is_available() else ""
    except UnidentifiedImageError:
        return ExtractedSignals(**base, error="corrupt")
    except OSError as exc:
        logger.warning("Could not open image %s: %s", record.path, exc)
        return ExtractedSignals(**base, error="corrupt")

    # OCR is context enrichment for images, never a substitute for the image itself in
    # classify.py - visual structure (a photo's content, a chart's shape) can't be read off
    # OCR text alone, so this is always additive, never used to skip the vision call.
    text_quality = assess_text_quality(ocr_text)
    text_source: Literal["ocr", None] = "ocr" if ocr_text else None

    return ExtractedSignals(
        **{**base, "render_png": render, "width": width, "height": height,
           "exif_datetime": exif_dt, "exif_orientation": orientation, "gps_lat": lat, "gps_lon": lon,
           "extracted_text": ocr_text or None, "text_quality": text_quality, "text_source": text_source},
    )


def _extract_pdf(record: FileRecord, cfg: NomiaConfig) -> ExtractedSignals:
    base = dict(
        path=record.path, media_type="pdf", render_png=None, width=None, height=None,
        exif_datetime=None, exif_orientation=None, gps_lat=None, gps_lon=None,
        pdf_page_count=None, path_context=_path_context(record), original_filename=record.path.name,
    )
    try:
        doc = fitz.open(record.path)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF's malformed-PDF exceptions vary by cause
        logger.warning("Could not open PDF %s: %s", record.path, exc)
        return ExtractedSignals(**base, error="corrupt")

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            # Only flag genuinely password-protected PDFs. Some PDFs are marked encrypted but
            # readable with an empty password (permissions-only encryption) - authenticate("")
            # succeeding means we can proceed normally for those.
            return ExtractedSignals(**base, error="encrypted")

        page_count = len(doc)
        if page_count == 0:
            return ExtractedSignals(**base, error="corrupt")

        pages_to_render = max(1, min(cfg.pdf_pages_to_render, 2, page_count))
        pngs: list[bytes] = []
        width = height = None
        first_page_image = None
        text_parts: list[str] = []
        for i in range(pages_to_render):
            page = doc[i]
            text_parts.append(page.get_text())
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            png_bytes = pix.tobytes("png")
            with Image.open(io.BytesIO(png_bytes)) as rendered:
                rendered = rendered.convert("RGB")
                if width is None:
                    width, height = rendered.size
                    first_page_image = rendered.copy()
                pngs.append(_to_png_bytes(_downscale(rendered)))

        # Tier 1: a real embedded text layer (born-digital PDF) is the cheapest, most reliable
        # signal available - classify.py uses this to skip the image entirely when it's good
        # enough. Tier 2: a scanned/image-only PDF has no usable text layer, so fall back to
        # OCR on the rendered first page as enrichment (never a substitute for the image, since
        # OCR alone can't be trusted the way a real embedded text layer can).
        pdf_text = "\n\n".join(t for t in text_parts if t).strip()
        text_quality = assess_text_quality(pdf_text)
        extracted_text: str | None = pdf_text or None
        text_source: Literal["pdf_layer", "ocr", None] = "pdf_layer" if pdf_text else None

        if text_quality < TEXT_QUALITY_THRESHOLD and ocr.is_available() and first_page_image is not None:
            ocr_text = ocr.ocr_image(first_page_image)
            ocr_quality = assess_text_quality(ocr_text)
            if ocr_quality > text_quality:
                extracted_text = ocr_text or None
                text_quality = ocr_quality
                text_source = "ocr" if ocr_text else None

        return ExtractedSignals(
            **{**base, "render_png": pngs[0], "extra_render_pngs": pngs[1:],
               "width": width, "height": height, "pdf_page_count": page_count,
               "extracted_text": extracted_text, "text_quality": text_quality, "text_source": text_source},
        )
    except Exception as exc:  # noqa: BLE001 - rendering a malformed page must not crash the batch
        logger.warning("Could not render PDF %s: %s", record.path, exc)
        return ExtractedSignals(**base, error="corrupt")
    finally:
        doc.close()


def extract_signals(record: FileRecord, cfg: NomiaConfig) -> ExtractedSignals:
    media_type = _media_type_for(record.ext)

    if record.size == 0:
        return ExtractedSignals(
            path=record.path, media_type=media_type, render_png=None, width=None, height=None,
            exif_datetime=None, exif_orientation=None, gps_lat=None, gps_lon=None,
            pdf_page_count=None, path_context=_path_context(record),
            original_filename=record.path.name, error="zero_byte",
        )

    if media_type == "unsupported":
        return ExtractedSignals(
            path=record.path, media_type=media_type, render_png=None, width=None, height=None,
            exif_datetime=None, exif_orientation=None, gps_lat=None, gps_lon=None,
            pdf_page_count=None, path_context=_path_context(record),
            original_filename=record.path.name, error="unsupported_format",
        )

    if media_type == "image":
        return _extract_image(record)
    return _extract_pdf(record, cfg)


def extract_all(
    records: list[FileRecord],
    cfg: NomiaConfig,
    *,
    max_workers: int = 8,
) -> list[ExtractedSignals]:
    """I/O-bound stage — parallelized with a thread pool. Ollama itself serializes inference,
    so this parallelism buys real wall-clock time here (scanning/EXIF/PDF rendering) without
    contending with the (separately serialized) classification stage."""
    results: list[ExtractedSignals | None] = [None] * len(records)

    def _run(index: int, record: FileRecord) -> None:
        try:
            results[index] = extract_signals(record, cfg)
        except Exception as exc:  # noqa: BLE001 - a bug in extraction must not kill the batch
            logger.exception("Unexpected error extracting signals for %s: %s", record.path, exc)
            results[index] = ExtractedSignals(
                path=record.path, media_type=_media_type_for(record.ext), render_png=None,
                width=None, height=None, exif_datetime=None, exif_orientation=None,
                gps_lat=None, gps_lon=None, pdf_page_count=None,
                path_context=_path_context(record), original_filename=record.path.name,
                error="unreadable",
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda item: _run(*item), enumerate(records)))

    return [r for r in results if r is not None]

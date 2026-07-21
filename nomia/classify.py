"""Single-call Ollama classification: bundles the rendered image(s) plus every free signal
(EXIF, path context, PDF page count) into one structured-JSON chat call. No multi-pass
reasoning, no chained prompts — see CLAUDE.md.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Literal

import ollama
from pydantic import BaseModel, ValidationError

from nomia.config import CategoryDef, ConfidenceThresholds, NomiaConfig
from nomia.errors import ModelNotAvailableError
from nomia.extract import ExtractedSignals
from nomia.text_quality import TEXT_QUALITY_THRESHOLD

logger = logging.getLogger(__name__)

# Small vision models occasionally fall into a degenerate repetition loop when generating JSON
# for an ambiguous image (observed empirically: moondream re-emitting the same field over and
# over - "confidence": 0.67, ...0.66, ...0.65 - never reaching a natural stop). num_predict caps
# how many tokens a single call can generate, turning that failure mode into a fast, bounded
# "ran out of tokens, produced garbage" case that validate_and_repair already handles safely,
# rather than a many-minute stall. CALL_TIMEOUT_SECONDS is a second, independent safety net in
# case a call is slow for any other reason - a single classification must never be able to hang
# the whole batch (see CLAUDE.md: one bad file never kills the batch).
MAX_PREDICT_TOKENS = 300
CALL_TIMEOUT_SECONDS = 45
MAX_PLAUSIBLE_CATEGORY_LENGTH = 40
# Bounds how much extracted/OCR'd text goes into a single prompt - a couple of dense pages is
# already ample signal for classification; capping avoids an unusually text-heavy PDF ballooning
# prompt size (and latency) for no real accuracy benefit.
MAX_TEXT_CHARS_FOR_PROMPT = 4000


def _run_with_timeout(fn, timeout: float):
    """Runs fn() with a hard wall-clock timeout. Deliberately uses a fresh single-worker
    executor per call rather than a shared pool: a timed-out call's thread is orphaned (still
    blocked on the underlying HTTP request - Python threads can't be force-killed) and keeps
    running in the background until Ollama itself eventually finishes or errors. A shared,
    size-bounded pool could have every worker occupied by such orphans after enough timeouts,
    silently reintroducing the very hang this is meant to prevent for every later call. A
    throwaway executor means an orphan only ever costs one extra background thread, never
    blocks a future classification."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nomia-ollama-call")
    try:
        return executor.submit(fn).result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)

Route = Literal["auto", "review", "unsorted", "failed"]

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Last-resort repair tier: small vision models occasionally fall into a repetition loop after
# an otherwise-good first answer - re-emitting "confidence": 0.71, ...0.72, ...0.73, ... forever
# instead of closing the JSON object, which makes the whole response unparseable as a single
# object (no closing brace ever arrives within the token budget). But the model's *first*
# occurrence of each field, before the loop takes over, is usually a perfectly good answer -
# observed empirically, e.g. category/description/confidence all reasonable in the leading
# ~150 characters of an otherwise-truncated, syntactically-broken response. re.search finds the
# first match by default, which is exactly the one worth keeping.
_FIELD_PATTERNS = {
    "category": re.compile(r'"category"\s*:\s*"([^"]*)"'),
    "subcategory": re.compile(r'"subcategory"\s*:\s*"([^"]*)"'),
    "description": re.compile(r'"description"\s*:\s*"([^"]*)"'),
    "reason": re.compile(r'"reason"\s*:\s*"([^"]*)"'),
    "confidence": re.compile(r'"confidence"\s*:\s*(-?[0-9.]+)'),
}


def _extract_fields_individually(text: str) -> dict | None:
    category_match = _FIELD_PATTERNS["category"].search(text)
    if not category_match or not category_match.group(1):
        return None
    result: dict[str, object] = {"category": category_match.group(1)}
    for key in ("subcategory", "description", "reason"):
        match = _FIELD_PATTERNS[key].search(text)
        if match:
            result[key] = match.group(1)
    confidence_match = _FIELD_PATTERNS["confidence"].search(text)
    if confidence_match:
        try:
            result["confidence"] = float(confidence_match.group(1))
        except ValueError:
            pass
    return result


class ClassificationResult(BaseModel):
    category: str
    subcategory: str | None = None
    description: str
    reason: str
    confidence: float


class ClassificationOutcome(BaseModel):
    result: ClassificationResult | None
    raw_response: str
    model_used: str
    route: Route
    error: str | None = None


class ModelStatus(BaseModel):
    name: str
    pulled: bool
    is_active: bool


class ModelsStatusReport(BaseModel):
    ollama_reachable: bool
    models: list[ModelStatus]


def build_prompt(
    signals: ExtractedSignals, taxonomy: list[CategoryDef], *, text_only: bool = False,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt_text). The category key list comes from the live,
    user-editable taxonomy, not a hardcoded list, so a renamed/added/removed category is
    reflected in the very next classification call.

    `text_only=True` means the caller is deliberately not attaching the rendered image (a
    high-quality embedded PDF text layer was found) - the prompt wording and system role adapt
    accordingly rather than describing an image that isn't actually there."""
    category_keys = ", ".join(cat.key for cat in taxonomy) or "other"

    if text_only:
        system_prompt = (
            "You are a document classification assistant for a personal file organizer. "
            "You will be given the actual extracted text of a document (a PDF's real embedded "
            "text layer, not an OCR guess) and asked to classify it - no image is attached for "
            "this call.\n\n"
            f"The \"category\" field must be exactly one of these keys: {category_keys}.\n\n"
            "Respond with ONLY a single JSON object - no markdown code fences, no explanation "
            "before or after it - matching exactly this schema:\n"
            '{"category": "<one of the category keys above>", '
            '"subcategory": "<short freeform subcategory, or null if not applicable>", '
            '"description": "<ONLY 2 to 5 words, lowercase, hyphen-separated, like a filename - '
            'NEVER a full sentence. Good: costco-grocery-receipt, dmv-drivers-license, '
            'chest-xray-scan. Bad: \\"a receipt for a purchase at a store\\">", '
            '"reason": "<one concise sentence explaining the classification>", '
            '"confidence": <a number between 0 and 1>}'
        )
    else:
        system_prompt = (
            "You are a document and photo classification assistant for a personal file organizer. "
            "You will be shown an image - it may be a photograph, screenshot, scan, or a rendered "
            "page from a PDF - and asked to classify it.\n\n"
            f"The \"category\" field must be exactly one of these keys: {category_keys}.\n\n"
            "Respond with ONLY a single JSON object - no markdown code fences, no explanation "
            "before or after it - matching exactly this schema:\n"
            '{"category": "<one of the category keys above>", '
            '"subcategory": "<short freeform subcategory, or null if not applicable>", '
            '"description": "<ONLY 2 to 5 words, lowercase, hyphen-separated, like a filename - '
            'NEVER a full sentence. Good: costco-grocery-receipt, dmv-drivers-license, '
            'chest-xray-scan. Bad: \\"a receipt for a purchase at a store\\">", '
            '"reason": "<one concise sentence explaining the classification>", '
            '"confidence": <a number between 0 and 1>}'
        )

    context: dict[str, object] = {
        "original_filename": signals.original_filename,
        "media_type": signals.media_type,
    }
    if signals.path_context:
        context["containing_folder"] = signals.path_context
    if signals.exif_datetime is not None:
        context["photo_or_scan_taken_at"] = signals.exif_datetime.isoformat()
    if signals.gps_lat is not None and signals.gps_lon is not None:
        context["has_gps_location"] = True
    if signals.pdf_page_count is not None:
        context["pdf_page_count"] = signals.pdf_page_count
    if signals.extracted_text:
        context["extracted_text"] = signals.extracted_text[:MAX_TEXT_CHARS_FOR_PROMPT]
        context["extracted_text_source"] = (
            "pdf_embedded_text_layer" if signals.text_source == "pdf_layer" else "on_device_ocr"
        )

    if text_only:
        user_prompt = (
            "Classify this file using the extracted text below - it is the actual embedded "
            "text layer of the PDF, not a guess, and is the primary signal for this call:\n"
            + json.dumps(context, default=str)
        )
    else:
        user_prompt = (
            "Classify this file. Additional context extracted from the file itself is provided "
            "below as supporting evidence - the image remains the primary signal:\n"
            + json.dumps(context, default=str)
        )
    return system_prompt, user_prompt


def validate_and_repair(raw: str) -> ClassificationResult | None:
    """Best-effort recovery of a ClassificationResult from a raw model response: strips
    markdown fences, extracts the first {...} block if the model wrapped it in prose, clamps
    confidence into [0, 1], and fills in cheap defaults for optional-in-spirit fields. Returns
    None only when the response is unsalvageable (no usable category) - callers must treat
    that as a failed classification, never crash on it."""
    if not raw or not raw.strip():
        return None

    text = _FENCE_RE.sub("", raw.strip()).strip()

    candidate: object = None
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                candidate = json.loads(match.group(0))
            except json.JSONDecodeError:
                candidate = None

    if not isinstance(candidate, dict):
        candidate = _extract_fields_individually(text)

    if not isinstance(candidate, dict):
        return None

    if not candidate.get("category"):
        return None  # the one field we cannot sensibly default

    # Observed empirically on genuinely ambiguous images: instead of picking one category key,
    # the model sometimes echoes back the entire enumerated list from the system prompt as its
    # "category" value (e.g. "receipt, invoice, id_document, bank_statement, medical, ..."). A
    # real category key is always short, so treat an implausibly long one as unparseable rather
    # than accepting it - this is also a safety consideration, not just data quality: a garbage
    # multi-clause "category" string could otherwise become a real (very ugly) destination
    # folder name if a user reviewing quickly doesn't notice it and confirms anyway.
    if len(str(candidate["category"])) > MAX_PLAUSIBLE_CATEGORY_LENGTH:
        return None

    try:
        candidate["confidence"] = max(0.0, min(1.0, float(candidate.get("confidence", 0.0))))
    except (TypeError, ValueError):
        candidate["confidence"] = 0.0

    candidate.setdefault("subcategory", None)
    if not candidate.get("description"):
        candidate["description"] = str(candidate["category"])
    if not candidate.get("reason"):
        candidate["reason"] = ""

    try:
        return ClassificationResult.model_validate(candidate)
    except ValidationError as exc:
        logger.warning("Model response had a category but failed schema validation: %s", exc)
        return None


def route_by_confidence(confidence: float, thresholds: ConfidenceThresholds) -> Literal["auto", "review", "unsorted"]:
    if confidence >= thresholds.auto_min:
        return "auto"
    if confidence >= thresholds.review_min:
        return "review"
    return "unsorted"


def _is_model_not_found(exc: ollama.ResponseError) -> bool:
    return exc.status_code == 404 or "not found" in str(exc.error or "").lower()


def classify_file(
    signals: ExtractedSignals,
    cfg: NomiaConfig,
    *,
    model: str | None = None,
) -> ClassificationOutcome:
    """The single Ollama call per file. Callers (pipeline.py) are expected to have already
    routed anything with signals.error set away from here - this function still degrades
    safely if called anyway, rather than trusting the caller blindly."""
    active_model = model or cfg.model.active_model

    if signals.error is not None or not signals.all_render_pngs():
        return ClassificationOutcome(
            result=None, raw_response="", model_used=active_model,
            route="failed", error=signals.error or "no_render_available",
        )

    # A real embedded PDF text layer (born-digital PDF, not a scan) is the one case where we
    # skip the image entirely: it's cheaper/faster than a vision call and a more reliable signal
    # than anything the vision model would read off the rendered page anyway. OCR text (tier 2,
    # `text_source == "ocr"`) never triggers this - it's noisier and, more importantly, images
    # in that bucket (photos, scans, charts, screenshots) often depend on visual structure OCR
    # can't capture, so OCR is always additive context there, never a substitute for the image.
    text_only = (
        signals.text_source == "pdf_layer" and signals.text_quality >= TEXT_QUALITY_THRESHOLD
    )

    system_prompt, user_prompt = build_prompt(signals, cfg.taxonomy, text_only=text_only)
    client = ollama.Client(host=cfg.model.ollama_host)

    # Deliberately a single "user" message with the system instructions folded in, rather than
    # a separate system + user message pair. Verified empirically against the actual installed
    # moondream/Ollama combination: a separate system-role message produced consistently empty
    # completions (moondream isn't instruction-tuned on the system-prompt convention), and when
    # additionally combined with format="json" that empty output sent the grammar-constrained
    # decoder into a multi-minute pathological stall instead of just returning quickly with bad
    # output. Folding everything into one user message fixed both the emptiness and the latency.
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    message: dict[str, object] = {"role": "user", "content": combined_prompt}
    if not text_only:
        message["images"] = signals.all_render_pngs()

    def _call() -> ollama.ChatResponse:
        return client.chat(
            model=active_model,
            messages=[message],
            format="json",
            options={"temperature": 0.1, "num_predict": MAX_PREDICT_TOKENS},
            keep_alive=cfg.model.keep_alive,
        )

    try:
        response = _run_with_timeout(_call, CALL_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.error(
            "Ollama call for %s did not return within %ds; treating as a failed classification "
            "rather than blocking the batch.", signals.path, CALL_TIMEOUT_SECONDS,
        )
        return ClassificationOutcome(
            result=None, raw_response="", model_used=active_model,
            route="failed", error="timeout",
        )
    except ollama.ResponseError as exc:
        if _is_model_not_found(exc):
            raise ModelNotAvailableError(active_model) from exc
        logger.error("Ollama chat call failed for %s: %s", signals.path, exc)
        return ClassificationOutcome(
            result=None, raw_response=str(exc), model_used=active_model,
            route="failed", error="ollama_error",
        )
    except Exception as exc:  # noqa: BLE001 - network hiccups etc. must not crash the batch
        logger.error("Unexpected error calling Ollama for %s: %s", signals.path, exc)
        return ClassificationOutcome(
            result=None, raw_response=str(exc), model_used=active_model,
            route="failed", error="ollama_error",
        )

    raw_content = response.message.content or ""
    result = validate_and_repair(raw_content)
    if result is None:
        logger.warning("Could not parse a usable classification from model output for %s. Raw: %r", signals.path, raw_content[:500])
        return ClassificationOutcome(
            result=None, raw_response=raw_content, model_used=active_model,
            route="failed", error="invalid_json",
        )

    route = route_by_confidence(result.confidence, cfg.thresholds)

    # A small model occasionally invents a category that isn't one of the configured taxonomy
    # keys at all (observed empirically), rather than picking the closest real one - the system
    # prompt lists valid keys, but instruction-following isn't perfect. Never let that
    # auto-file into a nonsense destination folder without a human glancing at it first; the
    # model's own confidence score doesn't capture this specific failure mode, so it's checked
    # independently of the usual confidence threshold.
    valid_keys = {cat.key for cat in cfg.taxonomy}
    if route == "auto" and result.category not in valid_keys:
        logger.info(
            "Model returned an unrecognized category '%s' for %s; downgrading route from "
            "'auto' to 'review' regardless of confidence.", result.category, signals.path,
        )
        route = "review"

    return ClassificationOutcome(result=result, raw_response=raw_content, model_used=active_model, route=route)


def check_model_available(model: str, *, host: str) -> bool:
    try:
        client = ollama.Client(host=host)
        pulled = {m.model for m in client.list().models}
    except Exception:  # noqa: BLE001 - "not available" covers both "not pulled" and "unreachable"
        return False
    base = model.split(":")[0]
    return any(name == model or name.startswith(base + ":") for name in pulled)


def get_models_status(cfg: NomiaConfig) -> ModelsStatusReport:
    """Backs the /api/models/status endpoint and could equally be used by `nomia doctor`."""
    candidates = [cfg.model.default_model, cfg.model.accuracy_model]
    try:
        client = ollama.Client(host=cfg.model.ollama_host)
        pulled = {m.model for m in client.list().models}
        reachable = True
    except Exception:  # noqa: BLE001
        pulled = set()
        reachable = False

    statuses = []
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        base = name.split(":")[0]
        is_pulled = any(p == name or p.startswith(base + ":") for p in pulled)
        statuses.append(ModelStatus(name=name, pulled=is_pulled, is_active=(name == cfg.model.active_model)))

    return ModelsStatusReport(ollama_reachable=reachable, models=statuses)

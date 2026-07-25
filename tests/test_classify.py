from types import SimpleNamespace

import ollama
import pytest

from nomia.classify import (
    ClassificationResult,
    build_prompt,
    classify_file,
    route_by_confidence,
    validate_and_repair,
)
from nomia.config import ConfidenceThresholds, NomiaConfig
from nomia.errors import ModelNotAvailableError
from nomia.extract import ExtractedSignals


def _signals(**overrides) -> ExtractedSignals:
    base = dict(
        path=overrides.pop("path", "photo.jpg"), media_type="image", render_png=b"fake-png-bytes",
        width=100, height=100, exif_datetime=None, exif_orientation=None, gps_lat=None,
        gps_lon=None, pdf_page_count=None, path_context="", original_filename="photo.jpg", error=None,
    )
    base.update(overrides)
    return ExtractedSignals(**base)


def test_validate_and_repair_clean_json():
    raw = '{"category": "receipt", "subcategory": "grocery", "description": "costco-receipt", "reason": "Store receipt", "confidence": 0.91}'
    result = validate_and_repair(raw)
    assert result == ClassificationResult(category="receipt", subcategory="grocery", description="costco-receipt", reason="Store receipt", confidence=0.91)


def test_validate_and_repair_strips_markdown_fences():
    raw = '```json\n{"category": "photo", "description": "beach-sunset", "reason": "A photo", "confidence": 0.7}\n```'
    result = validate_and_repair(raw)
    assert result is not None
    assert result.category == "photo"


def test_validate_and_repair_extracts_json_from_prose():
    raw = 'Sure! Here is the classification: {"category": "screenshot", "description": "app-screenshot", "reason": "UI screenshot", "confidence": 0.6} Hope that helps!'
    result = validate_and_repair(raw)
    assert result is not None
    assert result.category == "screenshot"


def test_validate_and_repair_clamps_out_of_range_confidence():
    raw = '{"category": "other", "description": "misc", "reason": "unclear", "confidence": 5.0}'
    result = validate_and_repair(raw)
    assert result.confidence == 1.0

    raw_negative = '{"category": "other", "description": "misc", "reason": "unclear", "confidence": -3.0}'
    result_negative = validate_and_repair(raw_negative)
    assert result_negative.confidence == 0.0


def test_validate_and_repair_defaults_missing_optional_fields():
    raw = '{"category": "invoice", "confidence": 0.8}'
    result = validate_and_repair(raw)
    assert result is not None
    assert result.description == "invoice"
    assert result.reason == ""
    assert result.subcategory is None


def test_validate_and_repair_salvages_first_occurrence_from_a_repetition_loop():
    # Real captured output from moondream: a good first answer, then the model loops re-emitting
    # "confidence"/"description" instead of closing the JSON object - never produces a valid
    # whole object, but the first occurrence of each field is still a perfectly good answer.
    raw = (
        '{"category": "receipt", "subcategory": "other", "confidence": 0.71, '
        '"description": "A receipt for a purchase at a store", "reason": "Costco-grocery-receipt", '
        '"confidence": 0.72, "description": "A receipt for a purchase at a store", '
        '"confidence": 0.73, "description": "A receipt for a purchase at a store"'
    )
    result = validate_and_repair(raw)
    assert result is not None
    assert result.category == "receipt"
    assert result.subcategory == "other"
    assert result.description == "A receipt for a purchase at a store"
    assert result.reason == "Costco-grocery-receipt"
    assert result.confidence == 0.71  # the first occurrence, not a later looped one


def test_validate_and_repair_rejects_implausibly_long_category():
    # Real captured output from moondream on an ambiguous image: instead of picking one key,
    # it echoed back the entire enumerated category list from the prompt as "category".
    raw = (
        '{"category": "receipt, invoice, id_document, bank_statement, medical, screenshot, '
        'photo, diagram_or_chart, handwritten_note, contract_or_form", "description": "unclear", '
        '"reason": "", "confidence": 0.67}'
    )
    assert validate_and_repair(raw) is None


def test_validate_and_repair_returns_none_for_garbage():
    assert validate_and_repair("") is None
    assert validate_and_repair("not json at all, sorry") is None
    assert validate_and_repair('{"description": "no category field"}') is None


def test_route_by_confidence_defaults():
    thresholds = ConfidenceThresholds()
    assert route_by_confidence(0.95, thresholds) == "auto"
    assert route_by_confidence(0.80, thresholds) == "auto"
    assert route_by_confidence(0.79, thresholds) == "review"
    assert route_by_confidence(0.50, thresholds) == "review"
    assert route_by_confidence(0.49, thresholds) == "unsorted"
    assert route_by_confidence(0.0, thresholds) == "unsorted"


def test_build_prompt_includes_taxonomy_keys_and_context():
    cfg = NomiaConfig()
    signals = _signals(path_context="Downloads / Receipts", exif_datetime=None, pdf_page_count=3)
    system_prompt, user_prompt = build_prompt(signals, cfg.taxonomy)
    for cat in cfg.taxonomy:
        assert cat.key in system_prompt
    assert "photo.jpg" in user_prompt
    assert "Downloads / Receipts" in user_prompt
    assert "3" in user_prompt


def test_build_prompt_text_only_surfaces_extracted_text_and_no_image_wording():
    cfg = NomiaConfig()
    signals = _signals(
        text_source="pdf_layer", text_quality=0.9,
        extracted_text="INVOICE\nAcme Corp\nTotal Due: $542.00",
    )
    system_prompt, user_prompt = build_prompt(signals, cfg.taxonomy, text_only=True)
    assert "no image is attached" in system_prompt
    assert "Acme Corp" in user_prompt
    assert "pdf_embedded_text_layer" in user_prompt


def test_build_prompt_excludes_ocr_text_when_image_is_attached():
    # Measured head-to-head: image + OCR text was strictly worse than image alone (accuracy AND
    # latency), so OCR text must not ride along with an attached image. It still powers the
    # fast path's keyword fusion - just not the VLM prompt.
    cfg = NomiaConfig()
    signals = _signals(text_source="ocr", text_quality=0.6, extracted_text="some ocr'd words")
    _, user_prompt = build_prompt(signals, cfg.taxonomy)
    assert "some ocr'd words" not in user_prompt


def test_build_prompt_keeps_pdf_layer_text_alongside_image():
    # A real embedded text layer is trustworthy corroboration, unlike OCR guesses - it stays in
    # the prompt even when the rendered page image is attached (low-quality-layer case).
    cfg = NomiaConfig()
    signals = _signals(text_source="pdf_layer", text_quality=0.2, extracted_text="Acme layer text")
    _, user_prompt = build_prompt(signals, cfg.taxonomy)
    assert "Acme layer text" in user_prompt
    assert "pdf_embedded_text_layer" in user_prompt


def test_classify_file_short_circuits_when_signals_have_error():
    cfg = NomiaConfig()
    signals = _signals(error="corrupt", render_png=None)
    outcome = classify_file(signals, cfg)
    assert outcome.route == "failed"
    assert outcome.error == "corrupt"


def test_classify_file_parses_valid_model_response(mocker):
    cfg = NomiaConfig()
    fake_message = SimpleNamespace(content='{"category": "receipt", "description": "costco-receipt", "reason": "Store receipt", "confidence": 0.91}')
    fake_response = SimpleNamespace(message=fake_message)
    mocker.patch("ollama.Client.chat", return_value=fake_response)

    outcome = classify_file(_signals(), cfg)

    assert outcome.route == "auto"
    assert outcome.result.category == "receipt"
    assert outcome.model_used == cfg.model.active_model


def test_classify_file_handles_garbage_model_output(mocker):
    cfg = NomiaConfig()
    fake_response = SimpleNamespace(message=SimpleNamespace(content="I cannot classify this."))
    mocker.patch("ollama.Client.chat", return_value=fake_response)

    outcome = classify_file(_signals(), cfg)

    assert outcome.route == "failed"
    assert outcome.error == "invalid_json"
    assert outcome.raw_response == "I cannot classify this."


def test_classify_file_raises_model_not_available_for_missing_model(mocker):
    cfg = NomiaConfig()
    mocker.patch(
        "ollama.Client.chat",
        side_effect=ollama.ResponseError('{"error": "model \'ghost-model\' not found, try pulling it first"}', 404),
    )

    with pytest.raises(ModelNotAvailableError):
        classify_file(_signals(), cfg, model="ghost-model")


def test_classify_file_downgrades_unrecognized_category_from_auto_to_review(mocker):
    cfg = NomiaConfig()
    fake_message = SimpleNamespace(
        content='{"category": "not-a-real-category", "description": "mystery-thing", "reason": "?", "confidence": 0.95}'
    )
    mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    assert outcome.route == "review"  # never auto-files an unrecognized category, however confident
    assert outcome.result.category == "not-a-real-category"


def test_classify_file_handles_generic_ollama_error_gracefully(mocker):
    cfg = NomiaConfig()
    mocker.patch("ollama.Client.chat", side_effect=ollama.ResponseError("internal server error", 500))

    outcome = classify_file(_signals(), cfg)

    assert outcome.route == "failed"
    assert outcome.error == "ollama_error"


def test_classify_file_skips_image_for_high_quality_pdf_text_layer(mocker):
    cfg = NomiaConfig()
    fake_message = SimpleNamespace(
        content='{"category": "invoice", "description": "acme-invoice", "reason": "Invoice text", "confidence": 0.9}'
    )
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    signals = _signals(
        text_source="pdf_layer", text_quality=0.9,
        extracted_text="INVOICE\nAcme Corp\nTotal Due: $500.00",
    )
    outcome = classify_file(signals, cfg)

    assert outcome.route == "auto"
    sent_message = mock_chat.call_args.kwargs["messages"][0]
    assert "images" not in sent_message


def test_classify_file_keeps_image_for_ocr_text_source(mocker):
    cfg = NomiaConfig()
    fake_message = SimpleNamespace(
        content='{"category": "receipt", "description": "receipt", "reason": "r", "confidence": 0.9}'
    )
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    signals = _signals(text_source="ocr", text_quality=0.9, extracted_text="some ocr text " * 20)
    classify_file(signals, cfg)

    sent_message = mock_chat.call_args.kwargs["messages"][0]
    assert sent_message.get("images") == [b"fake-png-bytes"]


def test_classify_file_keeps_image_when_pdf_text_quality_is_low(mocker):
    cfg = NomiaConfig()
    fake_message = SimpleNamespace(
        content='{"category": "receipt", "description": "receipt", "reason": "r", "confidence": 0.9}'
    )
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    signals = _signals(text_source="pdf_layer", text_quality=0.1, extracted_text="Hi")
    classify_file(signals, cfg)

    sent_message = mock_chat.call_args.kwargs["messages"][0]
    assert sent_message.get("images") == [b"fake-png-bytes"]


# --- fast path (SigLIP + keyword fusion) ---------------------------------------------------

def _fake_siglip(mocker, probs_by_key: dict[str, float]):
    """Patches nomia.siglip to be 'available' and return a controlled distribution. Categories
    not named share the remaining mass equally, so the list always sums to ~1 and stays aligned
    with cfg.taxonomy order."""
    keys = [c.key for c in NomiaConfig().taxonomy]
    remaining = 1.0 - sum(probs_by_key.values())
    unnamed = [k for k in keys if k not in probs_by_key]
    fill = remaining / len(unnamed) if unnamed else 0.0
    probs = [probs_by_key.get(k, fill) for k in keys]
    mocker.patch("nomia.siglip.is_available", return_value=True)
    return mocker.patch("nomia.siglip.scores", return_value=probs)


RECEIPT_TEXT = "COSTCO WHOLESALE\nSubtotal 42.10\nChange Due 0.00"


def test_fast_path_confident_result_skips_the_vlm(mocker):
    cfg = NomiaConfig()
    _fake_siglip(mocker, {"receipt": 0.95})
    mock_chat = mocker.patch("ollama.Client.chat")

    signals = _signals(text_source="ocr", text_quality=0.8, extracted_text=RECEIPT_TEXT)
    outcome = classify_file(signals, cfg)

    mock_chat.assert_not_called()
    assert outcome.route == "auto"
    assert outcome.model_used == "siglip+keywords"
    assert outcome.result.category == "receipt"
    assert outcome.result.description == "costco-wholesale-receipt"
    assert "keyword evidence" in outcome.result.reason

    import json as _json
    audit = _json.loads(outcome.raw_response)
    assert audit["tier"] == "fast"
    assert "receipt" in audit["siglip_probs"]
    assert "subtotal" in audit["keyword_hits"]["receipt"]


def test_fast_path_low_confidence_falls_back_to_vlm_in_router_mode(mocker):
    cfg = NomiaConfig()
    _fake_siglip(mocker, {"receipt": 0.4, "invoice": 0.35})
    fake_message = SimpleNamespace(
        content='{"category": "invoice", "description": "acme-invoice", "reason": "r", "confidence": 0.9}'
    )
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    mock_chat.assert_called_once()
    assert outcome.model_used == cfg.model.active_model
    assert outcome.result.category == "invoice"


def test_fast_path_keyword_evidence_flips_the_siglip_winner(mocker):
    # SigLIP slightly prefers bank_statement, but the text is full of invoice phrases -
    # exactly the disambiguation the keyword layer exists for.
    cfg = NomiaConfig()
    cfg.fastpath.mode = "fast_only"
    _fake_siglip(mocker, {"bank_statement": 0.45, "invoice": 0.40})
    mock_chat = mocker.patch("ollama.Client.chat")

    text = "INVOICE\nInvoice Number 42\nBill To: X\nAmount Due: $5\nPayment Terms: Net 30"
    signals = _signals(text_source="pdf_layer", text_quality=0.9, extracted_text=text)
    outcome = classify_file(signals, cfg)

    mock_chat.assert_not_called()
    assert outcome.result.category == "invoice"
    assert outcome.route in ("auto", "review")  # fused confidence routes normally


def test_fast_only_mode_returns_low_confidence_result_without_vlm(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.mode = "fast_only"
    _fake_siglip(mocker, {"receipt": 0.4})
    mock_chat = mocker.patch("ollama.Client.chat")

    outcome = classify_file(_signals(), cfg)

    mock_chat.assert_not_called()
    assert outcome.model_used == "siglip+keywords"
    assert outcome.route in ("review", "unsorted")


def test_fast_path_siglip_failure_falls_back_to_vlm(mocker):
    cfg = NomiaConfig()
    mocker.patch("nomia.siglip.is_available", return_value=True)
    mocker.patch("nomia.siglip.scores", return_value=None)  # load/inference failed
    fake_message = SimpleNamespace(
        content='{"category": "photo", "description": "beach", "reason": "r", "confidence": 0.9}'
    )
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    mock_chat.assert_called_once()
    assert outcome.result.category == "photo"


def test_fast_only_without_deps_is_a_clean_per_file_failure(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.mode = "fast_only"  # conftest's autouse fixture reports siglip unavailable
    mock_chat = mocker.patch("ollama.Client.chat")

    outcome = classify_file(_signals(), cfg)

    mock_chat.assert_not_called()
    assert outcome.route == "failed"
    assert outcome.error == "fastpath_unavailable"


def test_fast_path_off_mode_never_touches_siglip(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    mock_scores = mocker.patch("nomia.siglip.scores")
    fake_message = SimpleNamespace(
        content='{"category": "photo", "description": "beach", "reason": "r", "confidence": 0.9}'
    )
    mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    mock_scores.assert_not_called()
    assert outcome.model_used == cfg.model.active_model


def test_fast_path_description_falls_back_to_category_without_text(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.prob_temperature = 1.0  # routing calibration isn't what this test is about
    _fake_siglip(mocker, {"photo": 0.97})
    mocker.patch("ollama.Client.chat")

    outcome = classify_file(_signals(extracted_text=None), cfg)

    assert outcome.route == "auto"
    assert outcome.result.description == "photo"
    assert outcome.result.subcategory is None


def test_thinking_model_family_gets_think_disabled(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    fake_message = SimpleNamespace(content='{"category": "photo", "description": "x", "reason": "r", "confidence": 0.9}')
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    classify_file(_signals(), cfg, model="qwen3.5:4b")

    assert mock_chat.call_args.kwargs.get("think") is False


def test_non_thinking_model_gets_no_think_parameter(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    fake_message = SimpleNamespace(content='{"category": "photo", "description": "x", "reason": "r", "confidence": 0.9}')
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    classify_file(_signals(), cfg, model="moondream")

    assert "think" not in mock_chat.call_args.kwargs


def test_think_rejection_retries_without_the_parameter(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    fake_message = SimpleNamespace(content='{"category": "photo", "description": "x", "reason": "r", "confidence": 0.9}')
    mock_chat = mocker.patch(
        "ollama.Client.chat",
        side_effect=[
            ollama.ResponseError('"qwen3.5-alike" does not support thinking', 400),
            SimpleNamespace(message=fake_message),
        ],
    )

    outcome = classify_file(_signals(), cfg, model="qwen3.5:4b")

    assert outcome.result.category == "photo"
    assert mock_chat.call_count == 2
    assert "think" not in mock_chat.call_args.kwargs  # the retry dropped it


def test_first_call_per_model_gets_cold_start_timeout_then_strict(mocker):
    from nomia import classify as classify_mod

    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    fake_message = SimpleNamespace(content='{"category": "photo", "description": "x", "reason": "r", "confidence": 0.9}')
    mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))
    timeouts = []
    real_run = classify_mod._run_with_timeout
    mocker.patch.object(
        classify_mod, "_run_with_timeout",
        side_effect=lambda fn, timeout: timeouts.append(timeout) or real_run(fn, timeout),
    )

    classify_file(_signals(), cfg)
    classify_file(_signals(), cfg)

    assert timeouts[0] == classify_mod.CALL_TIMEOUT_SECONDS * classify_mod.COLD_START_TIMEOUT_MULTIPLIER
    assert timeouts[1] == classify_mod.CALL_TIMEOUT_SECONDS


def test_vlm_call_downscales_the_render(mocker):
    import io as _io

    from PIL import Image as _Image

    from nomia.classify import VLM_MAX_IMAGE_DIM

    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    buf = _io.BytesIO()
    _Image.new("RGB", (1200, 900), color="white").save(buf, format="PNG")
    fake_message = SimpleNamespace(content='{"category": "photo", "description": "x", "reason": "r", "confidence": 0.9}')
    mock_chat = mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    classify_file(_signals(render_png=buf.getvalue()), cfg)

    sent = mock_chat.call_args.kwargs["messages"][0]["images"][0]
    with _Image.open(_io.BytesIO(sent)) as img:
        assert max(img.size) <= VLM_MAX_IMAGE_DIM


def test_vlm_fallback_answer_never_auto_files_by_default(mocker):
    # The file reached the VLM only because the fast path judged it hard; a confident VLM
    # answer there still goes to review (small local VLMs are badly calibrated on hard files).
    cfg = NomiaConfig()
    _fake_siglip(mocker, {"receipt": 0.4, "invoice": 0.35})  # not auto -> escalates
    fake_message = SimpleNamespace(
        content='{"category": "invoice", "description": "x", "reason": "r", "confidence": 0.97}'
    )
    mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    assert outcome.model_used == cfg.model.active_model
    assert outcome.route == "review"


def test_vlm_fallback_can_auto_file_when_guard_disabled(mocker):
    cfg = NomiaConfig()
    cfg.fastpath.review_vlm_fallback = False
    _fake_siglip(mocker, {"receipt": 0.4, "invoice": 0.35})
    fake_message = SimpleNamespace(
        content='{"category": "invoice", "description": "x", "reason": "r", "confidence": 0.97}'
    )
    mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    assert outcome.route == "auto"


def test_vlm_as_primary_classifier_still_auto_files(mocker):
    # With the fast path off entirely, the VLM is the primary classifier, not a fallback -
    # the review guard must not apply.
    cfg = NomiaConfig()
    cfg.fastpath.mode = "off"
    fake_message = SimpleNamespace(
        content='{"category": "invoice", "description": "x", "reason": "r", "confidence": 0.97}'
    )
    mocker.patch("ollama.Client.chat", return_value=SimpleNamespace(message=fake_message))

    outcome = classify_file(_signals(), cfg)

    assert outcome.route == "auto"

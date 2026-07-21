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


def test_build_prompt_includes_ocr_text_as_supporting_context():
    cfg = NomiaConfig()
    signals = _signals(text_source="ocr", text_quality=0.6, extracted_text="some ocr'd words")
    _, user_prompt = build_prompt(signals, cfg.taxonomy)
    assert "some ocr'd words" in user_prompt
    assert "on_device_ocr" in user_prompt


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

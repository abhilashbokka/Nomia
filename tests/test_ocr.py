from PIL import Image

from nomia import ocr


def test_is_available_returns_a_bool():
    assert isinstance(ocr.is_available(), bool)


def test_ocr_image_returns_empty_string_when_unavailable(mocker):
    mocker.patch("nomia.ocr._text_from_image", None)
    assert ocr.ocr_image(Image.new("RGB", (10, 10))) == ""


def test_ocr_image_never_raises_on_backend_failure(mocker):
    mocker.patch("nomia.ocr._text_from_image", side_effect=RuntimeError("boom"))
    assert ocr.ocr_image(Image.new("RGB", (10, 10))) == ""


def test_ocr_image_joins_detected_lines(mocker):
    mocker.patch("nomia.ocr._text_from_image", return_value=["line one", "line two"])
    assert ocr.ocr_image(Image.new("RGB", (10, 10))) == "line one\nline two"

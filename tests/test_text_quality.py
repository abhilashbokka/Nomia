from nomia.text_quality import TEXT_QUALITY_THRESHOLD, assess_text_quality


def test_empty_text_scores_zero():
    assert assess_text_quality(None) == 0.0
    assert assess_text_quality("") == 0.0
    assert assess_text_quality("   ") == 0.0


def test_real_prose_scores_above_threshold():
    text = (
        "INVOICE\nAcme Corporation\nBill To: John Doe\n"
        "Total Due: $542.00\nThank you for your business."
    )
    assert assess_text_quality(text) >= TEXT_QUALITY_THRESHOLD


def test_single_short_word_scores_low():
    assert assess_text_quality("Hi") < TEXT_QUALITY_THRESHOLD


def test_symbol_noise_scores_low():
    # Simulates garbled OCR output: mostly symbols, no real words, no spaces.
    assert assess_text_quality("|||///###@@@***") < TEXT_QUALITY_THRESHOLD

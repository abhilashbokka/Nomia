from nomia.config import CategoryDef, NomiaConfig
from nomia.keywords import derive_description, match_keywords


def _taxonomy() -> list[CategoryDef]:
    return NomiaConfig().taxonomy


INVOICE_TEXT = """ACME SUPPLY CO
123 Industrial Way

INVOICE
Invoice Number: 2024-0042
Bill To: Jane Smith
Payment Terms: Net 30
Amount Due: $1,204.50
"""

BANK_TEXT = """FIRST NATIONAL BANK
Statement Period: 01/01/2026 - 01/31/2026
Beginning Balance: $2,410.00
Deposits and other credits
Withdrawals and other debits
Ending Balance: $2,655.12
"""


def test_match_keywords_finds_distinct_phrases_per_category():
    matches = match_keywords(INVOICE_TEXT, _taxonomy())
    assert "invoice" in matches
    assert "invoice" in matches["invoice"]
    assert "amount due" in matches["invoice"]
    assert "net 30" in matches["invoice"]
    # No bank-statement phrases in an invoice
    assert "bank_statement" not in matches


def test_match_keywords_is_case_insensitive():
    matches = match_keywords(BANK_TEXT.upper(), _taxonomy())
    assert "bank_statement" in matches
    assert "statement period" in matches["bank_statement"]


def test_match_keywords_respects_word_boundaries():
    # "invoiced" must not match the keyword "invoice"
    taxonomy = [CategoryDef(key="invoice", label="Invoice", destination_subfolder="invoice", keywords=["invoice"])]
    assert match_keywords("we invoiced the client yesterday", taxonomy) == {}
    assert match_keywords("see the attached invoice.", taxonomy) == {"invoice": ["invoice"]}


def test_match_keywords_matches_phrase_across_line_break():
    # OCR frequently splits a phrase across lines; \s+ between words must cover that.
    matches = match_keywords("Beginning\nBalance: $5.00", _taxonomy())
    assert "bank_statement" in matches


def test_match_keywords_empty_text_and_no_keyword_categories():
    assert match_keywords("", _taxonomy()) == {}
    assert match_keywords(None, _taxonomy()) == {}
    matches = match_keywords(INVOICE_TEXT, _taxonomy())
    assert "photo" not in matches  # photo has no keywords configured
    assert "screenshot" not in matches


def test_derive_description_uses_title_line_and_appends_category():
    text = "COSTCO WHOLESALE\n123 Warehouse Rd\nSubtotal 42.10\nTotal 45.99"
    assert derive_description(text, "receipt") == "costco-wholesale-receipt"


def test_derive_description_skips_redundant_category_suffix():
    # Title already contains the category word - no "invoice-invoice"
    assert derive_description(INVOICE_TEXT, "invoice").endswith("invoice")
    assert "invoice-invoice" not in derive_description(INVOICE_TEXT, "invoice")


def test_derive_description_falls_back_to_category_without_text():
    assert derive_description(None, "bank_statement") == "bank-statement"
    assert derive_description("", "photo") == "photo"
    # Filler words dropped from multi-word keys
    assert derive_description(None, "diagram_or_chart") == "diagram-chart"


def test_derive_description_skips_digit_heavy_and_short_lines():
    text = "1234567890\nA1\nFIRST NATIONAL BANK\nAccount 000123"
    result = derive_description(text, "bank_statement")
    assert result.startswith("first-national-bank")


def test_derive_description_caps_slug_length():
    text = "this is a very long title\nbody text"
    result = derive_description(text, "other")
    # at most 4 words taken from the title + the category suffix
    assert result == "this-is-a-very-other"


def test_derive_description_ignores_sentence_length_lines():
    # A line with more than 8 words is body text, not a title
    text = "this is a very long sentence line with many many words\nACME CORP"
    assert derive_description(text, "receipt") == "acme-corp-receipt"


def test_derive_description_no_usable_title_line():
    # Every line fails the title heuristics (digits / too short)
    assert derive_description("12345\n99\n#$%^\n2024-01-01 10:11", "receipt") == "receipt"

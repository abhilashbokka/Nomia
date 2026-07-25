import json

from nomia.config import (
    DEFAULT_TAXONOMY,
    CategoryDef,
    NomiaConfig,
    load_config,
    migrate_config,
    save_config,
)


def test_default_taxonomy_ships_vision_prompts_and_keywords():
    cfg = NomiaConfig()
    by_key = {c.key: c for c in cfg.taxonomy}
    assert by_key["receipt"].vision_prompt is not None
    assert by_key["invoice"].keywords  # text-doc categories have keyword evidence
    assert by_key["bank_statement"].keywords
    assert by_key["photo"].keywords == []  # visual categories deliberately have none
    for cat in cfg.taxonomy:
        assert cat.effective_vision_prompt()  # never empty, prompt or label-derived


def test_effective_vision_prompt_falls_back_to_label():
    cat = CategoryDef(key="tax", label="Tax Form", destination_subfolder="tax")
    assert cat.effective_vision_prompt() == "a document or photo of tax form"


def test_migrate_config_backfills_prefastpath_taxonomy():
    # A config saved before the fast path existed: no keywords/vision_prompt fields at all.
    raw = NomiaConfig().model_dump(mode="json")
    for cat in raw["taxonomy"]:
        cat.pop("keywords", None)
        cat.pop("vision_prompt", None)

    cfg = migrate_config(raw)
    by_key = {c.key: c for c in cfg.taxonomy}
    defaults = {c.key: c for c in DEFAULT_TAXONOMY}
    assert by_key["invoice"].keywords == defaults["invoice"].keywords
    assert by_key["receipt"].vision_prompt == defaults["receipt"].vision_prompt


def test_migrate_config_respects_user_edits_and_unknown_categories():
    raw = NomiaConfig().model_dump(mode="json")
    for cat in raw["taxonomy"]:
        if cat["key"] == "invoice":
            cat["keywords"] = []  # user deliberately cleared them - must stay cleared
        if cat["key"] == "receipt":
            cat["keywords"] = ["my custom phrase"]
    raw["taxonomy"].append(
        {"key": "tax", "label": "Tax", "destination_subfolder": "tax"}  # user-added category
    )

    cfg = migrate_config(raw)
    by_key = {c.key: c for c in cfg.taxonomy}
    assert by_key["invoice"].keywords == []
    assert by_key["receipt"].keywords == ["my custom phrase"]
    assert by_key["tax"].keywords == []
    assert by_key["tax"].vision_prompt is None


def test_fastpath_config_defaults_and_roundtrip(tmp_path):
    cfg = NomiaConfig()
    assert cfg.fastpath.mode == "router"
    assert cfg.fastpath.keyword_boost == 3.0
    assert cfg.fastpath.prob_temperature == 2.0

    cfg.fastpath.mode = "fast_only"
    path = tmp_path / "config.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.fastpath.mode == "fast_only"
    assert loaded.taxonomy[0].keywords == cfg.taxonomy[0].keywords


def test_load_config_from_prefastpath_file_on_disk(tmp_path):
    """End-to-end migration through the real load path, not just migrate_config directly."""
    raw = NomiaConfig().model_dump(mode="json")
    raw.pop("fastpath")
    for cat in raw["taxonomy"]:
        cat.pop("keywords", None)
        cat.pop("vision_prompt", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    cfg = load_config(path)
    assert cfg.fastpath.mode == "router"  # new section takes defaults
    assert cfg.category_by_key("bank_statement").keywords  # backfilled

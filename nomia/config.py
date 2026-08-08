"""Configuration schema, defaults, and atomic load/save.

NomiaConfig is the single source of truth for taxonomy, naming templates, confidence
thresholds, model choice, and the safety toggles (preserve_source, keep_dump_copies). Both the
CLI and the web UI (via server.py's /api/config) read and write the same config.json — there is
no separate config format for either entry point.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from nomia.paths import config_file_path

logger = logging.getLogger(__name__)

CONFIG_VERSION = 1


class CategoryDef(BaseModel):
    key: str
    """Stable identifier — referenced by journal rows and never changes once a run has used it,
    even if the user later renames the category's label/folder."""
    label: str
    """Display name shown in the UI; freely editable."""
    destination_subfolder: str
    """What the `{category}` naming token resolves to for this category. Defaults to `key`, but
    is independently editable — e.g. a user can rename the folder files land in ("Receipts") while
    keeping the internal key ("receipt") stable for historical journal rows."""
    keywords: list[str] = Field(default_factory=list)
    """Keyword phrases whose presence in a file's extracted text (PDF layer / OCR) counts as
    evidence for this category in the fast classification path. User-editable; an empty list
    simply means text evidence never boosts this category (correct for visual categories like
    photo/screenshot)."""
    vision_prompt: str | None = None
    """Natural-language description of this category for the SigLIP zero-shot classifier
    (fast path). None falls back to a generic prompt derived from the label — fine for
    user-added categories, but the starter taxonomy ships tuned prompts."""
    extra_vision_prompts: list[str] = Field(default_factory=list)
    """Optional additional visual descriptions of this category (a descriptor ensemble): the
    fast path averages the SigLIP text embeddings of all of a category's prompts, so a
    category can be recognized by several distinct looks ("a grid of numbers", "a slide with
    sparse large text") instead of one phrasing. Empty list = single-prompt behavior,
    unchanged."""

    def effective_vision_prompt(self) -> str:
        return self.vision_prompt or f"a document or photo of {self.label.lower()}"

    def effective_vision_prompts(self) -> list[str]:
        return [self.effective_vision_prompt(), *self.extra_vision_prompts]


class NamingPreset(BaseModel):
    key: str
    label: str
    template: str


class ConfidenceThresholds(BaseModel):
    auto_min: float = 0.80
    """Confidence at or above this routes a file to auto-file."""
    review_min: float = 0.50
    """Confidence at or above this (but below auto_min) routes to review; below this routes to
    _Unsorted/."""


class ModelConfig(BaseModel):
    # qwen3.5:4b replaced moondream as the default VLM fallback on 2026-07-25: measured on the
    # 207-file real fixture set, moondream scored 27.1% and never produced auto-confidence,
    # while qwen3.5:4b (with think=False and 640px inputs, both handled in classify.py) is the
    # best VLM measured on this hardware. See tests/benchmark.py history.
    default_model: str = "qwen3.5:4b"
    accuracy_model: str = "llama3.2-vision:11b"
    active_model: str = "qwen3.5:4b"
    keep_alive: str = "30m"
    ollama_host: str = "http://127.0.0.1:11434"


class FastPathConfig(BaseModel):
    mode: Literal["router", "fast_only", "off"] = "router"
    """router: SigLIP+keywords first, VLM fallback for anything below the auto threshold.
    fast_only: never call the VLM — low-confidence files route to review/unsorted with the fast
    path's best guess (works without Ollama). off: VLM-only, the pre-fast-path behavior."""
    model_id: str = "google/siglip-base-patch16-224"
    keyword_boost: float = 3.0
    """Multiplicative boost per distinct keyword phrase matched: a category's SigLIP probability
    is scaled by (1 + keyword_boost * hits) before renormalizing across categories."""
    prob_temperature: float = 2.0
    """Flattens SigLIP's softmax (p ** (1/T), renormalized) before keyword fusion. SigLIP's raw
    distribution is routinely >0.99 confident even when wrong; measured on the labeled fixture
    set, T=2.0 raised the auto-filed bucket's correctness from ~90% to ~96% by letting genuinely
    ambiguous files fall through to review/VLM instead of auto-filing on an overconfident
    visual prior. 1.0 disables the flattening."""
    corpus_calibration: bool = False
    """Divide each file's SigLIP probabilities by the batch-mean probability vector before
    fusion (label-free prior correction, computed per run over the batch being organized).
    Fixes "attractor" categories whose prompt is systematically closer to the whole corpus's
    look — measured +4-6 accuracy points on the RVL-CDIP external benchmark with the auto
    bucket's precision intact. Off by default: on small or homogeneous batches (one screenshot
    folder) the batch mean is dominated by the true class distribution rather than prompt
    bias, and dividing it out would fight correct answers. Turn on for large, mixed corpora."""
    review_vlm_fallback: bool = True
    """In router mode, a file only reaches the VLM because the fast path already judged it hard
    - and small local VLMs are badly calibrated exactly there (measured on the 207-file set:
    qwen3.5:4b reported auto-level confidence on essentially every fallback file while being
    right only ~57% of the time, i.e. 42 wrong answers would have auto-filed). With this on
    (default), a fallback answer routes to review at best; only the fast path may auto-file.
    Has no effect on fastpath.mode="off", where the VLM is the primary classifier, not a
    fallback."""


# The starter categories' vision_prompt values are the exact zero-shot prompts validated in the
# SigLIP benchmark run (2026-07, 80% on 55 labeled real files) — see tests/benchmark.py for the
# only legitimate way to re-derive accuracy numbers after editing them.
DEFAULT_TAXONOMY: list[CategoryDef] = [
    CategoryDef(
        key="receipt", label="Receipt", destination_subfolder="receipt",
        vision_prompt="a scanned store receipt with itemized purchases and a total",
        keywords=["receipt", "subtotal", "change due", "cashier", "thank you for shopping",
                  "merchant copy", "customer copy", "auth code", "approval code"],
    ),
    CategoryDef(
        key="invoice", label="Invoice", destination_subfolder="invoice",
        vision_prompt="a business invoice with billing details and an amount due",
        keywords=["invoice", "invoice number", "amount due", "bill to", "due date",
                  "payment terms", "net 30", "remit to", "purchase order", "balance due"],
    ),
    CategoryDef(
        key="id_document", label="ID Document", destination_subfolder="id_document",
        vision_prompt="a scan of an identity card, passport, or driver's license",
        keywords=["date of birth", "driver license", "driver's license", "passport",
                  "identification card", "expiration date", "id number", "nationality",
                  "place of birth"],
    ),
    CategoryDef(
        key="bank_statement", label="Bank Statement", destination_subfolder="bank_statement",
        vision_prompt="a bank account statement listing transactions and balances",
        keywords=["statement period", "beginning balance", "ending balance", "account summary",
                  "available balance", "deposits", "withdrawals", "transaction history",
                  "checking account", "savings account", "routing number", "interest earned"],
    ),
    CategoryDef(
        key="medical", label="Medical", destination_subfolder="medical",
        vision_prompt="a medical laboratory report or health test result document",
        keywords=["patient", "diagnosis", "physician", "lab results", "specimen",
                  "reference range", "prescription", "dosage", "clinic", "medical record",
                  "date of service", "test results", "blood pressure"],
    ),
    CategoryDef(
        key="screenshot", label="Screenshot", destination_subfolder="screenshot",
        vision_prompt="a screenshot of a computer or smartphone user interface",
    ),
    CategoryDef(
        key="photo", label="Photo", destination_subfolder="photo",
        vision_prompt="a photograph of a person, animal, place, or object",
    ),
    CategoryDef(
        key="diagram_or_chart", label="Diagram / Chart", destination_subfolder="diagram_or_chart",
        vision_prompt="a bar chart, line graph, pie chart, or data diagram",
        keywords=["figure", "chart", "graph", "diagram"],
    ),
    CategoryDef(
        key="handwritten_note", label="Handwritten Note", destination_subfolder="handwritten_note",
        vision_prompt="a handwritten note written in cursive on paper",
    ),
    CategoryDef(
        key="contract_or_form", label="Contract / Form", destination_subfolder="contract_or_form",
        vision_prompt="a printed contract or a blank form with fields to fill in",
        keywords=["agreement", "hereinafter", "terms and conditions", "hereby", "shall",
                  "effective date", "the parties", "witness", "applicant", "please print"],
    ),
    CategoryDef(
        key="other", label="Other", destination_subfolder="other",
        vision_prompt="a painting, artwork, or miscellaneous image",
    ),
]

# Verbatim from the product brief's naming-preset table.
DEFAULT_NAMING_PRESETS: list[NamingPreset] = [
    NamingPreset(key="category_date_index", label="Category + date + index", template="{category}_{yyyy}-{mm}-{dd}_{index}"),
    NamingPreset(key="date_description", label="Date + description", template="{yyyy}-{mm}-{dd}_{description}"),
    NamingPreset(key="description_date", label="Description + date", template="{description}_{yyyy}-{mm}-{dd}"),
    NamingPreset(key="foldered_category_year", label="Foldered by category/year", template="{category}/{yyyy}/{description}"),
    NamingPreset(key="keep_original_tag_category", label="Keep original, tag category", template="{original}__{category}"),
]

CUSTOM_PRESET_KEY = "custom"


class NomiaConfig(BaseModel):
    version: int = CONFIG_VERSION

    source_folders: list[str] = Field(default_factory=list)
    destination_root: str | None = None

    taxonomy: list[CategoryDef] = Field(default_factory=lambda: [c.model_copy() for c in DEFAULT_TAXONOMY])
    naming_presets: list[NamingPreset] = Field(default_factory=lambda: [p.model_copy() for p in DEFAULT_NAMING_PRESETS])
    naming_preset_key: str = "category_date_index"
    custom_template: str | None = None

    thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    model: ModelConfig = Field(default_factory=ModelConfig)
    fastpath: FastPathConfig = Field(default_factory=FastPathConfig)

    sweep_other_files: bool = False
    keep_well_named_originals: bool = True
    """When a file's existing name already carries what the naming template would say about
    it (category and/or description words, the date), skip the rename and keep the original
    filename - the file is still organized into its normal destination folder. Scored by
    naming.original_name_score; see well_named_min_score."""
    well_named_min_score: float = 0.55
    """Minimum original_name_score ([0,1]) for an existing filename to be kept. At the
    default, a name needs roughly the category word plus a matching year, or most of the
    proposed description, before it is considered already-descriptive. Camera/scanner names
    (IMG_2041, scan0001) score ~0 and always get renamed."""
    max_filename_bytes: int = 255
    pdf_pages_to_render: int = Field(default=1, ge=1, le=2)
    reverse_geocode_enabled: bool = False

    preserve_source: bool = False
    """Opt-in: when true, apply_plan never removes source files (copy-only)."""
    keep_dump_copies: bool = True
    """When true, every applied file also gets a verbatim, unrenamed copy under
    {destination_root}/_dump/, independent of preserve_source."""

    def active_naming_template(self) -> str:
        """Resolves the effective template for the currently-selected preset, honoring the
        'Custom…' escape hatch."""
        if self.naming_preset_key == CUSTOM_PRESET_KEY:
            return self.custom_template or ""
        for preset in self.naming_presets:
            if preset.key == self.naming_preset_key:
                return preset.template
        # Fall back to the first built-in preset rather than raising — a stale/unknown
        # naming_preset_key (e.g. from an edited config file) should degrade, not crash.
        logger.warning("Unknown naming_preset_key '%s'; falling back to default preset.", self.naming_preset_key)
        return DEFAULT_NAMING_PRESETS[0].template

    def category_by_key(self, key: str) -> CategoryDef | None:
        for cat in self.taxonomy:
            if cat.key == key:
                return cat
        return None


def default_config_path() -> Path:
    return config_file_path()


def _resolve_path(path: Path | str | None) -> Path:
    return Path(path) if path is not None else default_config_path()


def _backfill_taxonomy_fastpath_fields(raw: dict) -> None:
    """Configs saved before the fast path existed have taxonomy entries without keywords/
    vision_prompt. For starter categories (matched by stable key), fill in the shipped defaults
    so an old config doesn't silently neuter the fast path. Presence-sensitive: a category that
    *has* a "keywords" entry (even an explicitly emptied one) is left exactly as the user saved
    it; only a missing field is backfilled. A missing-or-null vision_prompt gets the tuned
    default, which beats the generic label-derived fallback for starter categories."""
    defaults_by_key = {c.key: c for c in DEFAULT_TAXONOMY}
    taxonomy = raw.get("taxonomy")
    if not isinstance(taxonomy, list):
        return
    for raw_cat in taxonomy:
        if not isinstance(raw_cat, dict):
            continue
        default = defaults_by_key.get(raw_cat.get("key"))
        if default is None:
            continue
        if "keywords" not in raw_cat:
            raw_cat["keywords"] = list(default.keywords)
        if raw_cat.get("vision_prompt") is None:
            raw_cat["vision_prompt"] = default.vision_prompt


def migrate_config(raw: dict) -> NomiaConfig:
    """Forward-compat stub: v1 is the only schema version today, but funnel all loads through
    here so a future version bump has one place to add migration steps."""
    version = raw.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        logger.warning("Config file has version %s; expected %s. Attempting to load as-is.", version, CONFIG_VERSION)
    _backfill_taxonomy_fastpath_fields(raw)
    return NomiaConfig.model_validate(raw)


def load_config(path: Path | str | None = None) -> NomiaConfig:
    """Loads config from disk, or returns defaults if no config file exists yet. Never raises
    for a missing file — a fresh install should just work with sensible defaults."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        logger.info("No config file at %s; using defaults.", resolved)
        return NomiaConfig()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        return migrate_config(raw)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read config at %s (%s); using defaults instead.", resolved, exc)
        return NomiaConfig()


def save_config(cfg: NomiaConfig, path: Path | str | None = None) -> Path:
    """Atomic write: write to a temp file in the same directory, then os.replace, so a crash
    mid-save never leaves a corrupt/truncated config.json behind."""
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
    tmp_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, resolved)
    logger.info("Saved config to %s", resolved)
    return resolved

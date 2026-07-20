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
    default_model: str = "moondream"
    accuracy_model: str = "llama3.2-vision:11b"
    active_model: str = "moondream"
    keep_alive: str = "30m"
    ollama_host: str = "http://127.0.0.1:11434"


DEFAULT_TAXONOMY: list[CategoryDef] = [
    CategoryDef(key="receipt", label="Receipt", destination_subfolder="receipt"),
    CategoryDef(key="invoice", label="Invoice", destination_subfolder="invoice"),
    CategoryDef(key="id_document", label="ID Document", destination_subfolder="id_document"),
    CategoryDef(key="bank_statement", label="Bank Statement", destination_subfolder="bank_statement"),
    CategoryDef(key="medical", label="Medical", destination_subfolder="medical"),
    CategoryDef(key="screenshot", label="Screenshot", destination_subfolder="screenshot"),
    CategoryDef(key="photo", label="Photo", destination_subfolder="photo"),
    CategoryDef(key="diagram_or_chart", label="Diagram / Chart", destination_subfolder="diagram_or_chart"),
    CategoryDef(key="handwritten_note", label="Handwritten Note", destination_subfolder="handwritten_note"),
    CategoryDef(key="contract_or_form", label="Contract / Form", destination_subfolder="contract_or_form"),
    CategoryDef(key="other", label="Other", destination_subfolder="other"),
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

    sweep_other_files: bool = False
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


def migrate_config(raw: dict) -> NomiaConfig:
    """Forward-compat stub: v1 is the only schema version today, but funnel all loads through
    here so a future version bump has one place to add migration steps."""
    version = raw.get("version", CONFIG_VERSION)
    if version != CONFIG_VERSION:
        logger.warning("Config file has version %s; expected %s. Attempting to load as-is.", version, CONFIG_VERSION)
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

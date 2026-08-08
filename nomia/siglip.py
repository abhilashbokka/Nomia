"""Optional SigLIP zero-shot image classifier — the vision half of the fast classification
path. Capability-gated exactly like nomia/ocr.py: the `fastpath` optional dependency group
(torch + transformers) may simply not be installed, in which case everything here degrades to
"unavailable" and classification falls back to the Ollama vision model, exactly as before this
module existed.

Why SigLIP instead of another Ollama call: classification to a fixed category list is a
discriminative task, and paying an autoregressive VLM ~8 tokens/second to emit an argmax is
the single biggest latency cost in the pipeline on CPU-only hardware. A 203M-parameter SigLIP
forward pass returns a calibrated distribution over exactly the configured categories in
well under a second on the same machine, and cannot hallucinate a category key or fall into a
repetition loop. Measured against the labeled fixture set before adoption — see
tests/benchmark.py --mode fast.

The model weights are fetched from the Hugging Face hub once (like `ollama pull`) and cached
locally; every subsequent load and all inference is fully offline.
"""

from __future__ import annotations

import io
import logging
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"

_availability: bool | None = None
_lock = threading.Lock()
# One loaded (processor, model) pair per model_id, kept for the life of the process — loading
# takes ~25s on the target hardware, so it must happen at most once per batch, not per file.
_models: dict[str, tuple[object, object]] = {}
# Normalized text embeddings per (model_id, prompts) — the taxonomy's prompts are identical for
# every file in a run, so the text tower only needs to run when the taxonomy changes.
_text_embeds: dict[tuple[str, tuple[str, ...]], object] = {}
# Per-group mean embeddings (prompt ensembling), cached under the same regime.
_group_embeds: dict[tuple[str, tuple[tuple[str, ...], ...]], object] = {}
# Memoized per-image results so a corpus-calibration pre-pass and the classification pass
# share ONE vision-tower forward per file. Keyed by image-content digest; bounded.
_score_cache: dict[tuple[str, str, int], list[float]] = {}
_SCORE_CACHE_MAX = 8192


def is_available() -> bool:
    """True when the optional torch/transformers stack is importable. Cached after the first
    check; never raises."""
    global _availability
    if _availability is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _availability = True
        except Exception as exc:  # noqa: BLE001 - a broken install must read as "unavailable"
            logger.debug("SigLIP fast path unavailable: %s", exc)
            _availability = False
    return _availability


def is_model_cached(model_id: str = DEFAULT_MODEL_ID) -> bool:
    """Whether the model weights are already in the local Hugging Face cache (i.e. no network
    needed on first classification). Backs `nomia doctor`; best-effort, never raises."""
    try:
        from huggingface_hub import scan_cache_dir

        return any(repo.repo_id == model_id for repo in scan_cache_dir().repos)
    except Exception:  # noqa: BLE001
        return False


def _get_model(model_id: str):
    with _lock:
        cached = _models.get(model_id)
        if cached is not None:
            return cached

        from transformers import AutoModel, AutoProcessor

        try:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet; fetch once, then offline forever
            logger.info(
                "SigLIP model '%s' not in the local cache; downloading once (one-time, "
                "like `ollama pull`). All later runs are fully offline.", model_id,
            )
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
        model.eval()
        _models[model_id] = (processor, model)
        logger.info("Loaded SigLIP model '%s'.", model_id)
        return processor, model


def _get_text_embeds(model_id: str, prompts: tuple[str, ...], processor, model):
    import torch

    key = (model_id, prompts)
    cached = _text_embeds.get(key)
    if cached is not None:
        return cached

    # padding="max_length" matters: SigLIP's text tower was trained with max-length padding and
    # underperforms noticeably without it (matches the validated benchmark configuration).
    inputs = processor(text=list(prompts), return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        embeds = model.get_text_features(**inputs)
    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
    _text_embeds[key] = embeds
    return embeds


def _get_group_embeds(model_id: str, groups: tuple[tuple[str, ...], ...], processor, model):
    """One normalized embedding per prompt group: the mean of the group's normalized prompt
    embeddings, re-normalized (canonical CLIP-style prompt ensembling). A single-prompt group
    reduces to that prompt's embedding exactly, so ensembling is a strict generalization of
    the single-prompt path."""
    import torch

    key = (model_id, groups)
    cached = _group_embeds.get(key)
    if cached is not None:
        return cached

    flat = tuple(p for group in groups for p in group)
    embeds = _get_text_embeds(model_id, flat, processor, model)
    means = []
    offset = 0
    for group in groups:
        chunk = embeds[offset:offset + len(group)]
        offset += len(group)
        mean = chunk.mean(dim=0, keepdim=True)
        means.append(mean / mean.norm(p=2, dim=-1, keepdim=True))
    result = torch.cat(means, dim=0)
    _group_embeds[key] = result
    return result


def scores_grouped(
    png_bytes: bytes, prompt_groups: list[list[str]], *, model_id: str = DEFAULT_MODEL_ID,
) -> list[float] | None:
    """Zero-shot probabilities for one image against prompt GROUPS (one group per category,
    each holding one or more descriptor prompts), in group order. Ensembling multiple visual
    descriptions per category lets text-poor categories be recognized by what is drawn on the
    page (a grid of numbers, sparse display type, a product photograph) rather than a single
    phrasing. Same failure contract as scores(): None on any failure, never raises."""
    if not is_available():
        return None
    try:
        import hashlib

        import torch
        from PIL import Image

        groups = tuple(tuple(g) for g in prompt_groups)
        cache_key = (hashlib.sha256(png_bytes).hexdigest()[:24], model_id, hash(groups))
        cached = _score_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        processor, model = _get_model(model_id)
        text_embeds = _get_group_embeds(model_id, groups, processor, model)

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        with torch.no_grad():
            pixel_inputs = processor(images=image, return_tensors="pt")
            image_embeds = model.get_image_features(**pixel_inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            logits = image_embeds @ text_embeds.t() * model.logit_scale.exp() + model.logit_bias
            probs = logits.softmax(dim=-1)[0]
        result = [float(p) for p in probs]
        if len(_score_cache) >= _SCORE_CACHE_MAX:
            _score_cache.clear()
        _score_cache[cache_key] = result
        return list(result)
    except Exception as exc:  # noqa: BLE001 - degrade to VLM fallback, never crash
        logger.warning("SigLIP grouped scoring failed (%s); falling back.", exc)
        return None


def scores(png_bytes: bytes, prompts: list[str], *, model_id: str = DEFAULT_MODEL_ID) -> list[float] | None:
    """Zero-shot probabilities for one image against the prompt list, in prompt order
    (softmax over the SigLIP image-text logits — the exact formulation validated against the
    labeled fixture set). Returns None on any failure: missing deps, unreadable image, model
    load/download failure — a fast-path problem must never crash the batch, only fall back."""
    if not is_available():
        return None
    try:
        import torch
        from PIL import Image

        processor, model = _get_model(model_id)
        text_embeds = _get_text_embeds(model_id, tuple(prompts), processor, model)

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        with torch.no_grad():
            pixel_inputs = processor(images=image, return_tensors="pt")
            image_embeds = model.get_image_features(**pixel_inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            logits = image_embeds @ text_embeds.t() * model.logit_scale.exp() + model.logit_bias
            probs = logits.softmax(dim=-1)[0]
        return [float(p) for p in probs]
    except Exception as exc:  # noqa: BLE001 - degrade to VLM fallback, never crash
        logger.warning("SigLIP scoring failed (%s); falling back.", exc)
        return None

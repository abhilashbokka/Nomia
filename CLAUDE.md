# CLAUDE.md — Nomia

Guidance for Claude Code (and any other contributor) working in this repository.

## What Nomia is

Nomia is a local-first, AI-powered file organizer. Point it at messy source folders; it uses a
**local vision model running through Ollama** to classify each image and PDF, moves files into an
editable destination structure, optionally renames them sensibly, and produces an **Excel log
explaining every decision**. Fully offline — no cloud API calls, ever.

**The name:** *Nomia* — from the Greek suffix *-nomia* (as in taxonomy, economy, autonomy): order,
distribution, classification. The product's entire purpose is bringing order to chaotic files.
Keep this framing in the README and any user-facing copy.

## The pipeline (do not restructure without discussion)

```
Source folders
   → scan (walk dirs, dedupe by content hash)
   → extract signals per file (EXIF, PDF page render / text layer / OCR, image decode, path context)
   → classify — tiered:
        fast path (default): SigLIP zero-shot on the rendered image × keyword evidence in the
                             extracted text → fused confidence (nomia/siglip.py, nomia/keywords.py)
        fallback: single Ollama vision call → structured JSON (only when the fast path isn't
                  auto-confident; `fastpath.mode` = router | fast_only | off)
   → confidence routing (auto / review / unsorted)
   → naming (template engine → collision + copy-ordering logic)
   → organize (dry-run preview → move/copy, with undo journal)
   → report (Excel log of every decision)
```

Each file is read **once**. All free signals (EXIF, path, dates) are bundled and passed to the
model alongside the rendered image in a **single structured-JSON call** — no multi-pass reasoning,
no chained prompts.

Ollama serializes model inference server-side, so a `ThreadPoolExecutor` helps the I/O-bound
stages (scanning, EXIF reads, PDF rendering) but buys nothing at the classification step itself.
Parallelize extraction; queue classification through one worker.

### The fast classification path (why it exists, how it's wired)

An autoregressive VLM emitting an 11-way label on CPU-only hardware costs 15–47s/file (~8.7
tokens/sec decode is the hardware floor on the 2019 Intel MBP this targets). Classification to a
fixed category list is a discriminative task, so the default path is:

- **SigLIP zero-shot** (`google/siglip-base-patch16-224`, `nomia/siglip.py`) scores the rendered
  image against each taxonomy category's `vision_prompt` (~0.2s/file on the same hardware).
- **Keyword fusion** (`nomia/keywords.py`): distinct keyword-phrase hits in the extracted text
  (PDF layer or Apple-Vision OCR) multiply up that category's probability —
  `p ** (1/prob_temperature) * (1 + keyword_boost * hits)`, renormalized. This is what separates
  invoice/bank_statement/medical/contract, which look alike to a vision embedding.
- Fast-path results route through the SAME confidence thresholds; in `router` mode (default)
  only an auto-confident fast result skips the VLM — everything ambiguous still gets the full
  vision call. `fast_only` never calls Ollama; `off` restores VLM-for-everything.
- Both `keywords` and `vision_prompt` live on `CategoryDef` (user-editable taxonomy);
  `keyword_boost`/`prob_temperature` defaults were tuned against the 207-file labeled set —
  don't change them without re-running `tests/benchmark.py --mode fast --sample-dir
  tests/real_sample_files`.
- The fast path is an **optional extra** (`uv sync --extra fastpath`, torch + transformers),
  capability-gated exactly like `nomia/ocr.py`: absent deps degrade to VLM-only, never crash.
  The SigLIP weights download once from HF (like `ollama pull`), then everything is offline.
- Per-item audit: fast-path decisions store the full evidence (per-category SigLIP probs,
  keyword hits, fused confidence) as JSON in `raw_model_response`, with `model_used =
  "siglip+keywords"` — invariant #3 applies to every tier equally.
- **Intel-Mac dependency ceiling:** the last x86-macOS torch wheel is 2.2.2 (cp312), which is why
  `.python-version` pins 3.12 and the `fastpath` extra forks its torch requirement by platform.
  Don't bump either without checking wheels exist for this machine.

## Five non-negotiable invariants

These are enforced **structurally in code**, not just by convention. Any change that weakens one
of these needs to be flagged explicitly, not slipped in as a side effect of a refactor.

1. **Never overwrite a file. Never.** Every collision — at the destination, in `_dump/`, anywhere —
   is resolved with a mechanical, non-semantic suffix (`__2`, `__3`, …), logged at WARNING.
2. **Never delete a source file — only move.** A move is: verify the destination copy's SHA-256
   against the source, *then* remove the source. If verification fails, the source is untouched.
   The `preserve_source` config flag goes further and disables source removal entirely (copy-only
   mode) for users who want the source folder to stay byte-for-byte untouched.
3. **Every action is logged with a reason.** Every planned and applied item — including skips,
   duplicates, and failures — has a row in the undo journal and the Excel report.
4. **Dry-run first.** The default flow previews all moves (a `Plan`); nothing touches disk until
   the user explicitly confirms "Apply."
5. **Every applied run writes an undo journal** so the whole batch can be rolled back, and a run
   interrupted mid-batch can be resumed or rolled back cleanly rather than left in an ambiguous
   state.

Two additions layered on top of these, not replacements for them:

- **`_dump/` safety net** (`keep_dump_copies`, default on): every applied file also gets a
  verbatim, unrenamed copy at `{destination_root}/_dump/`, independent of the organized/renamed
  copy in its category folder. This exists so a raw, untouched-name backup always survives even
  if a rename or category turns out wrong.
- **Verification pass**: every `apply_plan()` run ends with an automatic count reconciliation
  (every scanned file accounted for by final status) and a hash reconciliation (destination and
  dump bytes re-hashed from disk and checked against the source hash recorded at scan time). A
  mismatch is a critical, visible finding — surfaced in the API response and the Excel report's
  Verification sheet — never silently swallowed into a generic "success."

## Classification contract

- Default VLM (the fast path's fallback): `qwen3.5:4b` — replaced `moondream` 2026-07-25 after
  the 207-file real benchmark (moondream 27.1%, never auto-confident; qwen3.5:4b is the best VLM
  measured on this hardware, run with `think=False` + 640px inputs, both handled in classify.py).
  Accuracy mode: `llama3.2-vision:11b`, selectable in config/UI (still unmeasured here). Handle
  any model not being pulled yet as a normal, expected state — never let a missing model crash
  anything; surface the exact `ollama pull <model>` command.
- Call shape (verified against the installed `ollama` Python client — do not assume the shape from
  older docs/examples):
  ```python
  ollama.chat(
      model=...,
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt_text, "images": [png_bytes]},
      ],
      format="json",       # or a JSON-schema dict derived from the response model
      options={"temperature": 0.1},
      keep_alive="30m",    # TOP-LEVEL kwarg on chat(), not nested inside options
  )
  ```
- **Thinking-capable model families get `think=False`** (top-level chat kwarg, families listed in
  `classify.py`'s `_THINKING_MODEL_PREFIXES`): left enabled, a thinking model burns the whole
  `num_predict` budget on reasoning tokens and never emits JSON — measured 47.6s → 5.0s per file
  on qwen3.5:4b with the answer intact. A model that rejects the parameter is retried without it.
- Required JSON schema back from the model:
  ```json
  {
    "category": "receipt",
    "subcategory": "grocery",
    "description": "costco-grocery-receipt",
    "reason": "Printed store receipt with itemized totals and Costco header",
    "confidence": 0.91
  }
  ```
  `description` must be short and slug-friendly — it feeds the filename. Always validate and
  attempt to repair the JSON (strip markdown fences, extract the first `{...}` block, clamp
  confidence to `[0, 1]`); if repair fails, treat it as a failed classification — route to
  `_Unsorted/` and log the raw model output. Never crash the batch over one bad response.
- **Confidence routing** (defaults, configurable):
  - `>= 0.80` → auto-file
  - `0.50 – 0.80` → flag for review (pre-filled suggestion, requires explicit per-item confirm)
  - `< 0.50` → route to `_Unsorted/` (this is still an automatic filing action, just a flagged one)
- PDFs: render page 1 (optionally up to a capped page count, default 1, hard cap 2) to PNG with
  PyMuPDF and feed that through the same path as images. Never load a whole multi-hundred-page PDF
  into the model.

## Copy-ordering rule (the numbering scheme for grouped/duplicate-name files)

When several files resolve to the same base name (near-duplicate copies like `invoice.pdf`,
`invoice (1).pdf`, `invoice copy.pdf`, or unrelated files that happen to classify to the same
template output): group them, sort **ascending by effective creation date**, and assign a
sequential `{index}` (`_01`, `_02`, …) — chronological, oldest first, and **stable across
re-runs**. Never use random suffixes for this.

Effective date fallback chain, in order, and log which tier was used:
`OS creation time → OS modified time → date parsed from the original filename → discovery order`.
Within the last tier, tie-break on the normalized relative source path string (not raw directory
walk order) — raw walk order is not guaranteed identical across re-runs on every filesystem, and
the path-string tie-break is what actually delivers "stable across re-runs."

This `{index}` mechanism is distinct from the collision-suffix mechanism in invariant #1: `{index}`
is a semantic part of the naming template the user chose; the collision suffix is a purely
mechanical fallback for when two different planned names would otherwise land on the same path.

**Already-well-named files** (`keep_well_named_originals`, default on): if a file's existing name
already carries what the template would say about it — scored by `naming.original_name_score`
(0.5 × specific-description-word coverage + 0.3 × category word + 0.2 × year; a bare category
word like `invoice (1).pdf` tops out at 0.3 and is treated as generic) — the rename is skipped:
the file still moves to the template's folder, keeps its sanitized original filename, stays out
of `{index}` grouping, and still goes through collision resolution. The decision is visible in
the item's Reason ("kept original filename") per invariant #3.

## Stack and workflow

- Python 3.11+, dependency-managed with `uv` (`pyproject.toml` is the source of truth;
  `requirements.txt` is kept in sync via `uv export --no-hashes -o requirements.txt` for users who
  don't have `uv`, regenerate it whenever dependencies change).
- Key libraries: `ollama`, `pymupdf` (PDF rendering), `pillow` + `pillow-heif` (image decode incl.
  HEIC/HEIF), `openpyxl` (Excel report), `fastapi` + `uvicorn` (local server), `pydantic` (config
  and schema validation), `platformdirs` (config/data directory resolution), `python-dotenv`
  (loads `.env`, see `.env.example`).
- Setup: `uv sync`, then `ollama pull moondream` (and optionally `ollama pull llama3.2-vision:11b`
  for accuracy mode). Run `uv run nomia doctor` to confirm the environment is ready.
- Tests: `uv run pytest` for unit tests. `uv run python tests/benchmark.py` is the **only**
  legitimate source of any accuracy number quoted in the README — never publish a fabricated
  accuracy figure; if a number isn't reproducible from `benchmark.py`'s output, don't quote one.

## Coding conventions

- Small, single-responsibility modules (see the module list in `README.md` / the repo layout) —
  resist the urge to let `pipeline.py` accumulate business logic that belongs in `naming.py` or
  `classify.py`.
- Structured logging throughout (module-level loggers, not `print`), with enough detail that the
  Excel report and the log file tell the same story.
- Graceful per-file error handling everywhere in the pipeline: one corrupt, encrypted, locked, or
  otherwise unreadable file is caught, logged, and routed to `_Unsorted/` — it never aborts the
  batch. The only things allowed to raise all the way up are programmer errors (bad config, bad
  arguments), not bad input files.
- `server.py` and the CLI (`cli.py`) must call the exact same underlying functions in
  `pipeline.py` / `organizer.py` — no duplicated business logic between the two entry points.
- Cross-platform by default: macOS is the primary target, but nothing in `nomia/` should
  hard-depend on macOS-only APIs (e.g. Finder tags). Any OS-specific signal is an optional
  enhancement gated behind a capability check, with graceful degradation elsewhere (e.g. OS
  creation-time availability differs by platform — see `naming.py`'s fallback chain).

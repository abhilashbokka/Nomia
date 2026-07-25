"""Builds a labels.json for tests/benchmark.py from any folder you organize by category —
the "bring your own benchmark" helper. Your files never leave the machine (the whole
pipeline is offline), so this is a safe way to measure accuracy on your real, modern
documents instead of the shipped fixtures.

Layout expected (subfolder names = taxonomy category keys):

    ~/my_real_docs/
      receipt/         IMG_2041.jpg  costco-2026-03.pdf
      invoice/         comcast-bill.pdf
      photo/           beach.jpg
      ...

Usage:
    uv run python tests/make_labels.py ~/my_real_docs
    uv run python tests/benchmark.py --mode fast   --sample-dir ~/my_real_docs
    uv run python tests/benchmark.py --mode tiered --sample-dir ~/my_real_docs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomia.config import NomiaConfig  # noqa: E402
from nomia.extract import IMAGE_EXTENSIONS, PDF_EXTENSIONS  # noqa: E402

SUPPORTED = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def build_labels(root: Path) -> dict[str, dict[str, str]]:
    taxonomy_keys = {cat.key for cat in NomiaConfig().taxonomy}
    labels: dict[str, dict[str, str]] = {}
    owners: dict[str, str] = {}
    skipped_dirs: list[str] = []

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        category = sub.name
        if category.startswith((".", "_")):
            continue
        if category not in taxonomy_keys:
            skipped_dirs.append(category)
            continue
        for f in sorted(sub.iterdir()):
            if not f.is_file() or f.suffix.lower() not in SUPPORTED:
                continue
            # benchmark.py keys items by bare filename, so names must be unique across
            # the whole set - fail loudly rather than silently mislabeling.
            if f.name in owners:
                raise SystemExit(
                    f"Duplicate filename '{f.name}' in both '{owners[f.name]}' and "
                    f"'{category}'. Rename one of them - labels are keyed by filename."
                )
            owners[f.name] = category
            labels[f.name] = {"category": category}

    if skipped_dirs:
        print(f"Skipped non-taxonomy folders: {', '.join(skipped_dirs)}")
        print(f"Valid category keys: {', '.join(sorted(taxonomy_keys))}")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Root folder containing one subfolder per category key.")
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    labels = build_labels(root)
    if not labels:
        raise SystemExit("No supported files found in any category subfolder.")

    out = root / "labels.json"
    out.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")

    counts = Counter(v["category"] for v in labels.values())
    print(f"Wrote {out} with {len(labels)} files:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:20} {n}")
    print(f"\nNext: uv run python tests/benchmark.py --mode fast --sample-dir {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Benchmarks classification accuracy against the labeled set in tests/sample_files/.

This is the ONLY legitimate source of any accuracy number quoted in the README - never a
fabricated figure (see CLAUDE.md). Exercises extract.py + classify.py only (not naming/organizer).

Usage:
    uv run python tests/benchmark.py                  # default model (moondream)
    uv run python tests/benchmark.py --model accuracy # accuracy model (llama3.2-vision:11b)
    uv run python tests/benchmark.py --model both      # both, skipping any that aren't pulled
    uv run python tests/benchmark.py --model llama3.2-vision:11b   # any specific model name
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomia.classify import check_model_available, classify_file  # noqa: E402
from nomia.config import NomiaConfig  # noqa: E402
from nomia.extract import extract_signals  # noqa: E402
from nomia.scanner import scan  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_files"
LABELS_PATH = SAMPLE_DIR / "labels.json"
RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.json"


def load_labels() -> dict:
    if not LABELS_PATH.exists():
        raise SystemExit(f"No labels file at {LABELS_PATH}. Run tests/generate_sample_files.py first.")
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def run_for_model(model: str, labels: dict, cfg: NomiaConfig) -> dict:
    records_by_name = {r.path.name: r for r in scan([SAMPLE_DIR])}
    missing = [name for name in labels if name not in records_by_name]
    if missing:
        raise SystemExit(f"labels.json references sample files that no longer exist: {missing}")

    per_item = []
    latencies = []
    for name, label in sorted(labels.items()):
        record = records_by_name[name]
        signals = extract_signals(record, cfg)
        import time as _time

        t0 = _time.time()
        outcome = classify_file(signals, cfg, model=model)
        elapsed = _time.time() - t0
        latencies.append(elapsed)

        predicted = outcome.result.category if outcome.result else None
        per_item.append({
            "file": name,
            "expected": label["category"],
            "predicted": predicted,
            "confidence": outcome.result.confidence if outcome.result else None,
            "route": outcome.route,
            "correct": predicted == label["category"],
            "elapsed_seconds": round(elapsed, 2),
        })

    return _summarize(model, per_item, latencies)


def _summarize(model: str, per_item: list[dict], latencies: list[float]) -> dict:
    total = len(per_item)
    correct_count = sum(1 for i in per_item if i["correct"])
    accuracy = correct_count / total if total else 0.0

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    for item in per_item:
        expected, predicted = item["expected"], item["predicted"]
        by_category[expected]["support"] += 1
        if predicted == expected:
            by_category[expected]["tp"] += 1
        else:
            by_category[expected]["fn"] += 1
            if predicted is not None:
                by_category[predicted]["fp"] += 1

    category_metrics = {}
    for cat, counts in by_category.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        category_metrics[cat] = {
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "support": counts["support"],
        }

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in per_item:
        confusion[item["expected"]][item["predicted"] or "(failed)"] += 1

    # The engineering-relevant question: does confidence routing actually catch the model's
    # mistakes, or do wrong predictions sneak through at "auto" confidence?
    route_crosstab: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for item in per_item:
        bucket = route_crosstab[item["route"]]
        bucket["correct" if item["correct"] else "incorrect"] += 1

    return {
        "model": model,
        "total_files": total,
        "accuracy": round(accuracy, 3),
        "category_metrics": category_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "route_vs_correctness": {k: dict(v) for k, v in route_crosstab.items()},
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "total": round(sum(latencies), 2),
        },
        "per_item": per_item,
    }


def print_report(summary: dict) -> None:
    print(f"\n=== {summary['model']} ===")
    print(f"Files: {summary['total_files']}   Overall accuracy: {summary['accuracy'] * 100:.1f}%")
    lat = summary["latency_seconds"]
    print(f"Latency: mean {lat['mean']}s/file, total {lat['total']}s\n")

    print(f"{'Category':<20}{'Precision':<12}{'Recall':<10}{'F1':<8}Support")
    for cat, m in sorted(summary["category_metrics"].items()):
        print(f"{cat:<20}{m['precision']:<12}{m['recall']:<10}{m['f1']:<8}{m['support']}")

    print("\nRoute vs. correctness (does confidence routing catch mistakes?):")
    for route, counts in sorted(summary["route_vs_correctness"].items()):
        total_r = counts["correct"] + counts["incorrect"]
        pct = counts["correct"] / total_r * 100 if total_r else 0
        print(f"  {route:<24} correct={counts['correct']:<4} incorrect={counts['incorrect']:<4} ({pct:.0f}% correct)")

    wrong = [i for i in summary["per_item"] if not i["correct"]]
    if wrong:
        print("\nMisclassified files:")
        for item in wrong:
            print(f"  {item['file']:<30} expected={item['expected']:<18} predicted={item['predicted']} "
                  f"(route={item['route']}, confidence={item['confidence']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="default",
        help="'default' (moondream), 'accuracy' (llama3.2-vision:11b), 'both', or a specific model name.",
    )
    args = parser.parse_args()

    cfg = NomiaConfig()
    labels = load_labels()

    if args.model == "both":
        candidates = [cfg.model.default_model, cfg.model.accuracy_model]
    elif args.model == "default":
        candidates = [cfg.model.default_model]
    elif args.model == "accuracy":
        candidates = [cfg.model.accuracy_model]
    else:
        candidates = [args.model]

    all_results = {}
    for model in candidates:
        if not check_model_available(model, host=cfg.model.ollama_host):
            print(f"Skipping '{model}': not pulled. Run `ollama pull {model}` to include it in the benchmark.")
            continue
        print(f"Running benchmark for '{model}' against {len(labels)} labeled file(s)...")
        summary = run_for_model(model, labels, cfg)
        print_report(summary)
        all_results[model] = summary

    if not all_results:
        print("\nNo models were available to benchmark. Pull at least one, e.g.: ollama pull moondream")
        return 1

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

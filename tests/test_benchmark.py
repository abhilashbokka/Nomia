import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import _summarize  # noqa: E402


def test_summarize_computes_accuracy_and_per_category_metrics():
    per_item = [
        {"file": "a.jpg", "expected": "receipt", "predicted": "receipt", "confidence": 0.9, "route": "auto", "correct": True, "elapsed_seconds": 1.0},
        {"file": "b.jpg", "expected": "receipt", "predicted": "invoice", "confidence": 0.6, "route": "review", "correct": False, "elapsed_seconds": 1.5},
        {"file": "c.jpg", "expected": "photo", "predicted": "photo", "confidence": 0.95, "route": "auto", "correct": True, "elapsed_seconds": 0.8},
    ]
    summary = _summarize("moondream", per_item, [1.0, 1.5, 0.8])

    assert summary["total_files"] == 3
    assert summary["accuracy"] == round(2 / 3, 3)
    assert summary["category_metrics"]["receipt"]["support"] == 2
    assert summary["category_metrics"]["photo"]["recall"] == 1.0
    assert summary["latency_seconds"]["total"] == 3.3


def test_summarize_route_vs_correctness_crosstab():
    per_item = [
        {"file": "a.jpg", "expected": "receipt", "predicted": "receipt", "confidence": 0.9, "route": "auto", "correct": True, "elapsed_seconds": 1.0},
        {"file": "b.jpg", "expected": "receipt", "predicted": "invoice", "confidence": 0.85, "route": "auto", "correct": False, "elapsed_seconds": 1.0},
    ]
    summary = _summarize("moondream", per_item, [1.0, 1.0])

    # This is the key signal: a wrong prediction that still landed in "auto" is a confidence
    # miscalibration worth surfacing, not silently averaged away.
    assert summary["route_vs_correctness"]["auto"] == {"correct": 1, "incorrect": 1}


def test_summarize_confusion_matrix_tracks_failed_predictions():
    per_item = [
        {"file": "a.jpg", "expected": "receipt", "predicted": None, "confidence": None, "route": "unsorted", "correct": False, "elapsed_seconds": 1.0},
    ]
    summary = _summarize("moondream", per_item, [1.0])
    assert summary["confusion_matrix"]["receipt"]["(failed)"] == 1


def test_summarize_handles_empty_input():
    summary = _summarize("moondream", [], [])
    assert summary["total_files"] == 0
    assert summary["accuracy"] == 0.0
    assert summary["latency_seconds"]["mean"] is None

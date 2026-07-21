"""Command-line entry point for Nomia.

Subcommands map onto the pipeline stages so the whole pipeline is runnable and testable
headlessly, without the FastAPI/web layer: `doctor`, `config`, `scan`, `plan`, `apply`, `undo`,
`verify`, `report`, `serve`. Each subcommand calls the same underlying functions server.py uses —
no business logic lives here.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from dotenv import load_dotenv

from nomia.config import NomiaConfig, load_config, save_config
from nomia.logging_setup import configure_logging


def cmd_doctor(args: argparse.Namespace) -> int:
    """Checks that the local environment is ready to run Nomia: Ollama reachable, the default
    model pulled, the accuracy model's presence (warn, don't fail, if missing), and that the
    image/PDF libraries import cleanly. Never raises — always prints a clear pass/warn/fail
    report and returns a process exit code."""
    ok = True

    print("Nomia environment check")
    print("=" * 40)

    # --- Ollama reachability + models ---
    try:
        import ollama

        client = ollama.Client(host=_ollama_host())
        models_resp = client.list()
        pulled = {m.model for m in models_resp.models}
        print(f"[OK]   Ollama reachable at {_ollama_host()}")

        default_model = "moondream"
        accuracy_model = "llama3.2-vision:11b"

        if any(m == default_model or m.startswith(default_model + ":") for m in pulled):
            print(f"[OK]   Default model '{default_model}' is pulled")
        else:
            print(f"[FAIL] Default model '{default_model}' is not pulled. Run: ollama pull {default_model}")
            ok = False

        if any(m == accuracy_model or m.startswith(accuracy_model.split(':')[0] + ":") for m in pulled):
            print(f"[OK]   Accuracy model '{accuracy_model}' is pulled")
        else:
            print(
                f"[WARN] Accuracy model '{accuracy_model}' is not pulled yet — accuracy mode "
                f"will be unavailable until you run: ollama pull {accuracy_model}"
            )
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic command, report anything
        print(f"[FAIL] Could not reach Ollama at {_ollama_host()}: {exc}")
        print("       Is the Ollama app/service running? (`ollama serve` or the desktop app)")
        ok = False

    # --- Image/PDF libraries ---
    try:
        import fitz  # PyMuPDF

        print(f"[OK]   PyMuPDF imports cleanly (version {fitz.__version__ if hasattr(fitz, '__version__') else 'unknown'})")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] PyMuPDF failed to import: {exc}")
        ok = False

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        print("[OK]   pillow-heif imports and registers cleanly")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] pillow-heif failed to import: {exc}")
        ok = False

    try:
        from PIL import Image  # noqa: F401

        print("[OK]   Pillow imports cleanly")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Pillow failed to import: {exc}")
        ok = False

    print("=" * 40)
    print("All checks passed." if ok else "Some checks failed — see [FAIL] lines above.")
    return 0 if ok else 1


def _ollama_host() -> str:
    import os

    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def cmd_config_show(args: argparse.Namespace) -> int:
    cfg = load_config(args.config_path)
    print(cfg.model_dump_json(indent=2))
    return 0


def cmd_config_set(args: argparse.Namespace) -> int:
    cfg = load_config(args.config_path)

    if args.destination is not None:
        cfg.destination_root = args.destination
    if args.add_source is not None:
        for src in args.add_source:
            if src not in cfg.source_folders:
                cfg.source_folders.append(src)
    if args.model is not None:
        cfg.model.active_model = args.model
    if args.preset is not None:
        cfg.naming_preset_key = args.preset
    if args.custom_template is not None:
        cfg.naming_preset_key = "custom"
        cfg.custom_template = args.custom_template
    if args.preserve_source is not None:
        cfg.preserve_source = args.preserve_source
    if args.keep_dump_copies is not None:
        cfg.keep_dump_copies = args.keep_dump_copies

    path = save_config(cfg, args.config_path)
    print(f"Saved config to {path}")
    return 0


def _get_journal():
    from nomia.organizer import UndoJournal
    from nomia.paths import journal_db_path

    return UndoJournal(journal_db_path())


def cmd_scan(args: argparse.Namespace) -> int:
    """Pure scanner.py smoke test: walk + hash + dedupe, no config/destination/Ollama needed."""
    from nomia.scanner import dedupe_by_hash, scan

    records = dedupe_by_hash(scan([Path(p) for p in args.paths]))
    dup_count = sum(1 for r in records if r.is_duplicate_of is not None)
    print(f"Scanned {len(records)} file(s), {dup_count} duplicate(s) detected.\n")
    for record in records:
        marker = f"  (duplicate of {record.is_duplicate_of})" if record.is_duplicate_of else ""
        print(f"{record.path}{marker}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from nomia.errors import ModelNotAvailableError
    from nomia.pipeline import build_plan

    cfg = load_config(args.config_path)
    if args.source:
        cfg.source_folders = args.source
    if args.dest:
        cfg.destination_root = args.dest
    if args.source or args.dest:
        save_config(cfg, args.config_path)

    if not cfg.source_folders or not cfg.destination_root:
        print("No source folders / destination configured. Use --source and --dest, or `nomia config set`.")
        return 1

    journal = _get_journal()
    try:
        plan = build_plan(cfg, journal)
    except ModelNotAvailableError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"Run ID: {plan.run_id}")
    print(f"Summary: {plan.summary}\n")
    print(f"{'Source':<50} {'Route':<24} {'Confidence':<11} {'Destination'}")
    for item in plan.items:
        conf = f"{item.confidence:.2f}" if item.confidence is not None else "-"
        dest = str(item.dest_relative_path) if item.dest_relative_path else "(unchanged)"
        print(f"{str(item.record.path):<50} {item.route:<24} {conf:<11} {dest}")
    print(f"\nDry-run only - nothing has been moved. Run `nomia apply {plan.run_id}` to apply it.")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    from nomia.organizer import apply_plan
    from nomia.paths import reports_dir
    from nomia.report import generate_report

    journal = _get_journal()
    run = journal.get_run(args.run_id)
    if run is None:
        print(f"No such run: {args.run_id}")
        return 1

    cfg = NomiaConfig.model_validate(run.config_snapshot_json)
    result = apply_plan(args.run_id, cfg, journal)
    report_path = generate_report(args.run_id, journal, reports_dir() / f"run_{args.run_id}.xlsx")

    print(f"Applied: {result.applied}  Failed: {result.failed}  Skipped: {result.skipped}")
    print(f"Verification: {'OK' if result.verification.ok else 'ISSUES FOUND'} "
          f"({result.verification.hash_matches} hash matches, {len(result.verification.hash_mismatches)} mismatches)")
    print(f"Report: {report_path}")
    if not result.verification.ok:
        print("Run `nomia verify " + args.run_id + "` for full details.")
    return 0 if result.failed == 0 and result.verification.ok else 1


def cmd_undo(args: argparse.Namespace) -> int:
    from nomia.organizer import undo_run

    journal = _get_journal()
    if journal.get_run(args.run_id) is None:
        print(f"No such run: {args.run_id}")
        return 1
    result = undo_run(args.run_id, journal)
    print(f"Undone: {result.undone}  Skipped: {result.skipped}")
    for detail in result.details:
        print(f"  skipped item {detail['item_id']} ({detail['source_path']}): {detail['reason']}")
    return 0 if result.skipped == 0 else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from nomia.organizer import verify_run

    journal = _get_journal()
    if journal.get_run(args.run_id) is None:
        print(f"No such run: {args.run_id}")
        return 1
    report = verify_run(args.run_id, journal)
    print(f"Scanned: {report.scanned_count}  Accounted for: {report.accounted_count}")
    print(f"Hash matches: {report.hash_matches}  Mismatches: {len(report.hash_mismatches)}")
    for mismatch in report.hash_mismatches:
        print(f"  {mismatch['kind']}: {mismatch['source_path']} (expected {mismatch['expected']}, got {mismatch['actual']})")
    print("ALL CHECKS PASSED" if report.ok else "VERIFICATION ISSUES FOUND")
    return 0 if report.ok else 1


def cmd_resume(args: argparse.Namespace) -> int:
    from nomia.organizer import resume_crashed_run

    journal = _get_journal()
    if journal.get_run(args.run_id) is None:
        print(f"No such run: {args.run_id}")
        return 1
    result = resume_crashed_run(args.run_id, journal)
    print(f"Finalized: {result.finalized}  Failed: {result.failed}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from nomia.paths import reports_dir
    from nomia.report import generate_report

    journal = _get_journal()
    if journal.get_run(args.run_id) is None:
        print(f"No such run: {args.run_id}")
        return 1
    out_path = Path(args.out) if args.out else reports_dir() / f"run_{args.run_id}.xlsx"
    path = generate_report(args.run_id, journal, out_path)
    print(f"Wrote report to {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("nomia.server:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nomia", description="A local-first, AI-powered file organizer.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to a config.json to use instead of the default location.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check that Ollama and required libraries are ready.")
    doctor_parser.set_defaults(func=cmd_doctor)

    config_parser = subparsers.add_parser("config", help="Show or edit the persisted configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    config_show_parser = config_subparsers.add_parser("show", help="Print the current configuration as JSON.")
    config_show_parser.set_defaults(func=cmd_config_show)

    config_set_parser = config_subparsers.add_parser("set", help="Update one or more configuration fields.")
    config_set_parser.add_argument("--destination", help="Destination root folder.")
    config_set_parser.add_argument("--add-source", action="append", help="Add a source folder (repeatable).")
    config_set_parser.add_argument("--model", help="Active Ollama model (e.g. moondream, llama3.2-vision:11b).")
    config_set_parser.add_argument("--preset", help="Naming preset key.")
    config_set_parser.add_argument("--custom-template", help="Custom naming template (also switches naming_preset_key to 'custom').")
    config_set_parser.add_argument("--preserve-source", action=argparse.BooleanOptionalAction, default=None, help="Never remove files from source folders (copy-only).")
    config_set_parser.add_argument("--keep-dump-copies", action=argparse.BooleanOptionalAction, default=None, help="Keep verbatim unrenamed backups under _dump/.")
    config_set_parser.set_defaults(func=cmd_config_set)

    scan_parser = subparsers.add_parser("scan", help="Walk and hash-dedupe one or more folders (no classification).")
    scan_parser.add_argument("paths", nargs="+", help="One or more folders to scan.")
    scan_parser.set_defaults(func=cmd_scan)

    plan_parser = subparsers.add_parser("plan", help="Run the full dry-run pipeline and print a preview.")
    plan_parser.add_argument("--source", action="append", help="Source folder (repeatable). Overrides and saves the persisted config if given.")
    plan_parser.add_argument("--dest", help="Destination root. Overrides and saves the persisted config if given.")
    plan_parser.set_defaults(func=cmd_plan)

    apply_parser = subparsers.add_parser("apply", help="Apply a previously planned run.")
    apply_parser.add_argument("run_id")
    apply_parser.set_defaults(func=cmd_apply)

    undo_parser = subparsers.add_parser("undo", help="Undo a previously applied run.")
    undo_parser.add_argument("run_id")
    undo_parser.set_defaults(func=cmd_undo)

    verify_parser = subparsers.add_parser("verify", help="Re-check an applied run's counts and hashes.")
    verify_parser.add_argument("run_id")
    verify_parser.set_defaults(func=cmd_verify)

    resume_parser = subparsers.add_parser("resume", help="Resume/finalize a run interrupted mid-apply.")
    resume_parser.add_argument("run_id")
    resume_parser.set_defaults(func=cmd_resume)

    report_parser = subparsers.add_parser("report", help="(Re)generate the Excel report for a run.")
    report_parser.add_argument("run_id")
    report_parser.add_argument("--out", help="Output .xlsx path (defaults to the standard reports directory).")
    report_parser.set_defaults(func=cmd_report)

    serve_parser = subparsers.add_parser("serve", help="Launch the local web UI + API server.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

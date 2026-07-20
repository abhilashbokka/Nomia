"""One place to configure structured logging conventions for the whole app.

Both the CLI and the FastAPI server call configure_logging() once at startup. Every module
below gets its logger via `logging.getLogger(__name__)` as usual — this module only owns the
handler/formatter setup, not the per-module logger instances.
"""

from __future__ import annotations

import logging
import os
import sys

from nomia.paths import logs_dir

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Idempotent: safe to call multiple times (e.g. once from cli.py, once from server.py
    when the server is launched via the CLI) — only the first call takes effect."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    resolved_level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("nomia")
    root.setLevel(resolved_level)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    try:
        log_path = logs_dir() / "nomia.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Logging to disk is a nice-to-have; never let it prevent the app from starting.
        root.warning("Could not open log file for writing; continuing with console logging only.")

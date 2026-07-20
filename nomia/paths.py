"""Resolves where Nomia keeps its config, data (undo journal, reports, thumbnails), and logs.

Shared by config.py and logging_setup.py so both agree on the same locations without importing
each other. Default is the OS-standard app-data location via `platformdirs`; setting the
NOMIA_HOME environment variable (see .env.example) overrides both config and data dirs to a
single project-local folder, which is convenient for demos and for resetting all state at once.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP_NAME = "nomia"


def _nomia_home() -> Path | None:
    value = os.environ.get("NOMIA_HOME", "").strip()
    return Path(value).expanduser().resolve() if value else None


def config_dir() -> Path:
    home = _nomia_home()
    path = home if home is not None else Path(platformdirs.user_config_dir(APP_NAME))
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    home = _nomia_home()
    path = (home / "data") if home is not None else Path(platformdirs.user_data_dir(APP_NAME))
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_file_path() -> Path:
    return config_dir() / "config.json"


def journal_db_path() -> Path:
    return data_dir() / "journal.sqlite3"


def reports_dir() -> Path:
    path = data_dir() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def thumbnails_dir() -> Path:
    path = data_dir() / "thumbnails"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path

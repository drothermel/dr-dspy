from __future__ import annotations

import os
import re
from pathlib import Path

DSPY_LOG_DIR_ENV = "DSPY_LOG_DIR"
DSPY_RUN_ID_ENV = "DSPY_RUN_ID"
DEFAULT_LOG_DIR = "logs"
DEFAULT_RUN_ID = "default_run"


def slug_run_id(raw: str) -> str:
    slug = re.sub("[^a-zA-Z0-9._-]+", "_", raw.strip())
    return slug or DEFAULT_RUN_ID


def resolve_log_root(call_log_dir: str | None) -> Path:
    if call_log_dir:
        return Path(call_log_dir)
    return Path(os.environ.get(DSPY_LOG_DIR_ENV, DEFAULT_LOG_DIR))


def resolve_run_bucket() -> str:
    return slug_run_id(os.environ.get(DSPY_RUN_ID_ENV, DEFAULT_RUN_ID))

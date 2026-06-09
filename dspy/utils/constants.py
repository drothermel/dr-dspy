import os
from pathlib import Path

_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".dspy_cache")
DSPY_CACHEDIR = os.environ.get("DSPY_CACHEDIR") or _DEFAULT_CACHE_DIR

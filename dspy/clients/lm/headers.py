from typing import Any

from dspy.__metadata__ import __version__


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{__version__}",
        **headers,
    }

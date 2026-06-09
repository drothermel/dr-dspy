from __future__ import annotations

import json
from typing import Any

import json_repair


def load_json(text: str, *, repair: bool) -> Any:
    if repair:
        return json_repair.loads(text)
    return json.loads(text)

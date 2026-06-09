import os

import pytest


def require_env(*keys: str) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing live LM credentials: {', '.join(missing)}")

from typing import Any

from dspy.utils.lazy_import import _detect_dspy_dist


def _missing_gepa_error() -> ImportError:
    dist = _detect_dspy_dist()
    return ImportError(f"gepa is required to use dspy.GEPA. Install with `pip install {dist}[gepa]`.")


def __getattr__(name: str) -> Any:
    if name != "GEPA":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from dspy.teleprompt.gepa.gepa import GEPA as _GEPA
    except ImportError as exc:
        if exc.name is not None and (exc.name == "gepa" or exc.name.startswith("gepa.")):
            raise _missing_gepa_error() from exc
        raise

    globals()["GEPA"] = _GEPA
    return _GEPA

__all__ = ["GEPA"]

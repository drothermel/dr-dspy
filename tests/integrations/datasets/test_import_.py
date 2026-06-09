import builtins
import sys
from collections.abc import Mapping, Sequence

import pytest


def test_import_datasets_raises_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ):
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from dspy.integrations.datasets import import_ as datasets_import

    with pytest.raises(ImportError, match=r"\[datasets\]"):
        datasets_import.import_datasets(feature="Test feature")

"""Legacy dict subclass with attribute access. Prefer typed models for new code."""

import copy

from typing_extensions import override


class dotdict(dict):  # noqa: N801
    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            return super().__getattribute__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    @override
    def __setattr__(self, key, value) -> None:
        if key.startswith("__") and key.endswith("__"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    @override
    def __delattr__(self, key) -> None:
        if key.startswith("__") and key.endswith("__"):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        return dotdict(copy.deepcopy(dict(self), memo))

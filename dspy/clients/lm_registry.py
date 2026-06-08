BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"

_REGISTRY: dict[str, type] | None = None


def _lm_class_registry() -> dict[str, type]:
    global _REGISTRY
    if _REGISTRY is None:
        from dspy.clients.lm.client import LM

        _REGISTRY = {BUILTIN_LM_CLASS_PATH: LM}
    return _REGISTRY


def get_lm_class(class_path: str) -> type:
    return _lm_class_registry()[class_path]

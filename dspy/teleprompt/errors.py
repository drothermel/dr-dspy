from dspy.clients.lm.errors import _safe_litellm_exception_class
from dspy.errors import ContextWindowExceededError, LMInvalidRequestError


def is_demo_shrinkable_error(exc: BaseException) -> bool:
    if isinstance(exc, (ContextWindowExceededError, LMInvalidRequestError, ValueError)):
        return True
    for litellm_name in ("ContextWindowExceededError", "BadRequestError"):
        litellm_cls = _safe_litellm_exception_class(litellm_name)
        if litellm_cls is not None and isinstance(exc, litellm_cls):
            return True
    return False

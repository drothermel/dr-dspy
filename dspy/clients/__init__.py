import logging

logger = logging.getLogger(__name__)


def configure_litellm_logging(level: str = "ERROR") -> None:
    """Configure LiteLLM logging to the specified level."""
    # Litellm uses a global logger called `verbose_logger` to control all loggings.
    from dspy.clients._litellm import get_litellm

    litellm = get_litellm(feature="LiteLLM logging")
    verbose_logger = litellm._logging.verbose_logger

    numeric_logging_level = getattr(logging, level)

    verbose_logger.setLevel(numeric_logging_level)
    for h in verbose_logger.handlers:
        h.setLevel(numeric_logging_level)


def enable_litellm_logging() -> None:
    from dspy.clients._litellm import get_litellm

    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = False
    litellm._dspy_logging_configured = True
    configure_litellm_logging("DEBUG")


def disable_litellm_logging() -> None:
    from dspy.clients._litellm import get_litellm

    litellm = get_litellm(feature="LiteLLM logging")
    litellm.suppress_debug_info = True
    litellm._dspy_logging_configured = True
    configure_litellm_logging("ERROR")

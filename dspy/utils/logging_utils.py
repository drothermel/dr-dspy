import logging
import logging.config
import sys

LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class DSPyLoggingStream:
    def __init__(self) -> None:
        self._enabled = True

    def write(self, text: str) -> None:
        if self._enabled:
            sys.stderr.write(text)

    def flush(self) -> None:
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


DSPY_LOGGING_STREAM = DSPyLoggingStream()


def disable_logging() -> None:
    DSPY_LOGGING_STREAM.enabled = False


def enable_logging() -> None:
    DSPY_LOGGING_STREAM.enabled = True


def configure_dspy_loggers(root_module_name: str) -> None:
    formatter = logging.Formatter(fmt=LOGGING_LINE_FORMAT, datefmt=LOGGING_DATETIME_FORMAT)
    dspy_handler_name = "dspy_handler"
    handler = logging.StreamHandler(stream=DSPY_LOGGING_STREAM)
    handler.setFormatter(formatter)
    handler.set_name(dspy_handler_name)
    logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == dspy_handler_name:
            logger.removeHandler(existing_handler)
    logger.addHandler(handler)

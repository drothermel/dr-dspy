import copy
import os
from collections.abc import Iterator

import pytest

from dspy.dsp.utils.settings import settings
from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs  # noqa: F401

SKIP_DEFAULT_FLAGS = ["reliability", "extra", "llm_call", "deno"]


def _test_settings_config() -> dict:
    from dspy.dsp.utils.settings import DEFAULT_CONFIG

    test_config = copy.deepcopy(DEFAULT_CONFIG)
    test_config["run_log_enabled"] = False
    test_config["transparency"] = "off"
    return test_config


@pytest.fixture(autouse=True)
def clear_settings() -> Iterator[None]:
    import dspy.dsp.utils.settings as settings_module

    settings.configure(**_test_settings_config())
    try:
        yield
    finally:
        settings.configure(**_test_settings_config())
        settings_module.config_owner_async_task = None


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pytest_addoption(parser):
    for flag in SKIP_DEFAULT_FLAGS:
        parser.addoption(f"--{flag}", action="store_true", default=False, help=f"run {flag} tests")


def pytest_configure(config):
    for flag in SKIP_DEFAULT_FLAGS:
        config.addinivalue_line("markers", flag)


def pytest_collection_modifyitems(config, items):
    for flag in SKIP_DEFAULT_FLAGS:
        if config.getoption(f"--{flag}"):
            continue
        skip_mark = pytest.mark.skip(reason=f"need --{flag} option to run")
        for item in items:
            if flag in item.keywords:
                item.add_marker(skip_mark)


@pytest.fixture
def lm_for_test():
    model = os.environ.get("LM_FOR_TEST", None)
    if model is None:
        pytest.skip("LM_FOR_TEST is not set in the environment variables")
    return model

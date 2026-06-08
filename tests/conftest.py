import os

import pytest

from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs  # noqa: F401

SKIP_DEFAULT_FLAGS = ["reliability", "extra", "llm_call", "deno"]


@pytest.fixture
def json_adapter():
    from dspy.adapters.json_adapter import JSONAdapter

    return JSONAdapter()


@pytest.fixture
def run(make_run):
    from dspy.utils.dummies import DummyLM

    return make_run(lm=DummyLM([{}]))


@pytest.fixture
def make_run():
    def _make_run(lm, adapter=None, **kwargs):
        from dspy.adapters.chat_adapter import ChatAdapter
        from dspy.runtime import RunContext, TelemetryConfig

        adapter = adapter or ChatAdapter()
        base_telemetry = TelemetryConfig(transparency="off", run_log_enabled=False)
        telemetry = kwargs.pop("telemetry", None)
        if telemetry is None:
            merged_telemetry = base_telemetry
        elif isinstance(telemetry, TelemetryConfig):
            merged_telemetry = base_telemetry.model_copy(update=telemetry.model_dump(exclude_unset=True))
        else:
            merged_telemetry = base_telemetry.model_copy(update=telemetry)
        init_run_log = kwargs.pop("init_run_log", merged_telemetry.run_log_enabled)
        return RunContext.create(
            lm=lm,
            adapter=adapter,
            telemetry=merged_telemetry,
            init_run_log=init_run_log,
            **kwargs,
        )

    return _make_run


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

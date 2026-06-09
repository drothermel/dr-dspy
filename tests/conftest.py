import os

os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "True")

import pytest

from tests.test_utils.run_binding import collect_run_binding_violations
from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs  # noqa: F401

OPT_IN_MARKERS = ["integration", "llm_call", "deno", "slow"]


@pytest.fixture
def json_adapter():
    from dspy.adapters.json_adapter import JSONAdapter

    return JSONAdapter()


@pytest.fixture
def run(make_run):
    from tests.test_utils import DummyLM

    return make_run(lm=DummyLM([{}]))


@pytest.fixture
def make_run():
    def _make_run(lm, adapter=None, **kwargs):
        from dspy.adapters.chat_adapter import ChatAdapter
        from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode

        adapter = adapter or ChatAdapter()
        base_telemetry = TelemetryConfig(transparency=TransparencyMode.off, call_log=CallLogMode.memory)
        telemetry = kwargs.pop("telemetry", None)
        if telemetry is None:
            merged_telemetry = base_telemetry
        elif isinstance(telemetry, TelemetryConfig):
            merged_telemetry = base_telemetry.model_copy(update=telemetry.model_dump(exclude_unset=True))
        else:
            merged_telemetry = base_telemetry.model_copy(update=telemetry)
        return RunContext.create(
            lm=lm,
            adapter=adapter,
            telemetry=merged_telemetry,
            **kwargs,
        )

    return _make_run


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pytest_addoption(parser):
    for flag in OPT_IN_MARKERS:
        parser.addoption(f"--{flag}", action="store_true", default=False, help=f"run {flag} tests")


def pytest_collection_modifyitems(config, items):
    for flag in OPT_IN_MARKERS:
        if config.getoption(f"--{flag}"):
            continue
        skip_mark = pytest.mark.skip(reason=f"need --{flag} option to run")
        for item in items:
            if flag in item.keywords:
                item.add_marker(skip_mark)


def pytest_collection_finish(session):
    # xdist workers collect in parallel; validate once on the controller.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    violations = collect_run_binding_violations()
    if not violations:
        return
    message = "Unbound run=run in tests (use run fixture or run = make_run(...)):\n" + "\n".join(
        v.format() for v in violations
    )
    raise pytest.UsageError(message)


@pytest.fixture
def lm_for_test():
    model = os.environ.get("LM_FOR_TEST", None)
    if model is None:
        pytest.skip(reason="LM_FOR_TEST is not set in the environment variables")
    return model

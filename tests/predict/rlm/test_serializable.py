"""
Tests for the RLM (Recursive Language Model) module.
"""

import asyncio
import base64

import pytest
from typing_extensions import override

from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.sandbox_serializable import SandboxSerializable
from tests.mock_interpreter import MockInterpreter
from tests.predict.rlm.conftest import (
    _BinarySerializable,
    _StubSerializable,
    make_mock_predictor,
)
from tests.task_spec.helpers import ts


class TestBuildVariablesWithSerializable:
    """Tests for _build_variables with SandboxSerializable inputs."""

    def test_serializable_uses_build_repl_variable(self):
        """SandboxSerializable subclasses route through build_repl_variable."""
        rlm = RLM(ts("data, query -> answer"))
        stub = _StubSerializable("my_data")
        variables = rlm._build_variables(data=stub, query="test query")

        data_var = next(v for v in variables if v.name == "data")
        query_var = next(v for v in variables if v.name == "query")

        assert "StubData(my_data)" in data_var.preview
        assert "test query" in query_var.preview

        # sandbox_setup imports should be surfaced in the description.
        assert "import json" in data_var.desc

    def test_regular_values_unchanged(self):
        """Non-SandboxSerializable values should use default REPLVariable creation."""
        rlm = RLM(ts("context -> answer"))
        variables = rlm._build_variables(context="plain text")
        assert len(variables) == 1
        assert variables[0].name == "context"
        assert "plain text" in variables[0].preview


class TestPrepareSerializableVars:
    """Tests for _prepare_serializable_vars with MockInterpreter."""

    def test_separates_serializable_from_regular(self):
        """Serializable values are injected; regular values are returned."""
        mock = MockInterpreter(responses=["", FinalOutput({"answer": "42"})])
        rlm = RLM(ts("data, query -> answer"), max_iterations=3, interpreter=mock)

        stub = _StubSerializable("payload")

        # Manually call _prepare_serializable_vars
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        regular = rlm._prepare_serializable_vars({"data": stub, "query": "hello"}, mock)

        # Regular args should only contain non-serializable values
        assert "query" in regular
        assert regular["query"] == "hello"
        assert "data" not in regular

        # MockInterpreter should have received an execute call for the setup
        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert "import json" in code
        assert "_raw_data" in variables

    def test_no_serializable_returns_all(self):
        """When no SandboxSerializable values exist, all args are returned."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)

        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        regular = rlm._prepare_serializable_vars({"query": "hello"}, mock)

        assert regular == {"query": "hello"}
        assert mock.call_count == 0

    def test_binary_payload_uses_base64_transport(self):
        """Non-UTF8 bytes should be transported via base64 and decoded in sandbox code."""
        mock = MockInterpreter(responses=[""])
        rlm = RLM(ts("data, query -> answer"), interpreter=mock)

        payload = _BinarySerializable()
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": payload, "query": "hello"}, mock)

        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert "_raw_data = base64.b64decode(_raw_data_base64)" in code
        assert variables["_raw_data_base64"] == base64.b64encode(b"\xff\xfe\xfd").decode("ascii")

    def test_large_payload_not_inlined_in_code(self):
        """Large payloads should ride in the variables kwarg, not the code string.

        Inlining a multi-MB payload into the code text would balloon every
        subsequent prompt and could blow past sandbox limits. The transport
        contract is: code stays small, payload travels as a named variable.
        """
        mock = MockInterpreter(responses=[""])
        rlm = RLM(ts("data, query -> answer"), interpreter=mock)

        large_text = "x" * (2 * 1024 * 1024)  # 2 MB UTF-8 payload

        class _LargeText(SandboxSerializable):
            @override
            def sandbox_setup(self) -> str:
                return ""

            @override
            def to_sandbox(self) -> bytes:
                return large_text.encode("utf-8")

            @override
            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = {data_expr}"

            @override
            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"LargeText({len(large_text)} chars)"

        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": _LargeText(), "query": "hi"}, mock)

        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        # Payload must be in variables, not the code string.
        assert variables["_raw_data"] == large_text
        assert large_text not in code
        assert len(code) < 1000

    def test_forward_with_serializable(self):
        """Full async call with a SandboxSerializable input."""
        mock = MockInterpreter(
            responses=[
                "",  # setup execution for _prepare_serializable_vars
                FinalOutput({"answer": "done"}),
            ]
        )
        rlm = RLM(ts("data, query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Done", "code": 'SUBMIT("done")'},
            ]
        )

        stub = _StubSerializable("test_payload")
        result = asyncio.run(rlm(data=stub, query="test"))
        assert result.answer == "done"

        # First call should be the serializable setup, second should be the iteration
        assert mock.call_count == 2


@pytest.mark.deno
class TestLargeSerializableRoundTrip:
    """End-to-end test that large SandboxSerializable payloads survive the sandbox."""

    def test_large_payload_round_trips_through_real_sandbox(self):
        """A multi-MB payload should be reconstructable inside the real interpreter."""
        large_text = "abc123" * (200 * 1024)  # ~1.2 MB UTF-8

        class _LargeText(SandboxSerializable):
            @override
            def sandbox_setup(self) -> str:
                return ""

            @override
            def to_sandbox(self) -> bytes:
                return large_text.encode("utf-8")

            @override
            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = {data_expr}"

            @override
            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"LargeText({len(large_text)} chars)"

        with PythonInterpreter(tools={}) as interp:
            rlm = RLM(ts("data -> answer"), interpreter=interp)
            rlm._inject_execution_context(interp, rlm._prepare_execution_tools())
            rlm._prepare_serializable_vars({"data": _LargeText()}, interp)
            result = interp.execute("print(len(data)); print(data[:6])")

        assert str(len(large_text)) in result
        assert "abc123" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

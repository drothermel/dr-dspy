import asyncio
import base64

import pytest

from dspy.predict.rlm import RLM
from dspy.primitives import FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.testing import DummyLM
from tests.mock_interpreter import MockInterpreter
from tests.predict.rlm.conftest import _BinarySerializable, _StubSerializable, make_mock_predictor
from tests.task_spec.helpers import ts


class TestBuildVariablesWithSerializable:
    def test_serializable_uses_build_repl_variable(self):
        rlm = RLM(ts("data, query -> answer"))
        stub = _StubSerializable("my_data")
        variables = rlm._build_variables(data=stub, query="test query")
        data_var = next(v for v in variables if v.name == "data")
        query_var = next(v for v in variables if v.name == "query")
        assert "StubData(my_data)" in data_var.preview
        assert "test query" in query_var.preview
        assert "import json" in data_var.desc

    def test_regular_values_unchanged(self, make_run):
        rlm = RLM(ts("context -> answer"))
        variables = rlm._build_variables(context="plain text")
        assert len(variables) == 1
        assert variables[0].name == "context"
        assert "plain text" in variables[0].preview


class TestPrepareSerializableVars:
    def test_separates_serializable_from_regular(self, make_run):
        mock = MockInterpreter(responses=["", FinalOutput({"answer": "42"})])
        rlm = RLM(ts("data, query -> answer"), max_iterations=3, interpreter=mock)
        stub = _StubSerializable("payload")
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        regular = rlm._prepare_serializable_vars({"data": stub, "query": "hello"}, mock)
        assert "query" in regular
        assert regular["query"] == "hello"
        assert "data" not in regular
        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert "import json" in code
        assert "_raw_data" in variables

    def test_no_serializable_returns_all(self, make_run):
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        regular = rlm._prepare_serializable_vars({"query": "hello"}, mock)
        assert regular == {"query": "hello"}
        assert mock.call_count == 0

    def test_binary_payload_uses_base64_transport(self, make_run):
        mock = MockInterpreter(responses=[""])
        rlm = RLM(ts("data, query -> answer"), interpreter=mock)
        payload = _BinarySerializable()
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": payload, "query": "hello"}, mock)
        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert "_raw_data = base64.b64decode(_raw_data_base64)" in code
        assert variables["_raw_data_base64"] == base64.b64encode(b"\xff\xfe\xfd").decode("ascii")

    def test_large_payload_not_inlined_in_code(self, make_run):
        mock = MockInterpreter(responses=[""])
        rlm = RLM(ts("data, query -> answer"), interpreter=mock)
        large_text = "x" * (2 * 1024 * 1024)

        class _LargeText:
            def sandbox_setup(self) -> str:
                return ""

            def to_sandbox(self) -> bytes:
                return large_text.encode("utf-8")

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = {data_expr}"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"LargeText({len(large_text)} chars)"

        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": _LargeText(), "query": "hi"}, mock)
        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert variables["_raw_data"] == large_text
        assert large_text not in code
        assert len(code) < 1000

    def test_forward_with_serializable(self, make_run):
        mock = MockInterpreter(responses=["", FinalOutput({"answer": "done"})])
        rlm = RLM(ts("data, query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Done", "code": 'SUBMIT("done")'}])
        stub = _StubSerializable("test_payload")
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(data=stub, query="test", run=run))
        assert result.answer == "done"
        assert mock.call_count == 2


@pytest.mark.deno
class TestLargeSerializableRoundTrip:
    def test_large_payload_round_trips_through_real_sandbox(self):
        large_text = "abc123" * (200 * 1024)

        class _LargeText:
            def sandbox_setup(self) -> str:
                return ""

            def to_sandbox(self) -> bytes:
                return large_text.encode("utf-8")

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = {data_expr}"

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

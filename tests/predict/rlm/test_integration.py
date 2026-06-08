import asyncio

import pytest

from dspy.adapters.types.tool import Tool
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.rlm import RLM
from tests.predict.rlm.conftest import dummy_lm_context, make_mock_predictor
from tests.task_spec.helpers import ts


@pytest.mark.deno
class TestRLMTypeCoercion:
    @pytest.mark.parametrize(
        ("output_field", "output_type", "code", "expected", "expected_type"),
        [
            ("count", "int", "SUBMIT(42)", 42, int),
            ("score", "float", "SUBMIT(3.14)", 3.14, float),
            ("valid", "bool", "SUBMIT(True)", True, bool),
            ("numbers", "list[int]", "SUBMIT([1, 2, 3])", [1, 2, 3], list),
            ("data", "dict[str, str]", 'SUBMIT({"key": "value"})', {"key": "value"}, dict),
            ("answer", "Literal['yes', 'no']", 'SUBMIT("yes")', "yes", str),
        ],
    )
    def test_type_coercion(self, output_field, output_type, code, expected, expected_type):
        rlm = RLM(ts(f"query -> {output_field}: {output_type}"), max_iterations=3)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Return value", "code": code}])
        result = asyncio.run(rlm(query="test"))
        assert getattr(result, output_field) == expected
        assert isinstance(getattr(result, output_field), expected_type)

    def test_submit_extracts_typed_value(self):
        rlm = RLM(ts("query -> count: int"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Compute and return", "code": "result = 42\nSUBMIT(result)"}]
        )
        result = asyncio.run(rlm(query="count items"))
        assert result.count == 42
        assert isinstance(result.count, int)


@pytest.mark.deno
class TestRLMMultipleOutputs:
    def test_multi_output_final_kwargs(self):
        rlm = RLM(ts("query -> name: str, count: int"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Return both outputs", "code": 'SUBMIT(name="alice", count=5)'}]
        )
        result = asyncio.run(rlm(query="test"))
        assert result.name == "alice"
        assert result.count == 5
        assert isinstance(result.count, int)

    def test_multi_output_final_positional(self):
        rlm = RLM(ts("query -> name: str, count: int"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Return both outputs positionally", "code": 'SUBMIT("bob", 10)'}]
        )
        result = asyncio.run(rlm(query="test"))
        assert result.name == "bob"
        assert result.count == 10

    def test_multi_output_three_fields(self):
        rlm = RLM(ts("query -> name: str, age: int, active: bool"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Return all three", "code": 'SUBMIT(name="carol", age=30, active=True)'}]
        )
        result = asyncio.run(rlm(query="test"))
        assert result.name == "carol"
        assert result.age == 30
        assert result.active is True

    def test_multi_output_final_missing_field_errors(self):
        rlm = RLM(ts("query -> name: str, count: int"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Missing count field", "code": 'SUBMIT(name="alice")'},
                {"reasoning": "Now provide both", "code": 'SUBMIT(name="alice", count=5)'},
            ]
        )
        result = asyncio.run(rlm(query="test"))
        assert result.name == "alice"
        assert result.count == 5

    def test_multi_output_submit_vars(self):
        rlm = RLM(ts("query -> name: str, count: int"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Use SUBMIT", "code": 'n = "dave"\nc = 15\nSUBMIT(n, c)'}]
        )
        result = asyncio.run(rlm(query="test"))
        assert result.name == "dave"
        assert result.count == 15

    def test_multi_output_type_coercion(self):
        rlm = RLM(ts("query -> count: int, ratio: float, flag: bool"), max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Return mixed types", "code": "SUBMIT(count=42, ratio=3.14, flag=True)"}]
        )
        result = asyncio.run(rlm(query="test"))
        assert result.count == 42
        assert isinstance(result.count, int)
        assert result.ratio == 3.14
        assert isinstance(result.ratio, float)
        assert result.flag is True
        assert isinstance(result.flag, bool)


@pytest.mark.deno
class TestRLMWithDummyLM:
    def test_simple_computation_e2e(self):
        with dummy_lm_context([{"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nSUBMIT(result)"}]):
            rlm = RLM(ts("query -> answer: int"), max_iterations=3)
            result = asyncio.run(rlm(query="What is 2 + 3?"))
            assert result.answer == 5
            assert isinstance(result.answer, int)

    def test_multi_turn_computation_e2e(self):
        with dummy_lm_context(
            [
                {"reasoning": "First explore the data", "code": "x = 10\nprint(f'x = {x}')"},
                {"reasoning": "Now compute and return", "code": "y = x * 2\nSUBMIT(y)"},
            ]
        ):
            rlm = RLM(ts("query -> answer: int"), max_iterations=5)
            result = asyncio.run(rlm(query="Double ten"))
            assert result.answer == 20
            assert len(result.trajectory) == 2

    def test_with_input_variables_e2e(self):
        with dummy_lm_context([{"reasoning": "Sum the numbers in the list", "code": "SUBMIT(sum(numbers))"}]):
            rlm = RLM(ts("numbers: list[int] -> total: int"), max_iterations=3)
            result = asyncio.run(rlm(numbers=[1, 2, 3, 4, 5]))
            assert result.total == 15

    def test_with_tool_e2e(self):

        def lookup(key: str) -> str:
            return {"apple": "red", "banana": "yellow"}.get(key, "unknown")

        with dummy_lm_context(
            [{"reasoning": "Look up the color of apple", "code": 'color = lookup(key="apple")\nSUBMIT(color)'}]
        ):
            rlm = RLM(
                ts("fruit -> color: str"),
                max_iterations=3,
                tools=[Tool(lookup, description="Look up a fruit color by key.")],
            )
            result = asyncio.run(rlm(fruit="apple"))
            assert result.color == "red"

    @pytest.mark.asyncio
    async def test_aforward_simple_computation_e2e(self):
        with dummy_lm_context([{"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nSUBMIT(result)"}]):
            rlm = RLM(ts("query -> answer: int"), max_iterations=3)
            result = await rlm.aforward(query="What is 2 + 3?")
            assert result.answer == 5
            assert isinstance(result.answer, int)

    @pytest.mark.asyncio
    async def test_aforward_multi_turn_e2e(self):
        with dummy_lm_context(
            [
                {"reasoning": "First explore the data", "code": "x = 10\nprint(f'x = {x}')"},
                {"reasoning": "Now compute and return", "code": "y = x * 2\nSUBMIT(y)"},
            ]
        ):
            rlm = RLM(ts("query -> answer: int"), max_iterations=5)
            result = await rlm.aforward(query="Double ten")
            assert result.answer == 20
            assert len(result.trajectory) == 2

    @pytest.mark.asyncio
    async def test_aforward_with_input_variables_e2e(self):
        with dummy_lm_context([{"reasoning": "Sum the numbers in the list", "code": "SUBMIT(sum(numbers))"}]):
            rlm = RLM(ts("numbers: list[int] -> total: int"), max_iterations=3)
            result = await rlm.aforward(numbers=[1, 2, 3, 4, 5])
            assert result.total == 15


@pytest.mark.skip(reason="Requires actual LM and Deno - run manually")
class TestRLMIntegration:
    def test_simple_computation(self):
        settings.configure(lm=LM("openai/gpt-4o-mini"))
        rlm = RLM(ts("context, query -> answer"), max_iterations=5)
        result = asyncio.run(rlm(context={"numbers": [1, 2, 3, 4, 5]}, query="What is the sum of the numbers?"))
        assert "15" in result.answer

    def test_with_llm_query(self):
        settings.configure(lm=LM("openai/gpt-4o-mini"))
        rlm = RLM(ts("context, query -> answer"), max_iterations=5)
        result = asyncio.run(
            rlm(
                context="The quick brown fox jumps over the lazy dog.",
                query="Use llm_query to describe what animal is mentioned as lazy.",
            )
        )
        assert "dog" in result.answer.lower()

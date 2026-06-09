from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NoReturn, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic.fields import FieldInfo
from typing_extensions import override

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.format_shared import FIELD_HEADER_PATTERN, FieldInfoWithName
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import provider_tool_call_to_part
from dspy.core.types import LMOutput, LMPart, LMRequest, LMResponse, LMTextPart, LMThinkingPart
from dspy.utils.dotdict import dotdict
from dspy.utils.lazy_import import require

np = require("numpy")


class DummyLM(BaseLM):
    def __init__(
        self,
        answers: list[dict[str, Any]] | dict[str, dict[str, Any]],
        follow_examples: bool = False,
        reasoning: bool = False,
        adapter=None,
    ) -> None:
        super().__init__("dummy", "chat", temperature=0.0, max_tokens=1000, cache=False)
        self.cache = False
        self.answers = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples
        self.reasoning = reasoning
        if adapter is None:
            adapter = ChatAdapter()
        self.adapter = adapter

    def _use_example(self, messages):
        fields = defaultdict(int)
        for message in messages:
            content = getattr(message, "text", None)
            if content and (ma := FIELD_HEADER_PATTERN.match(content)):
                fields[content[ma.start() : ma.end()]] += 1
        max_count = max(fields.values())
        output_fields = [field for field, count in fields.items() if count != max_count]
        final_input = (messages[-1].text or "").split("\n\n")[0]
        for input, output in zip(reversed(messages[:-1]), reversed(messages), strict=False):
            input_content = getattr(input, "text", "") or ""
            output_content = getattr(output, "text", "") or ""
            if any(field in output_content for field in output_fields) and final_input in input_content:
                return output_content
        return None

    @staticmethod
    def _field_info_for_dummy_value(value: object) -> FieldInfo:
        if isinstance(value, bool):
            return FieldInfo(annotation=bool)
        if isinstance(value, int):
            return FieldInfo(annotation=int)
        if isinstance(value, float):
            return FieldInfo(annotation=float)
        if isinstance(value, list):
            if value and all(isinstance(item, str) for item in value):
                return FieldInfo(annotation=list[str])
            return FieldInfo(annotation=list[Any])
        if isinstance(value, dict):
            return FieldInfo(annotation=dict[str, Any])
        return FieldInfo(annotation=str)

    def _format_answer_fields(self, field_names_and_values: dict[str, Any]):
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=self._field_info_for_dummy_value(value)): value
            for field_name, value in field_names_and_values.items()
        }
        adapter = self.adapter
        role = adapter.capabilities.field_value_role
        if role == "assistant":
            return cast("Any", adapter).format_field_with_value(fields_with_values=fields_with_values, role="assistant")
        return adapter.format_field_with_value(fields_with_values=fields_with_values)

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        messages = request.messages
        kwargs = {**self.kwargs, **request.config.model_dump(exclude_none=True)}
        outputs = []
        for _ in range(kwargs.get("n", 1)):
            if self.follow_examples:
                current_output = self._use_example(messages)
            elif isinstance(self.answers, dict):
                answers = cast("dict[str, dict[str, Any]]", self.answers)
                last_message = messages[-1]
                last_content = getattr(last_message, "text", None)
                if last_content is None and isinstance(last_message, dict):
                    last_content = last_message.get("content")
                last_content_str = last_content if isinstance(last_content, str) else ""
                current_output = next(
                    (self._format_answer_fields(v) for k, v in answers.items() if k in last_content_str),
                    "No more responses",
                )
            else:
                if isinstance(self.answers, list):
                    answer = (
                        cast("dict[str, Any]", self.answers.pop(0)) if self.answers else {"answer": "No more responses"}
                    )
                    current_output = self._format_answer_fields(answer)
                else:
                    current_output = self._format_answer_fields(next(self.answers, {"answer": "No more responses"}))
            outputs.append(self._to_output(current_output))
        return LMResponse(
            model="dummy", outputs=outputs, usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

    def _to_output(self, current_output: Any) -> LMOutput:
        if isinstance(current_output, dict):
            parts: list[LMPart] = []
            text = current_output.get("text")
            if isinstance(text, str):
                parts.append(LMTextPart(text=text))
            if self.reasoning and (not any(isinstance(part, LMThinkingPart) for part in parts)):
                parts.append(LMThinkingPart(text="Some reasoning"))
            reasoning_content = current_output.get("reasoning_content")
            if isinstance(reasoning_content, str):
                parts.append(LMThinkingPart(text=reasoning_content))
            parts.extend(provider_tool_call_to_part(tool_call) for tool_call in current_output.get("tool_calls") or [])
            return LMOutput(parts=parts, provider_output=current_output)
        if current_output is None:
            return LMOutput(parts=[])
        parts: list[LMPart] = [LMTextPart(text=str(current_output))]
        if self.reasoning:
            parts.append(LMThinkingPart(text="Some reasoning"))
        return LMOutput(parts=parts, provider_output=current_output)

    def get_convo(self, index):
        entry = self.history[index]
        return (entry.messages_as_openai, entry.outputs)


def dummy_rm(passages=()) -> Callable[..., Any]:
    if not passages:

        def inner(query: str, *, k: int, **kwargs) -> NoReturn:
            raise ValueError("No passages defined")

        return inner
    max_length = max(map(len, passages)) + 100
    vectorizer = DummyVectorizer(max_length)
    passage_vecs = vectorizer(passages)

    def inner(query: str, *, k: int, **kwargs):
        assert k <= len(passages)
        query_vec = vectorizer([query])[0]
        scores = passage_vecs @ query_vec
        largest_idx = (-scores).argsort()[:k]
        return [dotdict(long_text=passages[i]) for i in largest_idx]

    return inner


class DummyVectorizer:
    def __init__(self, max_length=100, n_gram=2) -> None:
        self.max_length = max_length
        self.n_gram = n_gram
        self.P = 10**9 + 7
        random.seed(123)
        self.coeffs = [random.randrange(1, self.P) for _ in range(n_gram)]

    def _hash(self, gram):
        h = 1
        for coeff, c in zip(self.coeffs, gram, strict=False):
            h = h * coeff + ord(c)
            h %= self.P
        return h % self.max_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            grams = [text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)]
            vec = [0] * self.max_length
            for gram in grams:
                vec[self._hash(gram)] += 1
            vecs.append(vec)
        vecs = np.array(vecs, dtype=np.float32)
        vecs -= np.mean(vecs, axis=1, keepdims=True)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs

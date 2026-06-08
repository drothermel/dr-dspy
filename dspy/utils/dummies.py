from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, NoReturn, cast

from dspy.adapters.chat_adapter import FieldInfoWithName, field_header_pattern
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import provider_tool_call_to_part
from dspy.core.types import LMOutput, LMPart, LMRequest, LMResponse, LMTextPart, LMThinkingPart
from dspy.dsp.utils.utils import dotdict
from dspy.signatures.field import OutputField
from dspy.utils.lazy_import import require

np = require("numpy")


class DummyLM(BaseLM):
    """Dummy language model for unit testing purposes.

    Three modes of operation:

    Mode 1: List of dictionaries

    If a list of dictionaries is provided, the dummy model will return the next dictionary
    in the list for each request, formatted according to the `format_field_with_value` function.

    Examples:

    ```
    from dspy.dsp.utils.settings import settings

    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    settings.configure(lm=lm)
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # red
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # blue
    ```

    Mode 2: Dictionary of dictionaries

    If a dictionary of dictionaries is provided, the dummy model will return the value
    corresponding to the key which is contained with the final message of the prompt,
    formatted according to the `format_field_with_value` function from the chat adapter.

    ```
    from dspy.dsp.utils.settings import settings

    lm = DummyLM({"What color is the sky?": {"answer": "blue"}})
    settings.configure(lm=lm)
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # blue
    ```

    Mode 3: Follow examples

    If `follow_examples` is set to True, and the prompt contains an example input exactly equal to the prompt,
    the dummy model will return the output from that example.

    ```
    from dspy.dsp.utils.settings import settings
    from dspy.primitives.example import Example

    lm = DummyLM([{"answer": "red"}], follow_examples=True)
    settings.configure(lm=lm)
    predictor("What color is the sky?", demos=Example(input="What color is the sky?", output="blue"))
    # Output:
    # [[## answer ##]]
    # blue
    ```

    """

    def __init__(
        self,
        answers: list[dict[str, Any]] | dict[str, dict[str, Any]],
        follow_examples: bool = False,
        reasoning: bool = False,
        adapter=None,
    ) -> None:
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.answers = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples
        self.reasoning = reasoning

        # Set adapter, defaulting to ChatAdapter
        if adapter is None:
            from dspy.adapters.chat_adapter import ChatAdapter

            adapter = ChatAdapter()
        self.adapter = adapter

    def _use_example(self, messages):
        # find all field names
        fields = defaultdict(int)
        for message in messages:
            content = getattr(message, "text", None)
            if content and (ma := field_header_pattern.match(content)):
                fields[content[ma.start() : ma.end()]] += 1
        # find the fields which are missing from the final turns
        max_count = max(fields.values())
        output_fields = [field for field, count in fields.items() if count != max_count]

        # get the output from the last turn that has the output fields as headers
        final_input = (messages[-1].text or "").split("\n\n")[0]
        for input, output in zip(reversed(messages[:-1]), reversed(messages), strict=False):
            input_content = getattr(input, "text", "") or ""
            output_content = getattr(output, "text", "") or ""
            if any(field in output_content for field in output_fields) and final_input in input_content:
                return output_content
        return None

    def _format_answer_fields(self, field_names_and_values: dict[str, Any]):
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=OutputField()): value
            for field_name, value in field_names_and_values.items()
        }
        # The reason why DummyLM needs an adapter is because it needs to know which output format to mimic.
        # Normally LMs should not have any knowledge of an adapter, because the output format is defined in the prompt.
        adapter = self.adapter

        # Try to use role="assistant" if the adapter supports it (like JSONAdapter)
        try:
            return adapter.format_field_with_value(fields_with_values, role="assistant")  # ty:ignore[unknown-argument]
        except TypeError:
            # Fallback for adapters that don't support role parameter (like ChatAdapter)
            return adapter.format_field_with_value(fields_with_values)

    def forward(self, request: LMRequest) -> LMResponse:
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
                current_output = self._format_answer_fields(next(self.answers, {"answer": "No more responses"}))  # ty:ignore[invalid-argument-type]

            outputs.append(self._to_output(current_output))

        return LMResponse(
            model="dummy",
            outputs=outputs,
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def aforward(self, request: LMRequest) -> LMResponse:
        return self.forward(request)

    def _to_output(self, current_output: Any) -> LMOutput:
        if isinstance(current_output, dict):
            parts: list[LMPart] = []
            text = current_output.get("text")
            if isinstance(text, str):
                parts.append(LMTextPart(text=text))
            if self.reasoning and not any(isinstance(part, LMThinkingPart) for part in parts):
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
        """Get the prompt + answer from the ith message."""
        return self.history[index]["messages"], self.history[index]["outputs"]


def dummy_rm(passages=()) -> callable:  # ty:ignore[invalid-type-form]
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
    """Simple vectorizer based on n-grams."""

    def __init__(self, max_length=100, n_gram=2) -> None:
        self.max_length = max_length
        self.n_gram = n_gram
        self.P = 10**9 + 7  # A large prime number
        random.seed(123)
        self.coeffs = [random.randrange(1, self.P) for _ in range(n_gram)]

    def _hash(self, gram):
        """Hashes a string using a polynomial hash function."""
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
        vecs /= (
            np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        )  # Epsilon avoids division by zero for empty or constant vectors.
        return vecs

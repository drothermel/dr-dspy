from dspy.adapters.chat_adapter import ChatAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.history import rich_rendering_case
from tests.adapters.scenarios.pydantic_cases import nested_pydantic_chat
from tests.adapters.scenarios.qa import (
    demo_typed_outputs_chat,
    incomplete_demo_chat,
    simple_qa_chat,
)

CHAT_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    GoldenPromptCase(
        id="chat/simple_qa",
        adapter_builder=ChatAdapter,
        scenario=simple_qa_chat(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer the question.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nWhat is the capital of France?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/demo_typed_outputs",
        adapter_builder=ChatAdapter,
        scenario=demo_typed_outputs_chat(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answers` (list[str]): The answers.\n2. `scores` (list[float]): The scores.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answers ## ]]\n{answers}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}\n\n[[ ## scores ## ]]\n{scores}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "number"}}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer the question with multiple answers and scores',
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {
                "role": "assistant",
                "content": '[[ ## answers ## ]]\n["A1", "A2"]\n\n[[ ## scores ## ]]\n[0.1, 0.9]\n\n[[ ## completed ## ]]\n',
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ2\n\nRespond with the corresponding output fields, starting with the field `[[ ## answers ## ]]` (must be formatted as a valid Python list[str]), then `[[ ## scores ## ]]` (must be formatted as a valid Python list[float]), and then ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/nested_pydantic",
        adapter_builder=ChatAdapter,
        scenario=nested_pydantic_chat(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `person` (Person): The person.\nYour output fields are:\n1. `summary` (Summary): The summary.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## person ## ]]\n{person}\n\n[[ ## summary ## ]]\n{summary}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "properties": {"headline": {"type": "string", "title": "Headline"}, "score": {"type": "number", "title": "Score"}}, "required": ["headline", "score"], "title": "Summary"}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `person`, produce the fields `summary`.',
            },
            {
                "role": "user",
                "content": '[[ ## person ## ]]\n{"name": "Ada", "address": {"city": "London", "country": "UK"}, "tags": ["math", "code"]}\n\nRespond with the corresponding output fields, starting with the field `[[ ## summary ## ]]` (must be formatted as a valid Python Summary), and then ending with the marker for `[[ ## completed ## ]]`.',
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/incomplete_demo",
        adapter_builder=ChatAdapter,
        scenario=incomplete_demo_chat(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\n2. `context` (str): The context.\nYour output fields are:\n1. `answer` (str): The answer.\n2. `confidence` (float): The confidence.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## context ## ]]\n{context}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## confidence ## ]]\n{confidence}        # note: the value you produce must be a single float value\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `context`, produce the fields `answer`, `confidence`.",
            },
            {
                "role": "user",
                "content": "This is an example of the task, though some input or output fields are not supplied.\n\n[[ ## question ## ]]\nQ1",
            },
            {
                "role": "assistant",
                "content": "[[ ## answer ## ]]\nA1\n\n[[ ## confidence ## ]]\nNot supplied for this particular example.\n\n[[ ## completed ## ]]\n",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ2\n\n[[ ## context ## ]]\nC2\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, then `[[ ## confidence ## ]]` (must be formatted as a valid Python float), and then ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/history_demo_rich",
        adapter_builder=ChatAdapter,
        scenario=rich_rendering_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `turn_log` (TurnLog): The history.\n2. `image` (Image): The image.\n3. `tools` (list[Tool]): The tools.\n4. `profile` (Profile): The profile.\n5. `question` (str): The question.\nYour output fields are:\n1. `answer` (AnswerCard): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## image ## ]]\n{image}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## profile ## ]]\n{profile}\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, "sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, "required": ["answer", "sources"], "title": "AnswerCard"}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer using all supplied context.',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is an example of the task, though some input or output fields are not supplied.",
                    },
                    {"type": "text", "text": "[[ ## image ## ]]\n"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                    {
                        "type": "text",
                        "text": "\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]",
                    },
                    {
                        "type": "text",
                        "text": '\n\n[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}',
                    },
                    {"type": "text", "text": "\n\n[[ ## question ## ]]\nWhat should we mention?"},
                ],
            },
            {
                "role": "assistant",
                "content": '[[ ## answer ## ]]\n{"answer": "Mention analytical engines.", "sources": ["demo"]}\n\n[[ ## completed ## ]]\n',
            },
            {
                "role": "user",
                "content": '[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n\n[[ ## question ## ]]\nWho is Ada?',
            },
            {
                "role": "assistant",
                "content": '[[ ## answer ## ]]\n{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n\n[[ ## completed ## ]]\n',
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[[ ## image ## ]]\n"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                    {
                        "type": "text",
                        "text": "\n\n[[ ## tools ## ]]\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]",
                    },
                    {
                        "type": "text",
                        "text": '\n\n[[ ## profile ## ]]\n{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, "interests": ["compilers", "navy"]}',
                    },
                    {"type": "text", "text": "\n\n[[ ## question ## ]]\nWhat should the answer include?"},
                    {
                        "type": "text",
                        "text": "\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]` (must be formatted as a valid Python AnswerCard), and then ending with the marker for `[[ ## completed ## ]]`.",
                    },
                ],
            },
        ],
    ),
)

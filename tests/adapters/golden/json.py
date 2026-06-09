from dspy.adapters.json_adapter import JSONAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.history import rich_rendering_case
from tests.adapters.scenarios.json_cases import (
    described_bool_outputs_case,
    int_mapping_outputs_case,
    json_native_tool_calling_case,
    json_non_native_tool_history_case,
    literal_enum_outputs_case,
    tool_calls_output_demo_case,
)
from tests.adapters.scenarios.pydantic_cases import nested_pydantic_json
from tests.adapters.scenarios.qa import (
    demo_typed_outputs_json,
    incomplete_demo_json,
    simple_qa_json,
)


def json_native_tool_calling_adapter_builder() -> JSONAdapter:
    return JSONAdapter(use_native_function_calling=True)


def json_non_native_tool_history_adapter_builder() -> JSONAdapter:
    return JSONAdapter(use_native_function_calling=False)


def json_tool_calls_output_demo_adapter_builder() -> JSONAdapter:
    return JSONAdapter(use_native_function_calling=False)


JSON_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    GoldenPromptCase(
        id="json/simple_qa",
        adapter_builder=JSONAdapter,
        scenario=simple_qa_json(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.',
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nWhat is the capital of France?\n\nRespond with a JSON object in the following order of fields: `answer`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/demo_typed_output",
        adapter_builder=JSONAdapter,
        scenario=demo_typed_outputs_json(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\n2. `confidence` (float): The confidence.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}",\n  "confidence": "{confidence}        # note: the value you produce must be a single float value"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`, `confidence`.',
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {"role": "assistant", "content": '{\n  "answer": "A1",\n  "confidence": 0.9\n}'},
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ2\n\nRespond with a JSON object in the following order of fields: `answer`, then `confidence` (must be formatted as a valid Python float).",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/history_demo_rich",
        adapter_builder=JSONAdapter,
        scenario=rich_rendering_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `turn_log` (TurnLog): The history.\n2. `image` (Image): The image.\n3. `tools` (list[Tool]): The tools.\n4. `profile` (Profile): The profile.\n5. `question` (str): The question.\nYour output fields are:\n1. `answer` (AnswerCard): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## image ## ]]\n{image}\n\n[[ ## tools ## ]]\n{tools}\n\n[[ ## profile ## ]]\n{profile}\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"answer\\": {\\"type\\": \\"string\\", \\"title\\": \\"Answer\\"}, \\"sources\\": {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}, \\"title\\": \\"Sources\\"}}, \\"required\\": [\\"answer\\", \\"sources\\"], \\"title\\": \\"AnswerCard\\"}"\n}\nIn adhering to this structure, your objective is: \n        Answer using all supplied context.',
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
                "content": '{\n  "answer": {\n    "answer": "Mention analytical engines.",\n    "sources": [\n      "demo"\n    ]\n  }\n}',
            },
            {
                "role": "user",
                "content": '[[ ## profile ## ]]\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n\n[[ ## question ## ]]\nWho is Ada?',
            },
            {
                "role": "assistant",
                "content": '{\n  "answer": {\n    "answer": "Ada is a mathematician.",\n    "sources": [\n      "memory"\n    ]\n  }\n}',
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
                        "text": "\n\nRespond with a JSON object in the following order of fields: `answer` (must be formatted as a valid Python AnswerCard).",
                    },
                ],
            },
        ],
    ),
    GoldenPromptCase(
        id="json/nested_pydantic",
        adapter_builder=JSONAdapter,
        scenario=nested_pydantic_json(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `summary` (JsonNestedSummary): The summary.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "summary": "{summary}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"JsonNestedAddress\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"city\\": {\\"type\\": \\"string\\", \\"title\\": \\"City\\"}, \\"country\\": {\\"type\\": \\"string\\", \\"title\\": \\"Country\\"}}, \\"required\\": [\\"city\\", \\"country\\"], \\"title\\": \\"JsonNestedAddress\\"}}, \\"properties\\": {\\"address\\": {\\"$ref\\": \\"#/$defs/JsonNestedAddress\\"}, \\"scores\\": {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}, \\"title\\": \\"Scores\\"}, \\"title\\": {\\"type\\": \\"string\\", \\"title\\": \\"Title\\"}}, \\"required\\": [\\"title\\", \\"address\\", \\"scores\\"], \\"title\\": \\"JsonNestedSummary\\"}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `summary`.',
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nSummarize\n\nRespond with a JSON object in the following order of fields: `summary` (must be formatted as a valid Python JsonNestedSummary).",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/incomplete_demo",
        adapter_builder=JSONAdapter,
        scenario=incomplete_demo_json(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\n2. `context` (str): The context.\nYour output fields are:\n1. `answer` (str): The answer.\n2. `score` (float): The score.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\n[[ ## context ## ]]\n{context}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}",\n  "score": "{score}        # note: the value you produce must be a single float value"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `context`, produce the fields `answer`, `score`.',
            },
            {
                "role": "user",
                "content": "This is an example of the task, though some input or output fields are not supplied.\n\n[[ ## question ## ]]\nQ1",
            },
            {
                "role": "assistant",
                "content": '{\n  "answer": "A1",\n  "score": "Not supplied for this particular example. "\n}',
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ2\n\n[[ ## context ## ]]\nC2\n\nRespond with a JSON object in the following order of fields: `answer`, then `score` (must be formatted as a valid Python float).",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/described_bool_outputs",
        adapter_builder=JSONAdapter,
        scenario=described_bool_outputs_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `input1` (str): The input 1.\nYour output fields are:\n1. `output1` (str): String output field\n2. `output2` (bool): The output 2.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## input1 ## ]]\n{input1}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "output1": "{output1}",\n  "output2": "{output2}        # note: the value you produce must be True or False"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `input1`, produce the fields `output1`, `output2`.',
            },
            {
                "role": "user",
                "content": "[[ ## input1 ## ]]\nTest input\n\nRespond with a JSON object in the following order of fields: `output1`, then `output2` (must be formatted as a valid Python bool).",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/int_mapping_outputs",
        adapter_builder=JSONAdapter,
        scenario=int_mapping_outputs_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `count` (int): The count.\n2. `metadata` (dict[str, int]): The metadata.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "count": "{count}        # note: the value you produce must be a single int value",\n  "metadata": "{metadata}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"additionalProperties\\": {\\"type\\": \\"integer\\"}}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `count`, `metadata`.',
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nCount things\n\nRespond with a JSON object in the following order of fields: `count` (must be formatted as a valid Python int), then `metadata` (must be formatted as a valid Python dict[str, int]).",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/literal_enum_outputs",
        adapter_builder=JSONAdapter,
        scenario=literal_enum_outputs_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `text` (str): The text.\nYour output fields are:\n1. `decision` (Literal[\'accept\', \'reject\']): The decision.\n2. `label` (Label): The label.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## text ## ]]\n{text}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "decision": "{decision}        # note: the value you produce must exactly match (no extra characters) one of: accept; reject",\n  "label": "{label}        # note: the value you produce must be one of: positive; negative"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `text`, produce the fields `decision`, `label`.',
            },
            {
                "role": "user",
                "content": "[[ ## text ## ]]\nLooks good\n\nRespond with a JSON object in the following order of fields: `decision` (must be formatted as a valid Python Literal['accept', 'reject']), then `label` (must be formatted as a valid Python Label).",
            },
        ],
    ),
    GoldenPromptCase(
        id="json/native_tool_calling",
        adapter_builder=json_native_tool_calling_adapter_builder,
        scenario=json_native_tool_calling_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `tools`, produce the fields `tool_calls`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ?\n\nRespond with a JSON object in the following order of fields: .",
            },
        ],
        lm_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for documents.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 3}},
                            "required": ["query"],
                        },
                    },
                }
            ]
        },
    ),
    GoldenPromptCase(
        id="json/tool_calls_output_demo",
        adapter_builder=json_tool_calls_output_demo_adapter_builder,
        scenario=tool_calls_output_demo_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `tool_calls` (ToolCalls): The tool calls.\n    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "tool_calls": "{tool_calls}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"ToolCall\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"args\\": {\\"type\\": \\"object\\", \\"additionalProperties\\": true, \\"title\\": \\"Args\\"}, \\"name\\": {\\"type\\": \\"string\\", \\"title\\": \\"Name\\"}}, \\"required\\": [\\"name\\", \\"args\\"], \\"title\\": \\"ToolCall\\"}}, \\"properties\\": {\\"tool_calls\\": {\\"type\\": \\"array\\", \\"items\\": {\\"$ref\\": \\"#/$defs/ToolCall\\"}, \\"title\\": \\"Tool Calls\\"}}, \\"required\\": [\\"tool_calls\\"], \\"title\\": \\"ToolCalls\\"}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `tool_calls`.',
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {
                "role": "assistant",
                "content": '{\n  "tool_calls": {\n    "tool_calls": [\n      {\n        "name": "search",\n        "args": {\n          "query": "cats"\n        }\n      }\n    ]\n  }\n}',
            },
            {
                "role": "user",
                "content": '[[ ## question ## ]]\nQ2\n\nRespond with a JSON object in the following order of fields: `tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).',
            },
        ],
    ),
    GoldenPromptCase(
        id="json/non_native_tool_history",
        adapter_builder=json_non_native_tool_history_adapter_builder,
        scenario=json_non_native_tool_history_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\n2. `turn_log` (TurnLog): The history.\n3. `tools` (list[Tool]): The tools.\nYour output fields are:\n1. `next_thought` (str): The next thought.\n2. `tool_calls` (ToolCalls): The tool calls.\n    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\n[[ ## turn_log ## ]]\n{turn_log}\n\n[[ ## tools ## ]]\n{tools}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "next_thought": "{next_thought}",\n  "tool_calls": "{tool_calls}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"ToolCall\\": {\\"type\\": \\"object\\", \\"properties\\": {\\"args\\": {\\"type\\": \\"object\\", \\"additionalProperties\\": true, \\"title\\": \\"Args\\"}, \\"name\\": {\\"type\\": \\"string\\", \\"title\\": \\"Name\\"}}, \\"required\\": [\\"name\\", \\"args\\"], \\"title\\": \\"ToolCall\\"}}, \\"properties\\": {\\"tool_calls\\": {\\"type\\": \\"array\\", \\"items\\": {\\"$ref\\": \\"#/$defs/ToolCall\\"}, \\"title\\": \\"Tool Calls\\"}}, \\"required\\": [\\"tool_calls\\"], \\"title\\": \\"ToolCalls\\"}"\n}\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, `tool_calls`.',
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {
                "role": "assistant",
                "content": '{\n  "next_thought": "I should search.",\n  "tool_calls": {\n    "tool_calls": [\n      {\n        "name": "search",\n        "args": {\n          "query": "cats"\n        },\n        "id": "call_1"\n      }\n    ]\n  }\n}',
            },
            {
                "role": "user",
                "content": '[[ ## tool_call_results ## ]]\n{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}',
            },
            {
                "role": "user",
                "content": '[[ ## question ## ]]\nQ2\n\n[[ ## tools ## ]]\n["search, whose description is <desc>Search for documents.</desc>. It takes arguments {\'query\': {\'type\': \'string\'}}."]\n\nRespond with a JSON object in the following order of fields: `next_thought`, then `tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).',
            },
        ],
    ),
)

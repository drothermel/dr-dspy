from dspy.adapters.json_adapter import JSONAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.history import rich_rendering_case
from tests.adapters.scenarios.pydantic_cases import nested_pydantic_json
from tests.adapters.scenarios.qa import (
    demo_typed_outputs_json,
    incomplete_demo_json,
    simple_qa_json,
)

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
)

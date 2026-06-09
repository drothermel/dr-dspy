from dspy.adapters.xml_adapter import XMLAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.history import rich_rendering_case
from tests.adapters.scenarios.pydantic_cases import nested_pydantic_xml
from tests.adapters.scenarios.qa import (
    demo_typed_outputs_xml,
    incomplete_demo_xml,
    simple_qa_xml,
)

XML_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    GoldenPromptCase(
        id="xml/simple_qa",
        adapter_builder=XMLAdapter,
        scenario=simple_qa_xml(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": "<question>\nwhy did a chicken cross the kitchen?\n</question>\n\nRespond with the corresponding output fields wrapped in XML tags `<answer>`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="xml/demo_typed_output",
        adapter_builder=XMLAdapter,
        scenario=demo_typed_outputs_xml(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\n2. `score` (float): The score.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>\n\n<score>\n{score}        # note: the value you produce must be a single float value\n</score>\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`, `score`.",
            },
            {"role": "user", "content": "<question>\nQ1\n</question>"},
            {"role": "assistant", "content": "<answer>\nA1\n</answer>\n\n<score>\n0.9\n</score>"},
            {
                "role": "user",
                "content": "<question>\nQ2\n</question>\n\nRespond with the corresponding output fields wrapped in XML tags `<answer>`, then `<score>`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="xml/history_demo_rich",
        adapter_builder=XMLAdapter,
        scenario=rich_rendering_case(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `turn_log` (TurnLog): The history.\n2. `image` (Image): The image.\n3. `tools` (list[Tool]): The tools.\n4. `profile` (Profile): The profile.\n5. `question` (str): The question.\nYour output fields are:\n1. `answer` (AnswerCard): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<turn_log>\n{turn_log}\n</turn_log>\n\n<image>\n{image}\n</image>\n\n<tools>\n{tools}\n</tools>\n\n<profile>\n{profile}\n</profile>\n\n<question>\n{question}\n</question>\n\n<answer>\n{answer}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "properties": {"answer": {"type": "string", "title": "Answer"}, "sources": {"type": "array", "items": {"type": "string"}, "title": "Sources"}}, "required": ["answer", "sources"], "title": "AnswerCard"}\n</answer>\nIn adhering to this structure, your objective is: \n        Answer using all supplied context.',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is an example of the task, though some input or output fields are not supplied.",
                    },
                    {"type": "text", "text": "<image>\n"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                    {"type": "text", "text": "\n</image>"},
                    {
                        "type": "text",
                        "text": "\n\n<tools>\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]\n</tools>",
                    },
                    {
                        "type": "text",
                        "text": '\n\n<profile>\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n</profile>',
                    },
                    {"type": "text", "text": "\n\n<question>\nWhat should we mention?\n</question>"},
                ],
            },
            {
                "role": "assistant",
                "content": '<answer>\n{"answer": "Mention analytical engines.", "sources": ["demo"]}\n</answer>',
            },
            {
                "role": "user",
                "content": '<profile>\n{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": ["math", "machines"]}\n</profile>\n\n<question>\nWho is Ada?\n</question>',
            },
            {
                "role": "assistant",
                "content": '<answer>\n{"answer": "Ada is a mathematician.", "sources": ["memory"]}\n</answer>',
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image>\n"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                    {"type": "text", "text": "\n</image>"},
                    {
                        "type": "text",
                        "text": "\n\n<tools>\n[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]\n</tools>",
                    },
                    {
                        "type": "text",
                        "text": '\n\n<profile>\n{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, "interests": ["compilers", "navy"]}\n</profile>',
                    },
                    {"type": "text", "text": "\n\n<question>\nWhat should the answer include?\n</question>"},
                    {
                        "type": "text",
                        "text": "\n\nRespond with the corresponding output fields wrapped in XML tags `<answer>`.",
                    },
                ],
            },
        ],
    ),
    GoldenPromptCase(
        id="xml/nested_pydantic",
        adapter_builder=XMLAdapter,
        scenario=nested_pydantic_xml(),
        messages=[
            {
                "role": "system",
                "content": 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `summary` (XmlSummary): The summary.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<question>\n{question}\n</question>\n\n<summary>\n{summary}        # note: the value you produce must adhere to the JSON schema: {"type": "object", "$defs": {"XmlAddress": {"type": "object", "properties": {"city": {"type": "string", "title": "City"}, "country": {"type": "string", "title": "Country"}}, "required": ["city", "country"], "title": "XmlAddress"}}, "properties": {"address": {"$ref": "#/$defs/XmlAddress"}, "title": {"type": "string", "title": "Title"}}, "required": ["title", "address"], "title": "XmlSummary"}\n</summary>\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `summary`.',
            },
            {
                "role": "user",
                "content": "<question>\nSummarize\n</question>\n\nRespond with the corresponding output fields wrapped in XML tags `<summary>`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="xml/incomplete_demo",
        adapter_builder=XMLAdapter,
        scenario=incomplete_demo_xml(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\n2. `context` (str): The context.\nYour output fields are:\n1. `answer` (str): The answer.\n2. `score` (float): The score.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n<question>\n{question}\n</question>\n\n<context>\n{context}\n</context>\n\n<answer>\n{answer}\n</answer>\n\n<score>\n{score}        # note: the value you produce must be a single float value\n</score>\nIn adhering to this structure, your objective is: \n        Given the fields `question`, `context`, produce the fields `answer`, `score`.",
            },
            {
                "role": "user",
                "content": "This is an example of the task, though some input or output fields are not supplied.\n\n<question>\nQ1\n</question>",
            },
            {
                "role": "assistant",
                "content": "<answer>\nA1\n</answer>\n\n<score>\nNot supplied for this particular example. \n</score>",
            },
            {
                "role": "user",
                "content": "<question>\nQ2\n</question>\n\n<context>\nC2\n</context>\n\nRespond with the corresponding output fields wrapped in XML tags `<answer>`, then `<score>`.",
            },
        ],
    ),
)

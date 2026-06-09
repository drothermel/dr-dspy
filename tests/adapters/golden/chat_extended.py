from dspy.adapters.chat_adapter import ChatAdapter
from tests.adapters.assertions import normalize_citations_messages
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.chat_cases import (
    base_custom_type_case,
    citations_output_demo_case,
    history_case,
    list_value_string_case,
    literal_output_case,
    multimodal_custom_type_case,
    native_citations_case,
    native_reasoning_case,
    native_tool_calling_case,
    non_native_tool_history_case,
    passthrough_lm_kwargs_case,
    reasoning_code_outputs_case,
    tool_input_case,
)


def native_tool_calling_adapter_builder() -> ChatAdapter:
    return ChatAdapter(use_native_function_calling=True)


def non_native_tool_history_adapter_builder() -> ChatAdapter:
    return ChatAdapter(use_native_function_calling=False)


CHAT_EXTENDED_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    GoldenPromptCase(
        id="chat/history",
        adapter_builder=ChatAdapter,
        scenario=history_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `turn_log` (TurnLog): The history.\n"
                "2. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## turn_log ## ]]\n"
                "{turn_log}\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `turn_log`, `question`, produce the fields `answer`.",
            },
            {"role": "user", "content": "[[ ## question ## ]]\nWhat is 1+1?"},
            {"role": "assistant", "content": "[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n"},
            {"role": "user", "content": "[[ ## question ## ]]\nWhat is 2+2?"},
            {"role": "assistant", "content": "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]\n"},
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "What is 3+3?\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then "
                "ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/list_value_string",
        adapter_builder=ChatAdapter,
        scenario=list_value_string_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `context` (str): The context.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## context ## ]]\n"
                "{context}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `context`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": "[[ ## context ## ]]\n"
                "[1] «alpha»\n"
                "[2] «beta»\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then "
                "ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/literal_output",
        adapter_builder=ChatAdapter,
        scenario=literal_output_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `verdict` (Literal['yes', 'no']): The verdict.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## verdict ## ]]\n"
                "{verdict}        # note: the value you produce must exactly match (no extra characters) one of: yes; no\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, produce the fields `verdict`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Is the sky blue?\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## verdict ## ]]` (must be "
                "formatted as a valid Python Literal['yes', 'no']), and then ending with the marker for `[[ ## completed "
                "## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/multimodal_custom_type",
        adapter_builder=ChatAdapter,
        scenario=multimodal_custom_type_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `image` (Image): The image.\n"
                "2. `audio` (Audio): The audio.\n"
                "3. `file` (File): The file.\n"
                "4. `document` (Document): The document.\n"
                "    Type description of Document: A document containing text content that can be referenced and cited. "
                "Include the full text content and optionally a title for proper referencing.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## image ## ]]\n"
                "{image}\n"
                "\n"
                "[[ ## audio ## ]]\n"
                "{audio}\n"
                "\n"
                "[[ ## file ## ]]\n"
                "{file}\n"
                "\n"
                "[[ ## document ## ]]\n"
                "{document}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `image`, `audio`, `file`, `document`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[[ ## image ## ]]\n"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    {"type": "text", "text": "\n\n[[ ## audio ## ]]\n"},
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                    {"type": "text", "text": "\n\n[[ ## file ## ]]\n"},
                    {"type": "file", "file": {"file_id": "file-123", "filename": "notes.txt"}},
                    {"type": "text", "text": "\n\n[[ ## document ## ]]\n"},
                    {
                        "type": "document",
                        "source": {"type": "text", "media_type": "text/plain", "data": "Alpha beta"},
                        "citations": {"enabled": True},
                        "title": "Doc",
                    },
                    {
                        "type": "text",
                        "text": "\n"
                        "\n"
                        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, "
                        "and then ending with the marker for `[[ ## completed ## ]]`.",
                    },
                ],
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/base_custom_type",
        adapter_builder=ChatAdapter,
        scenario=base_custom_type_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `event` (Event): The event.\n"
                "    Type description of Event: An event block.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## event ## ]]\n"
                "{event}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `event`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[[ ## event ## ]]\n"},
                    {"type": "event", "event": {"label": "launch"}},
                    {
                        "type": "text",
                        "text": "\n"
                        "\n"
                        "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, "
                        "and then ending with the marker for `[[ ## completed ## ]]`.",
                    },
                ],
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/citations_output_demo",
        adapter_builder=ChatAdapter,
        scenario=citations_output_demo_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `citations` (Citations): The citations.\n"
                "    Type description of Citations: Citations with quoted text and source references. Include the exact "
                "text being cited and information about its source.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## citations ## ]]\n"
                '{citations}        # note: the value you produce must adhere to the JSON schema: {"type": "object", '
                '"$defs": {"Citation": {"type": "object", "properties": {"type": {"type": "string", "default": '
                '"char_location", "title": "Type"}, "cited_text": {"type": "string", "title": "Cited Text"}, '
                '"document_index": {"type": "integer", "title": "Document Index"}, "document_title": {"anyOf": [{"type": '
                '"string"}, {"type": "null"}], "default": null, "title": "Document Title"}, "end_char_index": {"type": '
                '"integer", "title": "End Char Index"}, "start_char_index": {"type": "integer", "title": "Start Char '
                'Index"}, "supported_text": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": '
                '"Supported Text"}}, "required": ["cited_text", "document_index", "start_char_index", "end_char_index"], '
                '"title": "Citation"}}, "properties": {"citations": {"type": "array", "items": {"$ref": '
                '"#/$defs/Citation"}, "title": "Citations"}}, "required": ["citations"], "title": "Citations"}\n'
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, produce the fields `citations`.",
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {
                "role": "assistant",
                "content": "[[ ## citations ## ]]\n"
                '[{"type": "char_location", "cited_text": "alpha", "document_index": 0, "start_char_index": 0, '
                '"end_char_index": 5}]\n'
                "\n"
                "[[ ## completed ## ]]\n",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q2\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## citations ## ]]` (must be "
                "formatted as a valid Python Citations), and then ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
        normalize=normalize_citations_messages,
    ),
    GoldenPromptCase(
        id="chat/native_citations",
        adapter_builder=ChatAdapter,
        scenario=native_citations_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, produce the fields `answer`, `citations`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q?\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then "
                "ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/passthrough_lm_kwargs",
        adapter_builder=ChatAdapter,
        scenario=passthrough_lm_kwargs_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q?\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then "
                "ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
        lm_kwargs={"temperature": 0.7, "max_tokens": 42, "stream": True},
    ),
    GoldenPromptCase(
        id="chat/native_reasoning",
        adapter_builder=ChatAdapter,
        scenario=native_reasoning_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, produce the fields `reasoning`, `answer`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q?\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then "
                "ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
        lm_kwargs={"reasoning_effort": "low"},
    ),
    GoldenPromptCase(
        id="chat/reasoning_code_outputs",
        adapter_builder=ChatAdapter,
        scenario=reasoning_code_outputs_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "1. `reasoning` (Reasoning): The reasoning.\n"
                "2. `code` (Code_python): The code.\n"
                "    Type description of Code_python: Code represented in a string, specified in the `code` field. If "
                "this is an output field, the code field should follow the markdown code block format, e.g. \n"
                "```python\n"
                "{code}\n"
                "```\n"
                "Programming language: python\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## reasoning ## ]]\n"
                "{reasoning}\n"
                "\n"
                "[[ ## code ## ]]\n"
                "{code}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, produce the fields `reasoning`, `code`.",
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {
                "role": "assistant",
                "content": "[[ ## reasoning ## ]]\nThink\n\n[[ ## code ## ]]\nprint('hi')\n\n[[ ## completed ## ]]\n",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q2\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]` (must be "
                "formatted as a valid Python Reasoning), then `[[ ## code ## ]]` (must be formatted as a valid Python "
                "Code_python), and then ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/native_tool_calling",
        adapter_builder=native_tool_calling_adapter_builder,
        scenario=native_tool_calling_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "Your output fields are:\n"
                "\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, `tools`, produce the fields `tool_calls`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q?\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field , and then ending with the marker "
                "for `[[ ## completed ## ]]`.",
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
        id="chat/non_native_tool_history",
        adapter_builder=non_native_tool_history_adapter_builder,
        scenario=non_native_tool_history_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "2. `turn_log` (TurnLog): The history.\n"
                "3. `tools` (list[Tool]): The tools.\n"
                "Your output fields are:\n"
                "1. `next_thought` (str): The next thought.\n"
                "2. `tool_calls` (ToolCalls): The tool calls.\n"
                "    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, a list of calls. "
                'Each call must include `name` and `args`. Example: {"tool_calls": [{"name": "search", "args": {"query": '
                '"cats"}}]}\n'
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## turn_log ## ]]\n"
                "{turn_log}\n"
                "\n"
                "[[ ## tools ## ]]\n"
                "{tools}\n"
                "\n"
                "[[ ## next_thought ## ]]\n"
                "{next_thought}\n"
                "\n"
                "[[ ## tool_calls ## ]]\n"
                '{tool_calls}        # note: the value you produce must adhere to the JSON schema: {"type": "object", '
                '"$defs": {"ToolCall": {"type": "object", "properties": {"args": {"type": "object", '
                '"additionalProperties": true, "title": "Args"}, "name": {"type": "string", "title": "Name"}}, '
                '"required": ["name", "args"], "title": "ToolCall"}}, "properties": {"tool_calls": {"type": "array", '
                '"items": {"$ref": "#/$defs/ToolCall"}, "title": "Tool Calls"}}, "required": ["tool_calls"], "title": '
                '"ToolCalls"}\n'
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, `turn_log`, `tools`, produce the fields `next_thought`, "
                "`tool_calls`.",
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {
                "role": "assistant",
                "content": "[[ ## next_thought ## ]]\n"
                "I should search.\n"
                "\n"
                "[[ ## tool_calls ## ]]\n"
                '{"tool_calls": [{"name": "search", "args": {"query": "cats"}, "id": "call_1"}]}\n'
                "\n"
                "[[ ## completed ## ]]\n",
            },
            {
                "role": "user",
                "content": "[[ ## tool_call_results ## ]]\n"
                '{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}',
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q2\n"
                "\n"
                "[[ ## tools ## ]]\n"
                "[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': "
                "{'type': 'string'}}.\"]\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## next_thought ## ]]`, then "
                '`[[ ## tool_calls ## ]]` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}), '
                "and then ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="chat/tool_input",
        adapter_builder=ChatAdapter,
        scenario=tool_input_case(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n"
                "1. `question` (str): The question.\n"
                "2. `tools` (list[Tool]): The tools.\n"
                "Your output fields are:\n"
                "1. `answer` (str): The answer.\n"
                "All interactions will be structured in the following way, with the appropriate values filled in.\n"
                "\n"
                "[[ ## question ## ]]\n"
                "{question}\n"
                "\n"
                "[[ ## tools ## ]]\n"
                "{tools}\n"
                "\n"
                "[[ ## answer ## ]]\n"
                "{answer}\n"
                "\n"
                "[[ ## completed ## ]]\n"
                "In adhering to this structure, your objective is: \n"
                "        Given the fields `question`, `tools`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\n"
                "Q?\n"
                "\n"
                "[[ ## tools ## ]]\n"
                "[\"search, whose description is <desc>Search for documents.</desc>. It takes arguments {'query': "
                "{'type': 'string'}, 'k': {'type': 'integer', 'default': 3}}.\"]\n"
                "\n"
                "Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then "
                "ending with the marker for `[[ ## completed ## ]]`.",
            },
        ],
    ),
)

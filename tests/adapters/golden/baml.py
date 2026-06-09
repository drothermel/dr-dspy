from dspy.adapters.baml_adapter import BAMLAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.baml_cases import nested_output_baml, simple_qa_with_demo_baml

BAML_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    GoldenPromptCase(
        id="baml/simple_qa_with_demo",
        adapter_builder=BAMLAdapter,
        scenario=simple_qa_with_demo_baml(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (str): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n# The answer.\nOutput field `answer` should be of type: string\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.",
            },
            {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
            {"role": "assistant", "content": '{\n  "answer": "A1"\n}'},
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ2\n\nRespond with a JSON object in the following order of fields: `answer`.",
            },
        ],
    ),
    GoldenPromptCase(
        id="baml/nested_output",
        adapter_builder=BAMLAdapter,
        scenario=nested_output_baml(),
        messages=[
            {
                "role": "system",
                "content": "Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answer` (BamlNested): The answer.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n# The answer.\nOutput field `answer` should be of type: {\n  value: int,\n  tags: string[],\n}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.",
            },
            {
                "role": "user",
                "content": "[[ ## question ## ]]\nQ\n\nRespond with a JSON object in the following order of fields: `answer` (must be formatted as a valid Python BamlNested).",
            },
        ],
    ),
)

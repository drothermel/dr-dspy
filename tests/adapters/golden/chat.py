from dspy.adapters.chat_adapter import ChatAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.qa import simple_qa_chat

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
)

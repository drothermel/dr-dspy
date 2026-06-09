from tests.adapters.golden.baml import BAML_GOLDEN_CASES
from tests.adapters.golden.chat import CHAT_GOLDEN_CASES
from tests.adapters.golden.chat_extended import CHAT_EXTENDED_GOLDEN_CASES
from tests.adapters.golden.json import JSON_GOLDEN_CASES
from tests.adapters.golden.two_step import TWO_STEP_GOLDEN_CASES
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.golden.xml import XML_GOLDEN_CASES

ALL_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    CHAT_GOLDEN_CASES
    + CHAT_EXTENDED_GOLDEN_CASES
    + JSON_GOLDEN_CASES
    + XML_GOLDEN_CASES
    + BAML_GOLDEN_CASES
    + TWO_STEP_GOLDEN_CASES
)

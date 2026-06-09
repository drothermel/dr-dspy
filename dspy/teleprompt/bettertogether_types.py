from enum import StrEnum


class BetterTogetherBuiltinKey(StrEnum):
    PROMPT = "p"
    WEIGHTS = "w"


DEFAULT_BETTER_TOGETHER_STRATEGY: list[str] = [
    BetterTogetherBuiltinKey.PROMPT,
    BetterTogetherBuiltinKey.WEIGHTS,
    BetterTogetherBuiltinKey.PROMPT,
]

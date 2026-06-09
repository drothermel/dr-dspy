"""DPR-style passage normalization helpers for token overlap metrics.

Import from ``dspy.evaluate.dpr`` when matching upstream DPR evaluation behavior.
"""

import unicodedata

import regex

__all__ = ["has_answer", "DPR_normalize"]

_ALPHA_NUM = "[\\p{L}\\p{N}\\p{M}]+"
_NON_WS = "[^\\p{Z}\\p{C}]"
_TOKEN_RE = regex.compile(
    f"({_ALPHA_NUM})|({_NON_WS})",
    flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
)


def _tokenize_words_uncased(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text)
    return [match.group().lower() for match in _TOKEN_RE.finditer(normalized)]


def has_answer(tokenized_answers: list[list[str]], text: str) -> bool:
    words = _tokenize_words_uncased(text)
    for single_answer in tokenized_answers:
        for i in range(len(words) - len(single_answer) + 1):
            if single_answer == words[i : i + len(single_answer)]:
                return True
    return False


def DPR_normalize(text: str) -> list[str]:
    return _tokenize_words_uncased(text)

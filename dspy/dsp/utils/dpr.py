import copy
import logging
import unicodedata

import regex
from typing_extensions import override

logger = logging.getLogger(__name__)


class Tokens:
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None) -> None:
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self) -> int:
        return len(self.data)

    def slice(self, i=None, j=None):
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        return [t[self.TEXT] for t in self.data]

    def offsets(self):
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for s, e in ngrams]
        return ngrams

    def entity_groups(self):
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            if ner_tag != non_ent:
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer:
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def __del__(self) -> None:
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = "[\\p{L}\\p{N}\\p{M}]+"
    NON_WS = "[^\\p{Z}\\p{C}]"

    def __init__(self, **kwargs) -> None:
        self._regexp = regex.compile(
            f"({self.ALPHA_NUM})|({self.NON_WS})", flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning("%s only tokenizes! Skipping annotators: %s", type(self).__name__, kwargs.get("annotators"))
        self.annotators = set()

    @override
    def tokenize(self, text):
        data = []
        matches = list(self._regexp.finditer(text))
        for i in range(len(matches)):
            token = matches[i].group()
            span = matches[i].span()
            start_ws = span[0]
            end_ws = matches[i + 1].span()[0] if i + 1 < len(matches) else span[1]
            data.append((token, text[start_ws:end_ws], span))
        return Tokens(data, self.annotators)


def has_answer(tokenized_answers, text) -> bool:
    text = DPR_normalize(text)
    for single_answer in tokenized_answers:
        for i in range(len(text) - len(single_answer) + 1):
            if single_answer == text[i : i + len(single_answer)]:
                return True
    return False


def locate_answers(tokenized_answers, text):
    tokenized_text = DPR_tokenize(text)
    occurrences = []
    text_words, text_word_positions = (tokenized_text.words(uncased=True), tokenized_text.offsets())
    answers_words = [ans.words(uncased=True) for ans in tokenized_answers]
    for single_answer in answers_words:
        for i in range(len(text_words) - len(single_answer) + 1):
            if single_answer == text_words[i : i + len(single_answer)]:
                (offset, _), (_, endpos) = (text_word_positions[i], text_word_positions[i + len(single_answer) - 1])
                occurrences.append((offset, endpos))
    return occurrences


STokenizer = SimpleTokenizer()


def DPR_tokenize(text):
    return STokenizer.tokenize(unicodedata.normalize("NFD", text))


def DPR_normalize(text):
    return DPR_tokenize(text).words(uncased=True)


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

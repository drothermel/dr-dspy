#!/usr/bin/env python3
from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

_PRESERVE_COMMENT = re.compile(r"(noqa|ty:ignore|type:\s*ignore)", re.IGNORECASE)


def _should_preserve_comment(comment_text: str) -> bool:
    return bool(_PRESERVE_COMMENT.search(comment_text))


def strip_comments_from_source(source: str) -> str:
    tokens: list[tokenize.TokenInfo] = []
    reader = io.StringIO(source).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.COMMENT and not _should_preserve_comment(token.string):
            continue
        tokens.append(token)
    return tokenize.untokenize(tokens)


def strip_file(path: Path) -> bool:
    source = path.read_text()
    updated = strip_comments_from_source(source)
    if updated != source:
        path.write_text(updated)
        return True
    return False


def main(argv: list[str]) -> int:
    roots = [Path(p) for p in argv[1:]] or [Path("dspy"), Path("tests")]
    changed = 0
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if strip_file(path):
                changed += 1
                print(path)
    print(f"updated {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

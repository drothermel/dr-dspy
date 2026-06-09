import hashlib
from pickle import dumps
from typing import Any


def hash_bytes(value: bytes | list[bytes]) -> str:
    chunks = [value] if isinstance(value, bytes) else value
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def hash_pickle(value: Any) -> str:
    return hash_bytes(dumps(value))

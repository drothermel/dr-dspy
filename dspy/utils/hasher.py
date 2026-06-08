import hashlib
from pickle import dumps
from typing import Any


class Hasher:
    dispatch: dict = {}

    def __init__(self) -> None:
        self.m = hashlib.sha256()

    @classmethod
    def hash_bytes(cls, value: bytes | list[bytes]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = hashlib.sha256()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()

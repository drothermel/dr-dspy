import asyncio
import importlib.util
from typing import Any

import pytest
from pydantic import BaseModel

requires_jsonschema = pytest.mark.skipif(
    importlib.util.find_spec("jsonschema") is None, reason="jsonschema is not installed"
)


def dummy_function(x: int, y: str = "hello") -> str:
    return f"{y} {x}"


class DummyModel(BaseModel):
    field1: str = "hello"
    field2: int


def dummy_with_pydantic(model: DummyModel) -> str:
    return f"{model.field1} {model.field2}"


class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    is_primary: bool = False


class ContactInfo(BaseModel):
    email: str
    phone: str | None = None
    addresses: list[Address]


class UserProfile(BaseModel):
    user_id: int
    name: str
    age: int | None = None
    contact: ContactInfo
    tags: list[str] = []


class Note(BaseModel):
    content: str
    author: str


def complex_dummy_function(profile: UserProfile, priority: int, notes: list[Note] | None = None) -> dict[str, Any]:
    primary_address = next(
        (addr for addr in profile.contact.addresses if addr.is_primary), profile.contact.addresses[0]
    )
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "priority": priority,
        "primary_address": primary_address.model_dump(),
        "notes": notes,
    }


async def async_dummy_function(x: int, y: str = "hello") -> str:
    await asyncio.sleep(0.1)
    return f"{y} {x}"


async def async_dummy_with_pydantic(model: DummyModel) -> str:
    await asyncio.sleep(0.1)
    return f"{model.field1} {model.field2}"


async def async_complex_dummy_function(
    profile: UserProfile, priority: int, notes: list[Note] | None = None
) -> dict[str, Any]:
    await asyncio.sleep(0.1)
    primary_address = next(
        (addr for addr in profile.contact.addresses if addr.is_primary), profile.contact.addresses[0]
    )
    await asyncio.sleep(0.1)
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "priority": priority,
        "primary_address": primary_address.model_dump(),
        "notes": notes,
    }

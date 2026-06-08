import asyncio
import importlib.util
from typing import Any

import pytest
from pydantic import BaseModel

requires_jsonschema = pytest.mark.skipif(
    importlib.util.find_spec("jsonschema") is None,
    reason="jsonschema is not installed",
)


# Test fixtures
def dummy_function(x: int, y: str = "hello") -> str:
    """A dummy function for testing.

    Args:
        x: An integer parameter
        y: A string parameter
    """
    return f"{y} {x}"


class DummyModel(BaseModel):
    field1: str = "hello"
    field2: int


def dummy_with_pydantic(model: DummyModel) -> str:
    """A dummy function that accepts a Pydantic model."""
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
    """Process user profile with complex nested structure.

    Args:
        profile: User profile containing nested contact and address information
        priority: Priority level of the processing
        notes: Optional processing notes
    """
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
    """An async dummy function for testing.

    Args:
        x: An integer parameter
        y: A string parameter
    """
    await asyncio.sleep(0.1)  # Simulate some async work
    return f"{y} {x}"


async def async_dummy_with_pydantic(model: DummyModel) -> str:
    """An async dummy function that accepts a Pydantic model."""
    await asyncio.sleep(0.1)  # Simulate some async work
    return f"{model.field1} {model.field2}"


async def async_complex_dummy_function(
    profile: UserProfile,
    priority: int,
    notes: list[Note] | None = None,
) -> dict[str, Any]:
    """Process user profile with complex nested structure asynchronously.

    Args:
        profile: User profile containing nested contact and address information
        priority: Priority level of the processing
        notes: Optional processing notes
    """
    # Simulate some async processing work
    await asyncio.sleep(0.1)

    primary_address = next(
        (addr for addr in profile.contact.addresses if addr.is_primary), profile.contact.addresses[0]
    )

    # Simulate more async work after finding primary address
    await asyncio.sleep(0.1)

    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "priority": priority,
        "primary_address": primary_address.model_dump(),
        "notes": notes,
    }

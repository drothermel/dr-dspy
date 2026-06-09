import pydantic


class Address(pydantic.BaseModel):
    city: str
    country: str


class Person(pydantic.BaseModel):
    name: str
    address: Address
    tags: list[str]


class Summary(pydantic.BaseModel):
    headline: str
    score: float


class Location(pydantic.BaseModel):
    city: str
    country: str


class Profile(pydantic.BaseModel):
    name: str
    location: Location
    interests: list[str]


class AnswerCard(pydantic.BaseModel):
    answer: str
    sources: list[str]


class JsonNestedAddress(pydantic.BaseModel):
    city: str
    country: str


class JsonNestedSummary(pydantic.BaseModel):
    title: str
    address: JsonNestedAddress
    scores: list[float]


class XmlAddress(pydantic.BaseModel):
    city: str
    country: str


class XmlSummary(pydantic.BaseModel):
    title: str
    address: XmlAddress

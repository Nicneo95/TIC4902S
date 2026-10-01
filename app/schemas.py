from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    # QUIZ DATA GOES HERE as an optional nested model, e.g. destination,
    # trip_days, budget, interests, and travel_pace. Keep it optional so
    # existing typed chat requests continue to work.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1_024, ge=1, le=32_768)


class ChatResponse(BaseModel):
    content: str
    model: str


# Speech-to-text response schema (disabled with the endpoint):
# class TranscriptionResponse(BaseModel):
#     text: str
#     model: str



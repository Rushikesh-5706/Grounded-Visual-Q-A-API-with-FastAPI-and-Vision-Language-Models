from __future__ import annotations

from pydantic import BaseModel


class VQAResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str

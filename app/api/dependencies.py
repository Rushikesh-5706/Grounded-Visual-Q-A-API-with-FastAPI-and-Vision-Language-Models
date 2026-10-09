from __future__ import annotations

from fastapi import Request

from app.core.config import Settings, get_settings
from app.services.vlm import VisionClient


def get_settings_dep() -> Settings:
    return get_settings()


def get_vision_client(request: Request) -> VisionClient:
    return request.app.state.vision_client

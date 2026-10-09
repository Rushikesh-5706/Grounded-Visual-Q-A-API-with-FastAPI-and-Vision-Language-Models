from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.core.audit import audit_image_payload
from app.core.config import Settings
from app.services.image import prepare_for_model, validate_image
from app.services.vlm import VisionClient, ask_vision_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    image_sha256: str


async def answer_question(
    request_id: str,
    image_bytes: bytes,
    declared_content_type: str | None,
    question: str,
    *,
    settings: Settings,
    client: VisionClient,
) -> AnswerResult:
    detected = await asyncio.to_thread(validate_image, image_bytes, declared_content_type)

    digest = await asyncio.to_thread(
        audit_image_payload, request_id, image_bytes, settings.audit_log_path
    )

    prepared = await asyncio.to_thread(
        prepare_for_model, image_bytes, detected, settings.max_image_dimension
    )

    t0 = time.monotonic()
    answer = await ask_vision_model(
        prepared.base64,
        prepared.mime_type,
        question,
        client=client,
        settings=settings,
    )
    latency_ms = round((time.monotonic() - t0) * 1000)

    logger.info(
        "request_id=%s sha256=%s original=%dx%d sent=%dx%d resized=%s "
        "provider=%s model=%s latency_ms=%d",
        request_id,
        digest,
        detected.width,
        detected.height,
        prepared.width,
        prepared.height,
        prepared.resized,
        settings.active_provider,
        settings.active_model,
        latency_ms,
    )

    return AnswerResult(answer=answer, image_sha256=digest)

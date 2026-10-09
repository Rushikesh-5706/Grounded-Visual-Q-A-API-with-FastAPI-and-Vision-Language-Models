from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from app.api.dependencies import get_settings_dep, get_vision_client
from app.core.config import Settings
from app.core.exceptions import InvalidRequestError, PayloadTooLargeError
from app.schemas import ErrorResponse, VQAResponse
from app.services.orchestrator import answer_question
from app.services.vlm import VisionClient

router = APIRouter()

_CHUNK = 1024 * 1024  # 1 MiB


@router.post(
    "/api/vqa",
    response_model=VQAResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        413: {"model": ErrorResponse, "description": "Payload too large"},
        422: {"model": ErrorResponse, "description": "Missing required field"},
        500: {"model": ErrorResponse, "description": "Audit failure"},
        502: {"model": ErrorResponse, "description": "Upstream error"},
        503: {"model": ErrorResponse, "description": "Provider not configured or rate limited"},
        504: {"model": ErrorResponse, "description": "Upstream timeout"},
    },
)
async def answer_visual_question(
    response: Response,
    file: UploadFile = File(...),
    question: str = Form(...),
    settings: Settings = Depends(get_settings_dep),
    vision_client: VisionClient = Depends(get_vision_client),
) -> VQAResponse:
    question_stripped = question.strip()
    if not question_stripped:
        raise InvalidRequestError("question must not be empty")
    if len(question_stripped) > settings.max_question_chars:
        raise InvalidRequestError(
            f"question exceeds {settings.max_question_chars} character limit"
        )

    if not file.filename:
        raise InvalidRequestError("file must have a filename")

    ct = file.content_type
    if ct and ct != "application/octet-stream" and not ct.startswith("image/"):
        raise InvalidRequestError(
            f"content type {ct!r} is not an image type"
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"upload exceeds {settings.max_upload_bytes} byte limit"
            )
        chunks.append(chunk)

    image_bytes = b"".join(chunks)
    if not image_bytes:
        raise InvalidRequestError("uploaded file is empty")

    request_id = str(uuid.uuid4())
    result = await answer_question(
        request_id,
        image_bytes,
        file.content_type,
        question_stripped,
        settings=settings,
        client=vision_client,
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Image-SHA256"] = result.image_sha256
    return VQAResponse(answer=result.answer)

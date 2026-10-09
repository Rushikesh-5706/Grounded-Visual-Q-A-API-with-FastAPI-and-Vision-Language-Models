from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import router
from app.core.config import get_settings
from app.core.exceptions import VQAError
from app.schemas import ErrorResponse, HealthResponse
from app.services.vlm import VisionClient


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)

    key_status = "configured" if settings.active_api_key else "not configured"
    logging.getLogger(__name__).info(
        "startup provider=%s model=%s key=%s",
        settings.active_provider,
        settings.active_model,
        key_status,
    )

    if settings.active_api_key and settings.vlm_provider != settings.active_provider:
        logging.getLogger(__name__).warning(
            "provider %r has no key; using %r instead",
            settings.vlm_provider,
            settings.active_provider,
        )

    client = VisionClient(settings)
    application.state.vision_client = client
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(
    title="Grounded VQA API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(VQAError)
async def vqa_error_handler(request: Request, exc: VQAError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc = " -> ".join(str(p) for p in first.get("loc", []))
        msg = first.get("msg", "validation error")
        detail = f"{loc}: {msg}" if loc else msg
    else:
        detail = "validation error"
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={200: {"model": HealthResponse}},
)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


app.include_router(router)

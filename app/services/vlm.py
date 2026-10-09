from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import TYPE_CHECKING

import httpx

from app.core.exceptions import (
    ProviderNotConfiguredError,
    UpstreamError,
    UpstreamRateLimitedError,
    UpstreamTimeoutError,
)

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a visually grounded AI. Answer the user's question accurately based ONLY on the "
    "provided image. Be concise. If the question asks for a count or quantity, reply with the "
    "numerical digit only (for example 3, not three). If the answer cannot be determined from "
    "the image, say so instead of guessing."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


class VisionClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.request_timeout_seconds,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(max_connections=20),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def _http(self) -> httpx.AsyncClient:
        return self._client


async def ask_vision_model(
    base64_image: str,
    mime_type: str,
    question: str,
    *,
    client: VisionClient | None = None,
    settings: "Settings | None" = None,
) -> str:
    from app.core.config import get_settings

    if settings is None:
        settings = get_settings()

    api_key = settings.active_api_key
    if not api_key:
        raise ProviderNotConfiguredError(
            f"no API key configured for provider '{settings.active_provider}'"
        )

    data_uri = f"data:{mime_type};base64,{base64_image}"
    payload = {
        "model": settings.active_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": settings.max_answer_tokens,
        "reasoning_effort": "none",
        "reasoning_format": "hidden",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.active_base_url}/chat/completions"

    http = client._http if client else httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=settings.request_timeout_seconds,
            write=30.0,
            pool=10.0,
        )
    )

    last_exc: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            response = await http.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < settings.max_retries - 1:
                await _backoff(attempt)
            continue
        except httpx.ConnectError as exc:
            last_exc = exc
            if attempt < settings.max_retries - 1:
                await _backoff(attempt)
            continue

        if response.status_code == 200:
            return _extract_content(response)

        if response.status_code not in _TRANSIENT_STATUSES:
            status = response.status_code
            snippet = response.text[:200]
            logger.warning(
                "upstream returned status %d, snippet: %s",
                status,
                snippet,
            )
            if status == 429:
                raise UpstreamRateLimitedError("upstream rate limited")
            raise UpstreamError(f"upstream returned status {status}")

        if response.status_code == 429:
            wait = _parse_retry_after(response, attempt)
            logger.warning("upstream rate limited (attempt %d), waiting %.1fs", attempt + 1, wait)
            last_exc = UpstreamRateLimitedError("upstream rate limited")
            if attempt < settings.max_retries - 1:
                await asyncio.sleep(wait)
            continue

        snippet = response.text[:200]
        logger.warning(
            "upstream transient error status %d on attempt %d, snippet: %s",
            response.status_code,
            attempt + 1,
            snippet,
        )
        last_exc = UpstreamError(f"upstream returned status {response.status_code}")
        if attempt < settings.max_retries - 1:
            await _backoff(attempt)

    if isinstance(last_exc, httpx.TimeoutException):
        raise UpstreamTimeoutError("request to upstream timed out")
    if isinstance(last_exc, UpstreamRateLimitedError):
        raise UpstreamRateLimitedError("upstream rate limited after retries")
    raise UpstreamError("upstream failed after retries")


def _extract_content(response: httpx.Response) -> str:
    try:
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise UpstreamError("upstream returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        content = _THINK_RE.sub("", content).strip()
        if not content:
            raise UpstreamError("upstream returned empty content")
        return content
    except (KeyError, IndexError, ValueError) as exc:
        raise UpstreamError(f"malformed upstream response: {exc}") from exc


def _backoff(attempt: int) -> "asyncio.coroutine":
    delay = min(0.5 * (2**attempt) + random.uniform(0, 0.3), 10.0)
    return asyncio.sleep(delay)


def _parse_retry_after(response: httpx.Response, attempt: int) -> float:
    raw = response.headers.get("retry-after", "")
    try:
        return min(float(raw), 10.0)
    except (ValueError, TypeError):
        pass
    delay = min(0.5 * (2**attempt) + random.uniform(0, 0.3), 10.0)
    return delay

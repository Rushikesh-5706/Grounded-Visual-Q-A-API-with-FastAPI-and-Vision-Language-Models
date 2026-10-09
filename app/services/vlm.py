from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx

from app.core.exceptions import (
    ProviderNotConfiguredError,
    UpstreamError,
    UpstreamRateLimitedError,
    UpstreamTimeoutError,
    VQAError,
)

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a visually grounded AI. Answer the user's question accurately based ONLY on the "
    "provided image. Be concise. If the question asks for a count or quantity, reply with the "
    "numerical digit only (for example 3, not three). If the answer cannot be determined from "
    "the image, say so instead of guessing."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_DELAY_SECONDS = 10.0


def _build_timeout(settings: Settings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=10.0,
        read=settings.request_timeout_seconds,
        write=30.0,
        pool=10.0,
    )


class VisionClient:
    """Owns the shared HTTP connection pool used for every provider call."""

    def __init__(self, settings: Settings) -> None:
        self._http = httpx.AsyncClient(
            timeout=_build_timeout(settings),
            limits=httpx.Limits(max_connections=20),
        )

    async def post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> httpx.Response:
        return await self._http.post(url, headers=headers, json=payload)

    async def aclose(self) -> None:
        await self._http.aclose()


def build_payload(
    settings: Settings, base64_image: str, mime_type: str, question: str
) -> dict[str, Any]:
    data_uri = f"data:{mime_type};base64,{base64_image}"
    payload: dict[str, Any] = {
        "model": settings.active_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
    }
    if settings.active_provider == "groq":
        # These parameters are Groq-specific and not part of the OpenAI spec.
        payload["reasoning_effort"] = "none"
        payload["reasoning_format"] = "hidden"
    return payload


async def ask_vision_model(
    base64_image: str,
    mime_type: str,
    question: str,
    *,
    client: VisionClient | None = None,
    settings: Settings | None = None,
) -> str:
    if settings is None:
        from app.core.config import get_settings

        settings = get_settings()

    api_key = settings.active_api_key
    if not api_key:
        raise ProviderNotConfiguredError(
            f"no API key configured for provider '{settings.active_provider}'"
        )

    url = f"{settings.active_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(settings, base64_image, mime_type, question)

    if client is not None:
        response = await _post_with_retries(client.post, url, headers, payload, settings)
    else:
        async with httpx.AsyncClient(timeout=_build_timeout(settings)) as http:

            async def _post(
                u: str, h: dict[str, str], p: dict[str, Any]
            ) -> httpx.Response:
                return await http.post(u, headers=h, json=p)

            response = await _post_with_retries(_post, url, headers, payload, settings)

    return _extract_answer(response)


async def _post_with_retries(
    post: Callable[[str, dict[str, str], dict[str, Any]], Awaitable[httpx.Response]],
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    settings: Settings,
) -> httpx.Response:
    last_error: VQAError = UpstreamError("upstream failed after retries")
    for attempt in range(1, settings.max_retries + 1):
        delay = _backoff_delay(attempt)
        try:
            response = await post(url, headers, payload)
        except httpx.TimeoutException:
            last_error = UpstreamTimeoutError("request to upstream timed out")
        except httpx.TransportError:
            last_error = UpstreamError("could not reach upstream")
        else:
            if response.status_code == 200:
                return response
            if response.status_code not in _TRANSIENT_STATUSES:
                logger.warning(
                    "upstream status=%d body=%s",
                    response.status_code,
                    response.text[:200],
                )
                raise UpstreamError(
                    f"upstream returned status {response.status_code}"
                )
            logger.warning(
                "upstream transient status=%d attempt=%d body=%s",
                response.status_code,
                attempt,
                response.text[:200],
            )
            if response.status_code == 429:
                last_error = UpstreamRateLimitedError(
                    "upstream rate limited after retries"
                )
                delay = _retry_after(response, delay)
            else:
                last_error = UpstreamError(
                    f"upstream returned status {response.status_code}"
                )
        if attempt < settings.max_retries:
            await asyncio.sleep(delay)
    raise last_error


def _extract_answer(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError as exc:
        raise UpstreamError("upstream returned invalid JSON") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise UpstreamError("upstream response has no message content") from exc

    if isinstance(content, list):
        content = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not isinstance(content, str):
        raise UpstreamError("upstream message content is not text")

    answer = _THINK_BLOCK.sub("", content).strip()
    if not answer:
        raise UpstreamError("upstream returned an empty answer")
    return answer


def _backoff_delay(attempt: int) -> float:
    return min(0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.3), _MAX_DELAY_SECONDS)


def _retry_after(response: httpx.Response, fallback: float) -> float:
    try:
        return min(float(response.headers.get("retry-after", "")), _MAX_DELAY_SECONDS)
    except ValueError:
        return fallback

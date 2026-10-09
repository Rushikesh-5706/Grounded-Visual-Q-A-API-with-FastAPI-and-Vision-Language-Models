import base64
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.api.dependencies import get_settings_dep
from app.core.config import Settings
from app.main import app


def _override_settings(tmp_path: Path) -> Settings:
    return Settings(
        groq_api_key="test-key-groq",
        openai_api_key="test-key-openai",
        audit_log_path=tmp_path / "audit.log",
        max_retries=3,
    )


@pytest.fixture()
def client(tmp_path: Path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def test_image(tmp_path):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 50, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@respx.mock
def test_payload_integrity(client, test_image, tmp_path):
    import hashlib

    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "3"}}]})
    )

    r = client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 200
    assert route.called

    request = route.calls.last.request
    body = json.loads(request.content)

    # Assert question
    assert body["messages"][1]["content"][0]["text"] == "What?"
    assert body["messages"][0]["role"] == "system"
    assert "You are a visually grounded AI" in body["messages"][0]["content"]

    # Assert image bytes and hash
    data_uri = body["messages"][1]["content"][1]["image_url"]["url"]
    assert data_uri.startswith("data:image/jpeg;base64,")
    b64 = data_uri.split(",", 1)[1]
    decoded = base64.b64decode(b64)

    expected_digest = hashlib.sha256(test_image).hexdigest()
    assert hashlib.sha256(decoded).hexdigest() == expected_digest
    assert r.headers["X-Image-SHA256"] == expected_digest

    audit_log = (tmp_path / "audit.log").read_text()
    assert expected_digest in audit_log

    # Verify API key is NOT in body
    assert "test-key-groq" not in request.content.decode()


@respx.mock
def test_upstream_401_returns_502(client, test_image):
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    r = client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 502
    assert route.call_count == 1


@respx.mock
def test_upstream_429_retries(client, test_image):
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0.1"}),
            httpx.Response(429, headers={"retry-after": "0.1"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    r = client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 200
    assert route.call_count == 3


@respx.mock
def test_upstream_timeout_returns_504(client, test_image):
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    r = client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 504
    assert route.call_count == 3


@respx.mock
def test_upstream_malformed_returns_502(client, test_image):
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        ]
    )
    r = client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 502
    assert route.call_count == 1


def test_large_image_rejected(client):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (7071, 7072), color=(200, 50, 50)).save(buf, format="JPEG")
    r = client.post(
        "/api/vqa",
        files={"file": ("large.jpg", buf.getvalue(), "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 400


@respx.mock
def test_reasoning_effort_in_groq_not_in_openai(client, test_image, tmp_path):
    settings = Settings(
        groq_api_key="test-key-groq",
        audit_log_path=tmp_path / "audit.log",
        vlm_provider="groq",
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings

    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    body = json.loads(route.calls.last.request.content)
    assert "reasoning_effort" in body
    assert "reasoning_format" in body

    # Switch to OpenAI
    settings = Settings(
        vlm_provider="openai",
        openai_api_key="test-key-openai",
        audit_log_path=tmp_path / "audit.log",
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings
    route2 = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    body2 = json.loads(route2.calls.last.request.content)
    assert "reasoning_effort" not in body2
    assert "reasoning_format" not in body2


def test_unexpected_exception_handler(test_image):
    with (
        patch("app.services.orchestrator.validate_image", side_effect=RuntimeError("unexpected")),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        r = c.post(
            "/api/vqa",
            files={"file": ("test.jpg", test_image, "image/jpeg")},
            data={"question": "What?"},
        )
        assert r.status_code == 500
        assert r.json() == {"detail": "internal server error"}


@respx.mock
def test_upstream_500_retries(client, test_image):
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=httpx.Response(500, json={"error": "internal"})
    )
    r = client.post(
        "/api/vqa",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"question": "What?"},
    )
    assert r.status_code == 502
    assert route.call_count == 3

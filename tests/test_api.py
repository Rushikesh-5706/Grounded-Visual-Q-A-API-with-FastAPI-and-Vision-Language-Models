"""Unit and integration tests for the VQA API.

Tests prefixed with 'test_mock_' use a mocked HTTP transport; the name
makes the mocking explicit per the project spec.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings, get_settings
from app.main import app


def _make_jpeg(width: int = 100, height: int = 100, color: tuple = (200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _make_webp(width: int = 100, height: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(80, 180, 80)).save(buf, format="WEBP")
    return buf.getvalue()


def _override_settings(tmp_path: Path) -> Settings:
    return Settings(
        groq_api_key="test-key-groq",
        audit_log_path=tmp_path / "audit.log",
    )


from app.api.dependencies import get_settings_dep

@pytest.fixture()
def client(tmp_path: Path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def mock_vlm_answer():
    """Patch ask_vision_model to return a fixed answer without any network call."""
    with patch("app.services.orchestrator.ask_vision_model", new_callable=AsyncMock) as m:
        m.return_value = "mock answer"
        yield m


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_empty_question_returns_400(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
        data={"question": ""},
    )
    assert r.status_code in (400, 422)
    assert "question" in r.json()["detail"].lower()
    mock_vlm_answer.assert_not_called()


def test_whitespace_question_returns_400(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
        data={"question": "   "},
    )
    assert r.status_code in (400, 422)
    mock_vlm_answer.assert_not_called()


def test_missing_question_returns_422(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
    )
    assert r.status_code == 422
    assert "detail" in r.json()
    mock_vlm_answer.assert_not_called()


def test_missing_file_returns_422(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        data={"question": "What is this?"},
    )
    assert r.status_code == 422
    assert "detail" in r.json()
    mock_vlm_answer.assert_not_called()


def test_zero_byte_file_returns_400(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", b"", "image/jpeg")},
        data={"question": "What is this?"},
    )
    assert r.status_code == 400
    mock_vlm_answer.assert_not_called()


def test_txt_file_with_text_plain_returns_400(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"question": "What is this?"},
    )
    assert r.status_code == 400
    mock_vlm_answer.assert_not_called()


def test_text_bytes_declared_as_jpeg_returns_400(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", b"not an image at all", "image/jpeg")},
        data={"question": "What color?"},
    )
    assert r.status_code == 400
    mock_vlm_answer.assert_not_called()


def test_truncated_jpeg_returns_400(client, mock_vlm_answer):
    data = _make_jpeg()[:50]
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", data, "image/jpeg")},
        data={"question": "What color?"},
    )
    assert r.status_code == 400
    mock_vlm_answer.assert_not_called()


def test_gif_returns_400(client, mock_vlm_answer):
    buf = io.BytesIO()
    Image.new("P", (10, 10)).save(buf, format="GIF")
    r = client.post(
        "/api/vqa",
        files={"file": ("img.gif", buf.getvalue(), "image/gif")},
        data={"question": "What is this?"},
    )
    assert r.status_code == 400
    mock_vlm_answer.assert_not_called()


def test_upload_above_limit_returns_413(tmp_path, mock_vlm_answer):
    tiny_limit = Settings(
        groq_api_key="k",
        audit_log_path=tmp_path / "audit.log",
        max_upload_bytes=100,
    )
    app.dependency_overrides[get_settings_dep] = lambda: tiny_limit
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "What?"},
            )
        assert r.status_code == 413
    finally:
        app.dependency_overrides.clear()
    mock_vlm_answer.assert_not_called()


def test_question_above_max_chars_returns_400(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
        data={"question": "x" * 2001},
    )
    assert r.status_code == 400
    mock_vlm_answer.assert_not_called()


def test_octet_stream_content_type_accepted(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", _make_jpeg(), "application/octet-stream")},
        data={"question": "What color?"},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_mock_jpeg_returns_200_with_answer_key(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
        data={"question": "What is in the image?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"answer"}
    assert isinstance(body["answer"], str)


def test_mock_png_returns_200_with_answer_key(client, mock_vlm_answer):
    r = client.post(
        "/api/vqa",
        files={"file": ("img.png", _make_png(), "image/png")},
        data={"question": "Describe this image."},
    )
    assert r.status_code == 200
    assert set(r.json().keys()) == {"answer"}


def test_mock_response_headers_contain_request_id_and_sha256(client, mock_vlm_answer):
    img = _make_jpeg()
    expected_digest = hashlib.sha256(img).hexdigest()
    r = client.post(
        "/api/vqa",
        files={"file": ("img.jpg", img, "image/jpeg")},
        data={"question": "What is in the image?"},
    )
    assert r.status_code == 200
    request_id = r.headers.get("x-request-id", "")
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        request_id,
    ), f"x-request-id not a UUID: {request_id!r}"
    assert r.headers.get("x-image-sha256") == expected_digest


# ---------------------------------------------------------------------------
# Audit integrity
# ---------------------------------------------------------------------------


def test_mock_audit_log_contains_matching_line(tmp_path, mock_vlm_answer):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    img = _make_jpeg()
    expected_digest = hashlib.sha256(img).hexdigest()
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", img, "image/jpeg")},
                data={"question": "What color?"},
            )
        assert r.status_code == 200
        request_id = r.headers["x-request-id"]
        lines = (tmp_path / "audit.log").read_text().splitlines()
        assert len(lines) == 1
        assert re.match(r"^\S+ \| [0-9a-f-]{36} \| [0-9a-f]{64}$", lines[0])
        assert expected_digest in lines[0]
        assert request_id in lines[0]
    finally:
        app.dependency_overrides.clear()


def test_mock_audit_log_appends_two_lines_unchanged(tmp_path, mock_vlm_answer):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    img1 = _make_jpeg(color=(200, 50, 50))
    img2 = _make_jpeg(color=(50, 200, 50))
    try:
        with TestClient(app) as c:
            r1 = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", img1, "image/jpeg")},
                data={"question": "q"},
            )
            r2 = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", img2, "image/jpeg")},
                data={"question": "q"},
            )
        assert r1.status_code == 200
        assert r2.status_code == 200
        lines = (tmp_path / "audit.log").read_text().splitlines()
        assert len(lines) == 2
        assert hashlib.sha256(img1).hexdigest() in lines[0]
        assert hashlib.sha256(img2).hexdigest() in lines[1]
    finally:
        app.dependency_overrides.clear()


def test_mock_20_concurrent_requests_produce_20_intact_lines(tmp_path, mock_vlm_answer):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    img = _make_jpeg()
    results = []
    try:
        with TestClient(app) as c:

            def send():
                r = c.post(
                    "/api/vqa",
                    files={"file": ("img.jpg", img, "image/jpeg")},
                    data={"question": "q"},
                )
                results.append(r.status_code)

            threads = [threading.Thread(target=send) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert all(s == 200 for s in results), f"not all 200: {results}"
        lines = (tmp_path / "audit.log").read_text().splitlines()
        assert len(lines) == 20
        pattern = re.compile(r"^\S+ \| [0-9a-f-]{36} \| [0-9a-f]{64}$")
        for line in lines:
            assert pattern.match(line), f"malformed line: {line!r}"
    finally:
        app.dependency_overrides.clear()


def test_mock_rejected_request_writes_no_audit_line(tmp_path, mock_vlm_answer):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": ""},
            )
        assert r.status_code in (400, 422)
        log_path = tmp_path / "audit.log"
        assert not log_path.exists() or log_path.read_text() == ""
    finally:
        app.dependency_overrides.clear()


def test_mock_unwritable_audit_path_returns_500(tmp_path, mock_vlm_answer):
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    import os

    os.chmod(readonly_dir, 0o555)
    settings = Settings(
        groq_api_key="k",
        audit_log_path=readonly_dir / "audit.log",
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "What?"},
            )
        assert r.status_code == 500
        mock_vlm_answer.assert_not_called()
    finally:
        app.dependency_overrides.clear()
        os.chmod(readonly_dir, 0o755)


# ---------------------------------------------------------------------------
# Payload integrity (core grounding guard)
# ---------------------------------------------------------------------------


def test_mock_payload_image_url_sha256_matches_audit_hash(tmp_path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    captured_payload = {}

    async def fake_ask(base64_image, mime_type, question, *, client=None, settings=None):
        captured_payload["base64"] = base64_image
        captured_payload["mime_type"] = mime_type
        captured_payload["question"] = question
        return "test answer"

    with patch("app.services.orchestrator.ask_vision_model", fake_ask), TestClient(app) as c:
        img = _make_jpeg()
        r = c.post(
            "/api/vqa",
            files={"file": ("img.jpg", img, "image/jpeg")},
            data={"question": "Test question"},
        )
    assert r.status_code == 200
    audit_digest = r.headers["x-image-sha256"]
    decoded = base64.b64decode(captured_payload["base64"])
    assert hashlib.sha256(decoded).hexdigest() == audit_digest
    app.dependency_overrides.clear()


def test_mock_question_text_in_payload_matches_submitted(tmp_path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    captured = {}

    async def fake_ask(base64_image, mime_type, question, *, client=None, settings=None):
        captured["question"] = question
        return "answer"

    with patch("app.services.orchestrator.ask_vision_model", fake_ask), TestClient(app) as c:
        r = c.post(
            "/api/vqa",
            files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
            data={"question": "How many cats?"},
        )
    assert r.status_code == 200
    assert captured["question"] == "How many cats?"
    app.dependency_overrides.clear()


def test_mock_system_message_contains_grounding_prompt(tmp_path):
    from app.services.vlm import _SYSTEM_PROMPT

    assert "visually grounded" in _SYSTEM_PROMPT.lower()
    assert "ONLY" in _SYSTEM_PROMPT


def test_mock_two_images_produce_different_data_uris(tmp_path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    uris = []

    async def fake_ask(base64_image, mime_type, question, *, client=None, settings=None):
        uris.append(base64_image)
        return "answer"

    with patch("app.services.orchestrator.ask_vision_model", fake_ask), TestClient(app) as c:
        c.post(
            "/api/vqa",
            files={"file": ("img.jpg", _make_jpeg(color=(200, 50, 50)), "image/jpeg")},
            data={"question": "q"},
        )
        c.post(
            "/api/vqa",
            files={"file": ("img.jpg", _make_jpeg(color=(50, 200, 50)), "image/jpeg")},
            data={"question": "q"},
        )
    assert len(uris) == 2
    assert uris[0] != uris[1]
    app.dependency_overrides.clear()


def test_mock_api_key_not_in_body_or_logs(tmp_path, caplog):
    import logging

    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    with patch("app.services.orchestrator.ask_vision_model", new_callable=AsyncMock) as m:
        m.return_value = "answer"
        with caplog.at_level(logging.DEBUG), TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 200
    for record in caplog.records:
        assert "test-key-groq" not in record.getMessage()
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_large_image_downscaled_to_1024_width(tmp_path, mock_vlm_answer):
    from app.services.image import prepare_for_model, validate_image

    big = io.BytesIO()
    Image.new("RGB", (3000, 2000), color=(100, 100, 200)).save(big, format="JPEG")
    data = big.getvalue()

    detected = validate_image(data, "image/jpeg")
    prepared = prepare_for_model(data, detected, 1024)

    assert prepared.width == 1024
    assert prepared.height == 683
    assert prepared.resized is True


def test_image_within_limit_is_byte_identical(tmp_path, mock_vlm_answer):
    from app.services.image import prepare_for_model, validate_image

    data = _make_jpeg(200, 150)
    detected = validate_image(data, "image/jpeg")
    prepared = prepare_for_model(data, detected, 1024)

    assert prepared.data == data
    assert prepared.resized is False


def test_exif_portrait_jpeg_rotated_before_measuring(tmp_path):
    from app.services.image import prepare_for_model, validate_image

    buf = io.BytesIO()
    img = Image.new("RGB", (100, 300), color=(150, 100, 50))
    exif = img.getexif()
    exif[274] = 6  # orientation = 90 degrees CW
    img.save(buf, format="JPEG", exif=exif.tobytes())
    data = buf.getvalue()

    detected = validate_image(data, "image/jpeg")
    prepared = prepare_for_model(data, detected, 1024)
    assert prepared.resized is False


def test_base64_has_no_newline_and_decodes_correctly(tmp_path, mock_vlm_answer):
    from app.services.image import prepare_for_model, validate_image

    data = _make_jpeg()
    detected = validate_image(data, "image/jpeg")
    prepared = prepare_for_model(data, detected, 1024)

    assert "\n" not in prepared.base64
    assert base64.b64decode(prepared.base64) == prepared.data


# ---------------------------------------------------------------------------
# Upstream behavior (all mocked)
# ---------------------------------------------------------------------------


def test_mock_upstream_401_returns_502(tmp_path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    with respx.mock:
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 502
    assert "key" not in r.json()["detail"].lower()
    app.dependency_overrides.clear()


def test_mock_upstream_429_twice_then_200_returns_200(tmp_path):
    settings = Settings(
        groq_api_key="test-key",
        audit_log_path=tmp_path / "audit.log",
        max_retries=3,
        request_timeout_seconds=5,
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(429, headers={"retry-after": "0.1"}, json={"error": "ratelimit"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    with respx.mock:
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(side_effect=side_effect)
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 200
    assert r.json()["answer"] == "42"
    app.dependency_overrides.clear()


def test_mock_upstream_500_three_times_returns_502(tmp_path):
    settings = Settings(
        groq_api_key="test-key",
        audit_log_path=tmp_path / "audit.log",
        max_retries=3,
        request_timeout_seconds=5,
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings

    with respx.mock:
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "internal"})
        )
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 502
    app.dependency_overrides.clear()


def test_mock_read_timeout_returns_504(tmp_path):
    settings = Settings(
        groq_api_key="test-key",
        audit_log_path=tmp_path / "audit.log",
        max_retries=1,
        request_timeout_seconds=1,
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings

    with respx.mock:
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 504
    app.dependency_overrides.clear()


def test_mock_200_with_empty_content_returns_502(tmp_path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    with respx.mock:
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": ""}}]}
            )
        )
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 502
    app.dependency_overrides.clear()


def test_mock_think_block_stripped_from_content(tmp_path):
    settings = _override_settings(tmp_path)
    app.dependency_overrides[get_settings_dep] = lambda: settings

    with respx.mock:
        respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "<think>lots of thinking</think>3"}}]},
            )
        )
        with TestClient(app) as c:
            r = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
    assert r.status_code == 200
    assert r.json()["answer"] == "3"
    app.dependency_overrides.clear()


def test_no_key_configured_returns_503_vqa_health_200(tmp_path):
    settings = Settings(
        groq_api_key=None,
        openai_api_key=None,
        audit_log_path=tmp_path / "audit.log",
    )
    app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        with TestClient(app) as c:
            health = c.get("/health")
            assert health.status_code == 200
            vqa = c.post(
                "/api/vqa",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"question": "q"},
            )
            assert vqa.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_mock_provider_fallback_groq_selected_no_key_uses_openai(tmp_path):
    settings = Settings(
        vlm_provider="groq",
        groq_api_key=None,
        openai_api_key="sk-openai-test-key",
        audit_log_path=tmp_path / "audit.log",
    )
    assert settings.active_provider == "openai"
    assert settings.active_base_url == "https://api.openai.com/v1"
    assert settings.active_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Other
# ---------------------------------------------------------------------------


def test_health_returns_200_and_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_fixtures_json_parses_images_exist_expectations_differ():
    project_root = Path(__file__).resolve().parents[1]
    fixtures_path = project_root / "fixtures" / "fixtures.json"
    assert fixtures_path.exists()
    data = json.loads(fixtures_path.read_text())

    for key in ("question", "image_a_path", "expected_a_contains", "image_b_path", "expected_b_contains"):
        assert key in data

    img_a = project_root / data["image_a_path"]
    img_b = project_root / data["image_b_path"]
    assert img_a.exists(), f"{img_a} not found"
    assert img_b.exists(), f"{img_b} not found"

    from app.services.image import validate_image

    validate_image(img_a.read_bytes(), None)
    validate_image(img_b.read_bytes(), None)

    assert data["expected_a_contains"] != data["expected_b_contains"]


def test_env_example_parses_and_contains_keys_no_real_secrets():
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env.example"
    assert env_path.exists()
    text = env_path.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "=" in stripped, f"line not KEY=VALUE: {stripped!r}"

    assert "GROQ_API_KEY" in text
    assert "OPENAI_API_KEY" in text

    assert not re.search(r"gsk_[A-Za-z0-9]{10,}", text)
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)


def test_submission_json_matches_settings_defaults():
    project_root = Path(__file__).resolve().parents[1]
    sub_path = project_root / "submission.json"
    assert sub_path.exists()
    data = json.loads(sub_path.read_text())

    assert set(data.keys()) == {"vlm_provider", "model_name"}
    assert isinstance(data["vlm_provider"], str)
    assert isinstance(data["model_name"], str)

    settings = Settings()
    assert data["vlm_provider"] == settings.vlm_provider
    assert data["model_name"] == settings.active_model


def test_env_example_variables_match_settings_fields():
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env.example"
    text = env_path.read_text()

    documented_keys = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split("=", 1)[0].lower()
        documented_keys.add(key)

    settings_fields = {f.lower() for f in Settings.model_fields}
    for key in documented_keys:
        assert key in settings_fields, f"{key!r} in .env.example has no matching Settings field"

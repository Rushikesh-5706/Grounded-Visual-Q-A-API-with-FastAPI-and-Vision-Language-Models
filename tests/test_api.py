"""Unit and integration tests for the VQA API.

Tests prefixed with 'test_mock_' use a mocked HTTP transport; the name
makes the mocking explicit per the project spec.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_settings_dep
from app.core.config import Settings
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


# Validation tests


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


# Success path


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


# Audit integrity


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
    import os

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("cannot test unwritable paths as root")

    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()

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

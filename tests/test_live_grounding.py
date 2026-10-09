"""Live grounding tests that call the real Groq API.

These tests are marked 'live' and are skipped unless GROQ_API_KEY is present.
They use the in-process FastAPI app with the real VLM client.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES_PATH = _PROJECT_ROOT / "fixtures" / "fixtures.json"


def _groq_key_available() -> bool:
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GROQ_API_KEY=") and len(line.split("=", 1)[1].strip()) > 10:
                return True
    import os

    return bool(os.environ.get("GROQ_API_KEY", "").strip())


pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_client():
    if not _groq_key_available():
        pytest.skip("GROQ_API_KEY not available")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def fixtures():
    return json.loads(_FIXTURES_PATH.read_text())


def test_live_fixture_a_answer_contains_expected(live_client, fixtures):
    img_path = _PROJECT_ROOT / fixtures["image_a_path"]
    r = live_client.post(
        "/api/vqa",
        files={"file": ("image_a.jpg", img_path.read_bytes(), "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    assert r.status_code == 200, f"status={r.status_code} body={r.text}"
    answer = r.json()["answer"]
    assert fixtures["expected_a_contains"].lower() in answer.lower(), (
        f"expected '{fixtures['expected_a_contains']}' in answer, got: {answer!r}"
    )


def test_live_fixture_b_answer_contains_expected(live_client, fixtures):
    img_path = _PROJECT_ROOT / fixtures["image_b_path"]
    r = live_client.post(
        "/api/vqa",
        files={"file": ("image_b.jpg", img_path.read_bytes(), "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    assert r.status_code == 200, f"status={r.status_code} body={r.text}"
    answer = r.json()["answer"]
    assert fixtures["expected_b_contains"].lower() in answer.lower(), (
        f"expected '{fixtures['expected_b_contains']}' in answer, got: {answer!r}"
    )


def test_live_fixture_answers_differ(live_client, fixtures):
    img_a = (_PROJECT_ROOT / fixtures["image_a_path"]).read_bytes()
    img_b = (_PROJECT_ROOT / fixtures["image_b_path"]).read_bytes()

    r_a = live_client.post(
        "/api/vqa",
        files={"file": ("image_a.jpg", img_a, "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    r_b = live_client.post(
        "/api/vqa",
        files={"file": ("image_b.jpg", img_b, "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    assert r_a.status_code == 200
    assert r_b.status_code == 200
    assert r_a.json()["answer"] != r_b.json()["answer"], (
        f"answers must differ but both returned: {r_a.json()['answer']!r}"
    )


def test_live_white_image_does_not_contain_3_or_5(live_client, fixtures):
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), color=(255, 255, 255)).save(buf, format="JPEG")
    r = live_client.post(
        "/api/vqa",
        files={"file": ("white.jpg", buf.getvalue(), "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    assert r.status_code == 200
    answer = r.json()["answer"].lower()
    assert "3" not in answer and "5" not in answer, (
        f"language-prior control failed: white image returned {answer!r}"
    )


def test_live_order_independent_no_caching(live_client, fixtures):
    img_a = (_PROJECT_ROOT / fixtures["image_a_path"]).read_bytes()
    img_b = (_PROJECT_ROOT / fixtures["image_b_path"]).read_bytes()

    r_b_first = live_client.post(
        "/api/vqa",
        files={"file": ("image_b.jpg", img_b, "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    r_a_second = live_client.post(
        "/api/vqa",
        files={"file": ("image_a.jpg", img_a, "image/jpeg")},
        data={"question": fixtures["question"]},
    )
    assert r_b_first.status_code == 200
    assert r_a_second.status_code == 200
    assert fixtures["expected_b_contains"].lower() in r_b_first.json()["answer"].lower()
    assert fixtures["expected_a_contains"].lower() in r_a_second.json()["answer"].lower()

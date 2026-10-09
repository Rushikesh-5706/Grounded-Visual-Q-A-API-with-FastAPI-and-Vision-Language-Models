"""Verify all ten core requirements against a running server.

Usage:
    python scripts/verify_requirements.py [--base-url URL] [--workspace PATH]
        [--audit-source file|container] [--check-container]

Exit 0 only when all ten checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

_REQ_IDS = [
    "req-1-fixtures",
    "req-2-validation-empty-question",
    "req-3-validation-missing-image",
    "req-4-vqa-endpoint-success",
    "req-5-grounding-proof",
    "req-6-audit-log",
    "req-7-health-check",
    "req-8-docker-compose",
    "req-9-env-example",
    "req-10-submission-config",
]


def verify_req_1_fixtures(workspace: str) -> None:
    root = Path(workspace)
    fixtures_path = root / "fixtures" / "fixtures.json"
    if not fixtures_path.exists():
        raise AssertionError("req-1-fixtures: fixtures/fixtures.json does not exist")

    try:
        data = json.loads(fixtures_path.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(f"req-1-fixtures: fixtures.json is not valid JSON: {exc}") from exc

    for key in (
        "question",
        "image_a_path",
        "expected_a_contains",
        "image_b_path",
        "expected_b_contains",
    ):
        if key not in data:
            raise AssertionError(f"req-1-fixtures: missing key {key!r} in fixtures.json")

    img_a = root / data["image_a_path"]
    img_b = root / data["image_b_path"]
    if not img_a.exists():
        raise AssertionError(f"req-1-fixtures: {img_a} not found")
    if not img_b.exists():
        raise AssertionError(f"req-1-fixtures: {img_b} not found")

    if data["expected_a_contains"] == data["expected_b_contains"]:
        raise AssertionError("req-1-fixtures: expected_a_contains equals expected_b_contains")


def verify_req_2_validation_empty_question(
    workspace: str, base_url: str = "http://localhost:8000"
) -> None:
    root = Path(workspace)
    fixtures = json.loads((root / "fixtures" / "fixtures.json").read_text())
    img = (root / fixtures["image_a_path"]).read_bytes()

    r = httpx.post(
        f"{base_url}/api/vqa",
        files={"file": ("image.jpg", img, "image/jpeg")},
        data={"question": ""},
        timeout=30,
    )
    if r.status_code not in (400, 422):
        raise AssertionError(
            f"req-2-validation-empty-question: expected 400 or 422, got {r.status_code}"
        )

    r2 = httpx.post(
        f"{base_url}/api/vqa",
        files={"file": ("image.jpg", img, "image/jpeg")},
        timeout=30,
    )
    if r2.status_code not in (400, 422):
        raise AssertionError(
            "req-2-validation-empty-question: omitted question expected 400/422, "
            f"got {r2.status_code}"
        )


def verify_req_3_validation_missing_image(
    workspace: str, base_url: str = "http://localhost:8000"
) -> None:
    r = httpx.post(
        f"{base_url}/api/vqa",
        data={"question": "What is this?"},
        timeout=30,
    )
    if r.status_code not in (400, 422):
        raise AssertionError(
            f"req-3-validation-missing-image: expected 400 or 422, got {r.status_code}"
        )


def verify_req_4_vqa_endpoint_success(
    workspace: str, base_url: str = "http://localhost:8000"
) -> None:
    root = Path(workspace)
    fixtures = json.loads((root / "fixtures" / "fixtures.json").read_text())
    img = (root / fixtures["image_a_path"]).read_bytes()

    r = httpx.post(
        f"{base_url}/api/vqa",
        files={"file": ("image.jpg", img, "image/jpeg")},
        data={"question": fixtures["question"]},
        timeout=120,
    )
    if r.status_code != 200:
        raise AssertionError(
            f"req-4-vqa-endpoint-success: expected 200, got {r.status_code}: {r.text[:200]}"
        )

    body = r.json()
    if "answer" not in body:
        raise AssertionError(f"req-4-vqa-endpoint-success: response missing 'answer' key: {body}")
    if not isinstance(body["answer"], str):
        raise AssertionError(
            f"req-4-vqa-endpoint-success: answer is not a string: {body['answer']!r}"
        )


def verify_req_5_grounding_proof(workspace: str, base_url: str = "http://localhost:8000") -> None:
    root = Path(workspace)
    fixtures = json.loads((root / "fixtures" / "fixtures.json").read_text())

    img_a = (root / fixtures["image_a_path"]).read_bytes()
    img_b = (root / fixtures["image_b_path"]).read_bytes()
    question = fixtures["question"]
    exp_a = fixtures["expected_a_contains"]
    exp_b = fixtures["expected_b_contains"]

    r_a = httpx.post(
        f"{base_url}/api/vqa",
        files={"file": ("image_a.jpg", img_a, "image/jpeg")},
        data={"question": question},
        timeout=120,
    )
    if r_a.status_code != 200:
        raise AssertionError(
            f"req-5-grounding-proof: image_a request failed {r_a.status_code}: {r_a.text[:200]}"
        )
    answer_a = r_a.json()["answer"]

    r_b = httpx.post(
        f"{base_url}/api/vqa",
        files={"file": ("image_b.jpg", img_b, "image/jpeg")},
        data={"question": question},
        timeout=120,
    )
    if r_b.status_code != 200:
        raise AssertionError(
            f"req-5-grounding-proof: image_b request failed {r_b.status_code}: {r_b.text[:200]}"
        )
    answer_b = r_b.json()["answer"]

    if exp_a.lower() not in answer_a.lower():
        raise AssertionError(
            f"req-5-grounding-proof: expected '{exp_a}' in image_a answer, got: {answer_a!r}"
        )
    if exp_b.lower() not in answer_b.lower():
        raise AssertionError(
            f"req-5-grounding-proof: expected '{exp_b}' in image_b answer, got: {answer_b!r}"
        )
    if answer_a == answer_b:
        raise AssertionError(
            f"req-5-grounding-proof: answers must differ but both are: {answer_a!r}"
        )


def verify_req_6_audit_log(
    workspace: str,
    base_url: str = "http://localhost:8000",
    audit_source: str = "file",
) -> None:
    root = Path(workspace)
    fixtures = json.loads((root / "fixtures" / "fixtures.json").read_text())
    img_path = root / fixtures["image_a_path"]
    img_bytes = img_path.read_bytes()
    expected_digest = hashlib.sha256(img_bytes).hexdigest()

    def read_log() -> str:
        if audit_source == "container":
            result = subprocess.run(
                ["docker", "compose", "exec", "-T", "api", "cat", "/app/audit.log"],
                capture_output=True,
                text=True,
                cwd=workspace,
            )
            return result.stdout
        log_path = root / "audit.log"
        if not log_path.exists():
            return ""
        return log_path.read_text()

    lines_before = len(read_log().splitlines())

    r = httpx.post(
        f"{base_url}/api/vqa",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
        data={"question": fixtures["question"]},
        timeout=120,
    )
    if r.status_code != 200:
        raise AssertionError(f"req-6-audit-log: request failed {r.status_code}: {r.text[:200]}")

    time.sleep(0.2)
    log_content = read_log()
    lines_after = log_content.splitlines()

    if len(lines_after) <= lines_before:
        raise AssertionError(
            f"req-6-audit-log: no new line added to audit log (had {lines_before}, "
            f"now {len(lines_after)})"
        )

    new_lines = lines_after[lines_before:]
    found = any(expected_digest in line for line in new_lines)
    if not found:
        raise AssertionError(
            f"req-6-audit-log: digest {expected_digest[:16]}... not found in "
            f"new audit lines: {new_lines}"
        )


def verify_req_7_health_check(workspace: str, base_url: str = "http://localhost:8000") -> None:
    r = httpx.get(f"{base_url}/health", timeout=15)
    if r.status_code != 200:
        raise AssertionError(f"req-7-health-check: expected 200, got {r.status_code}")
    body = r.json()
    if "status" not in body:
        raise AssertionError(f"req-7-health-check: response missing 'status' key: {body}")


def verify_req_8_docker_compose(
    workspace: str,
    base_url: str = "http://localhost:8000",
    check_container: bool = False,
) -> None:
    root = Path(workspace)
    compose_path = root / "docker-compose.yml"
    if not compose_path.exists():
        raise AssertionError("req-8-docker-compose: docker-compose.yml not found")

    try:
        data = yaml.safe_load(compose_path.read_text())
    except yaml.YAMLError as exc:
        raise AssertionError(
            f"req-8-docker-compose: docker-compose.yml is invalid YAML: {exc}"
        ) from exc

    services = data.get("services", {})
    if not services:
        raise AssertionError("req-8-docker-compose: no services defined")

    service_name = next(iter(services))
    svc = services[service_name]

    if "build" not in svc:
        raise AssertionError(f"req-8-docker-compose: service '{service_name}' has no 'build' key")

    ports = svc.get("ports", [])
    mapped = any("8000" in str(p) for p in ports)
    if not mapped:
        raise AssertionError(
            f"req-8-docker-compose: port 8000 not mapped in service '{service_name}'"
        )

    env_file = svc.get("env_file", [])
    if not env_file:
        raise AssertionError(f"req-8-docker-compose: service '{service_name}' missing env_file")

    hc = svc.get("healthcheck", {})
    hc_cmd = str(hc.get("test", ""))
    if "curl" not in hc_cmd or "/health" not in hc_cmd:
        raise AssertionError(
            f"req-8-docker-compose: healthcheck must contain 'curl' and '/health', got: {hc_cmd!r}"
        )

    if check_container:
        id_result = subprocess.run(
            ["docker", "compose", "ps", "-q", service_name],
            capture_output=True,
            text=True,
            cwd=workspace,
        )
        container_id = id_result.stdout.strip()
        if not container_id:
            raise AssertionError(f"req-8-docker-compose: container '{service_name}' not running")

        health_result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_id],
            capture_output=True,
            text=True,
        )
        health_status = health_result.stdout.strip()
        if health_status != "healthy":
            raise AssertionError(
                f"req-8-docker-compose: container health is '{health_status}', expected 'healthy'"
            )

        r = httpx.get(f"{base_url}/health", timeout=15)
        if r.status_code != 200:
            raise AssertionError(
                f"req-8-docker-compose: /health returned {r.status_code} from container"
            )


def verify_req_9_env_example(workspace: str) -> None:
    root = Path(workspace)
    env_path = root / ".env.example"
    if not env_path.exists():
        raise AssertionError("req-9-env-example: .env.example not found")

    text = env_path.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise AssertionError(f"req-9-env-example: line is not KEY=VALUE: {stripped!r}")

    if "GROQ_API_KEY" not in text:
        raise AssertionError("req-9-env-example: GROQ_API_KEY not documented in .env.example")
    if "OPENAI_API_KEY" not in text:
        raise AssertionError("req-9-env-example: OPENAI_API_KEY not documented in .env.example")


def verify_req_10_submission_config(workspace: str) -> None:
    root = Path(workspace)
    sub_path = root / "submission.json"
    if not sub_path.exists():
        raise AssertionError("req-10-submission-config: submission.json not found")

    try:
        data = json.loads(sub_path.read_text())
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"req-10-submission-config: submission.json is not valid JSON: {exc}"
        ) from exc

    if set(data.keys()) != {"vlm_provider", "model_name"}:
        raise AssertionError(
            f"req-10-submission-config: expected keys {{vlm_provider, model_name}}, "
            f"got {set(data.keys())}"
        )
    if not isinstance(data["vlm_provider"], str):
        raise AssertionError("req-10-submission-config: vlm_provider must be a string")
    if not isinstance(data["model_name"], str):
        raise AssertionError("req-10-submission-config: model_name must be a string")


def _run_all(
    workspace: str,
    base_url: str,
    audit_source: str,
    check_container: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    results = []
    warnings = []

    checks = [
        ("req-1-fixtures", lambda: verify_req_1_fixtures(workspace)),
        (
            "req-2-validation-empty-question",
            lambda: verify_req_2_validation_empty_question(workspace, base_url),
        ),
        (
            "req-3-validation-missing-image",
            lambda: verify_req_3_validation_missing_image(workspace, base_url),
        ),
        (
            "req-4-vqa-endpoint-success",
            lambda: verify_req_4_vqa_endpoint_success(workspace, base_url),
        ),
        ("req-5-grounding-proof", lambda: verify_req_5_grounding_proof(workspace, base_url)),
        ("req-6-audit-log", lambda: verify_req_6_audit_log(workspace, base_url, audit_source)),
        ("req-7-health-check", lambda: verify_req_7_health_check(workspace, base_url)),
        (
            "req-8-docker-compose",
            lambda: verify_req_8_docker_compose(workspace, base_url, check_container),
        ),
        ("req-9-env-example", lambda: verify_req_9_env_example(workspace)),
        ("req-10-submission-config", lambda: verify_req_10_submission_config(workspace)),
    ]

    for req_id, fn in checks:
        try:
            fn()
            results.append({"id": req_id, "result": "pass", "detail": ""})
        except AssertionError as exc:
            results.append({"id": req_id, "result": "fail", "detail": str(exc)})
        except Exception as exc:
            results.append(
                {"id": req_id, "result": "error", "detail": f"{type(exc).__name__}: {exc}"}
            )

    return results, warnings


def _print_table(results: list[dict[str, Any]]) -> None:
    id_w = max(len(r["id"]) for r in results)
    res_w = 6
    print(f"\n{'requirement id':<{id_w}}  {'result':<{res_w}}  detail")
    print("-" * (id_w + res_w + 40))
    for r in results:
        print(f"{r['id']:<{id_w}}  {r['result']:<{res_w}}  {r['detail']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all ten VQA API requirements.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--audit-source", choices=["file", "container"], default="file")
    parser.add_argument("--check-container", action="store_true")
    args = parser.parse_args()

    results, warnings = _run_all(
        workspace=args.workspace,
        base_url=args.base_url,
        audit_source=args.audit_source,
        check_container=args.check_container,
    )

    _print_table(results)

    passed = [r["id"] for r in results if r["result"] == "pass"]
    all_pass = len(passed) == 10

    summary = {
        "status": "pass" if all_pass else "fail",
        "checked": passed,
        "metrics": {"primary": f"{len(passed)}/10 requirements passed"},
        "warnings": warnings,
    }
    print(json.dumps(summary))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

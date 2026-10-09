"""Fixture image generator verification and live tuning runner."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def run_live_tuning(base_url: str = "http://localhost:8000") -> None:
    import httpx

    fixtures_path = Path(__file__).resolve().parents[1] / "fixtures" / "fixtures.json"
    fixtures = json.loads(fixtures_path.read_text())

    project_root = Path(__file__).resolve().parents[1]
    image_a = (project_root / fixtures["image_a_path"]).read_bytes()
    image_b = (project_root / fixtures["image_b_path"]).read_bytes()
    question = fixtures["question"]

    print("Running 5 requests per image against the live server...\n")

    def send(img_bytes: bytes, label: str) -> list[str]:
        answers = []
        for i in range(1, 6):
            t0 = time.monotonic()
            r = httpx.post(
                f"{base_url}/api/vqa",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                data={"question": question},
                timeout=120,
            )
            elapsed = round((time.monotonic() - t0) * 1000)
            ans = r.json().get("answer", f"ERROR: {r.text}")
            print(f"{label} run {i}: {ans!r}  ({elapsed}ms)")
            answers.append(ans)
        return answers

    answers_a = send(image_a, "image_a")
    answers_b = send(image_b, "image_b")

    exp_a = fixtures["expected_a_contains"]
    exp_b = fixtures["expected_b_contains"]

    ok_a = all(exp_a in a for a in answers_a)
    ok_b = all(exp_b in a for a in answers_b)
    wrong_a = any(exp_b in a for a in answers_a)
    wrong_b = any(exp_a in a for a in answers_b)

    print(f"\nimage_a all contain '{exp_a}': {ok_a}")
    print(f"image_a none contain '{exp_b}': {not wrong_a}")
    print(f"image_b all contain '{exp_b}': {ok_b}")
    print(f"image_b none contain '{exp_a}': {not wrong_b}")

    if not (ok_a and ok_b and not wrong_a and not wrong_b):
        print("\nFAIL: grounding requirements not met")
        sys.exit(1)
    print("\nPASS: all 10 runs produced correct grounded answers")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    run_live_tuning(args.base_url)

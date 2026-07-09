import json
import os
from pathlib import Path

import pytest
import requests


BASE_URL = os.getenv("SYSTEMINFORANDE_BASE_URL", "https://helmfridsson-systeminforande.hf.space").rstrip("/")
TIMEOUT_SECONDS = float(os.getenv("SYSTEMINFORANDE_API_TIMEOUT", "60"))
SCENARIO_PATH = Path(__file__).parent / "data" / "live_api_scenarios.json"


def _load_scenarios():
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


def _post_question(question: str, **extra_payload):
    payload = {"question": question, **extra_payload}
    response = requests.post(
        f"{BASE_URL}/api/ask",
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@pytest.mark.live_api
def test_health_endpoint():
    response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "ok"
    assert "revision" in payload


@pytest.mark.live_api
def test_ready_endpoint():
    response = requests.get(f"{BASE_URL}/ready", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "ok"
    assert "revision" in payload


@pytest.mark.live_api
def test_api_ask_rejects_empty_question():
    response = requests.post(
        f"{BASE_URL}/api/ask",
        json={"question": "   "},
        timeout=TIMEOUT_SECONDS,
    )
    assert response.status_code == 400
    assert "question" in response.text.lower()


@pytest.mark.live_api
@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda item: item["id"])
def test_api_ask_regression_scenarios(scenario):
    payload = _post_question(scenario["question"])

    assert payload["question"] == scenario["question"]
    assert payload["normalized_question"] == scenario["question"].strip()
    assert payload["route"] in {"rag", "predefined"}
    assert isinstance(payload["answer_markdown"], str)
    assert payload["answer_markdown"].strip()
    assert isinstance(payload["sources"], list)
    assert payload["timing_ms"] >= 0

    answer_lower = payload["answer_markdown"].lower()
    assert any(keyword.lower() in answer_lower for keyword in scenario["answer_contains_any"])

    expected_sources = set(scenario["expected_source_any"])
    actual_sources = {source["source"] for source in payload["sources"] if source.get("source")}

    if expected_sources:
        assert actual_sources & expected_sources, (
            f"Expected one of {sorted(expected_sources)} but got {sorted(actual_sources)}"
        )
        for source in payload["sources"]:
            url = source.get("url")
            if source.get("source_type") == "pdf":
                assert url
                assert url.startswith("https://tomashelmfridsson.github.io/systeminforande/pdfs/")


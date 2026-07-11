import json
import os
import re
import time
from pathlib import Path

import pytest
import requests
from gradio_client import Client


BASE_URL = os.getenv("SYSTEMINFORANDE_BASE_URL", "https://helmfridsson-systeminforande.hf.space").rstrip("/")
TIMEOUT_SECONDS = float(os.getenv("SYSTEMINFORANDE_API_TIMEOUT", "60"))
DEFAULT_MODEL = os.getenv("SYSTEMINFORANDE_LLM_MODEL", "zai-org/GLM-5.2")
SCENARIO_PATH = Path(__file__).parent / "data" / "live_api_scenarios.json"


def _load_scenarios():
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def gradio_client():
    return Client(BASE_URL)


def _submit_question(client: Client, question: str, debug_mode: bool = False, llm_model: str = DEFAULT_MODEL) -> str:
    result = client.predict(question, debug_mode, llm_model, api_name="/submit")
    assert isinstance(result, str)
    return result


def _extract_sources(answer_markdown: str) -> set[str]:
    matches = re.findall(r"\[([^\]]+\.(?:pdf|PDF))\]\(", answer_markdown)
    return {match.strip() for match in matches}


def _extract_answer_body(answer_markdown: str) -> str:
    body = answer_markdown
    for marker in ("\n\n---\n\n### Källor", "\n\n---\n\n### Debug"):
        if marker in body:
            body = body.split(marker, 1)[0]
    return body.strip()


def _has_narrative_answer(answer_markdown: str) -> bool:
    body = _extract_answer_body(answer_markdown)
    if not body:
        return False

    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", body)
    body = re.sub(r"[#>*_`-]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) < 40:
        return False

    words = re.findall(r"[A-Za-zÅÄÖåäö0-9]+", body)
    return len(words) >= 8


@pytest.mark.live_api
def test_gradio_info_endpoint_is_available():
    response = requests.get(f"{BASE_URL}/gradio_api/info", timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    assert "/submit" in payload["named_endpoints"]


@pytest.mark.live_api
@pytest.mark.parametrize("scenario", _load_scenarios(), ids=lambda item: item["id"])
def test_submit_regression_scenarios(gradio_client, scenario):
    answer = _submit_question(gradio_client, scenario["question"])

    assert answer.strip()
    answer_lower = answer.lower()
    assert any(keyword.lower() in answer_lower for keyword in scenario["answer_contains_any"])

    expected_sources = set(scenario["expected_source_any"])
    if expected_sources:
        actual_sources = _extract_sources(answer)
        assert actual_sources & expected_sources, (
            f"Expected one of {sorted(expected_sources)} but got {sorted(actual_sources)}"
        )


@pytest.mark.live_api
@pytest.mark.parametrize(
    ("canonical_question", "misspelled_question", "expected_keyword"),
    [
        (
            "Hur används acceptanstest i införandet?",
            "Hur anvnds acceptanstst i införandet?",
            "acceptanstest",
        ),
        (
            "På vilket sätt bör implementeringen planeras och följas upp?",
            "På vilket sätt bör implmenteringen planeras och följas up?",
            "implementering",
        ),
    ],
)
def test_submit_is_robust_to_common_misspellings(
    gradio_client,
    canonical_question,
    misspelled_question,
    expected_keyword,
):
    canonical_answer = _submit_question(gradio_client, canonical_question)
    misspelled_answer = _submit_question(gradio_client, misspelled_question)

    assert expected_keyword.lower() in canonical_answer.lower()
    assert expected_keyword.lower() in misspelled_answer.lower()


@pytest.mark.live_api
def test_submit_rejects_unsupported_question_with_fallback(gradio_client):
    answer = _submit_question(gradio_client, "Vilken färg har månen i projektmodellen?")
    answer_lower = answer.lower()
    assert (
        "inte tillräckligt underlag" in answer_lower
        or "inte relevant stöd" in answer_lower
        or "källmaterialet" in answer_lower
    )


@pytest.mark.live_api
@pytest.mark.parametrize(
    ("question", "expected_keywords"),
    [
        (
            "Hur testar man ett nytt system",
            ["test", "acceptans", "verifier", "prestanda", "godkänn"],
        ),
        (
            "Vilka etapper finns det",
            ["etapp", "planering", "acceptanstest", "pilotdrift", "driftsättning"],
        ),
    ],
)
def test_submit_returns_narrative_before_sources_for_known_regressions(
    gradio_client,
    question,
    expected_keywords,
):
    answer = _submit_question(gradio_client, question)
    answer_lower = answer.lower()

    assert _has_narrative_answer(answer), (
        f"Expected narrative answer before sources for question {question!r}, got: {answer!r}"
    )
    assert any(keyword in answer_lower for keyword in expected_keywords)


@pytest.mark.live_api
def test_submit_known_regression_question_completes_within_budget(gradio_client):
    max_seconds = float(os.getenv("SYSTEMINFORANDE_MAX_REGRESSION_RESPONSE_SECONDS", "20"))
    started_at = time.perf_counter()
    answer = _submit_question(gradio_client, "Hur testar man ett nytt system")
    elapsed = time.perf_counter() - started_at

    assert answer.strip()
    assert elapsed <= max_seconds, (
        f"Regression question exceeded latency budget: {elapsed:.2f}s > {max_seconds:.2f}s"
    )

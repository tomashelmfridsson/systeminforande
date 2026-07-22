import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

Client = pytest.importorskip("gradio_client").Client


BASE_URL = os.getenv("SYSTEMINFORANDE_BASE_URL", "https://helmfridsson-systeminforande.hf.space").rstrip("/")
TIMEOUT_SECONDS = float(os.getenv("SYSTEMINFORANDE_API_TIMEOUT", "60"))
DEFAULT_MODEL = os.getenv("SYSTEMINFORANDE_LLM_MODEL", "openai/gpt-oss-120b")
SCENARIO_PATH = Path(__file__).parent / "data" / "live_api_scenarios.json"


def _load_scenarios():
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def gradio_client():
    return Client(BASE_URL)


def _submit_question(client: Client, question: str, debug_mode: bool = False, llm_model: str = DEFAULT_MODEL) -> str:
    # The live Gradio endpoint takes four inputs:
    # message, current_doc state, debug_mode, llm_model.
    result = client.predict(question, None, debug_mode, llm_model, api_name="/submit")
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


def _load_local_app_without_launch(monkeypatch):
    import gradio as gr
    import uvicorn

    def _noop_launch(self, *args, **kwargs):
        return None

    monkeypatch.setattr(gr.Blocks, "launch", _noop_launch)
    monkeypatch.setattr(gr, "mount_gradio_app", lambda app, *args, **kwargs: app)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    monkeypatch.setattr(
        app_module,
        "generate_retrieval_rewrite",
        lambda question, llm_rewrite: {
            "status": "fallback",
            "original_question": question,
            "retrieval_queries": [{"query": question, "purpose": "literal", "weight": 1.0}],
            "debug": {"dropped_queries": [], "fallback_reason": "local_test_no_agent1_llm"},
        },
    )
    return app_module


def _jsonl_records(log_dir: Path) -> list[dict]:
    records = []
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def test_local_api_ask_honors_explicit_llm_model_and_returns_structured_metadata(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask",
        json={
            "question": "Hur testar man ett nytt system?",
            "enable_synthesis": False,
            "llm_model": "Qwen/Qwen3-32B",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_model"] == "Qwen/Qwen3-32B"
    assert payload["enable_synthesis"] is False
    assert payload["retrieval"]["llm_synthesis_model"] == "Qwen/Qwen3-32B"
    assert payload["retrieval"]["llm_synthesis_enabled"] is False
    assert payload["retrieval"]["llm_status"]


def test_local_api_ask_uses_synthesis_by_default_when_not_explicitly_disabled(monkeypatch):
    monkeypatch.delenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", raising=False)
    app_module = _load_local_app_without_launch(monkeypatch)

    def _fake_final_grounded_answer(query, chunks, *, enable_synthesis, llm_model, llm_rewrite):
        return {
            "extractive_answer": "Extraktivt svar från källmaterialet.",
            "final_answer": "Naturligt LLM-omskrivet svar från källmaterialet.",
            "synthesis_enabled": enable_synthesis,
            "synthesis_used": True,
            "llm_model": llm_model,
            "llm_status": "rewrite_applied",
            "synthesis_prompt": "",
        }

    monkeypatch.setattr(app_module, "build_final_grounded_answer", _fake_final_grounded_answer)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask",
        json={
            "question": "Hur testar man ett nytt system?",
            "llm_model": "Qwen/Qwen3-32B",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["enable_synthesis"] is True
    assert payload["retrieval"]["llm_synthesis_enabled"] is True
    assert payload["retrieval"]["llm_synthesis_used"] is True
    assert payload["answer_markdown"].startswith("Naturligt LLM-omskrivet svar")


def test_local_api_ask_accepts_query_param_model_alias_when_body_model_missing(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask?LLM=Qwen/Qwen3-32B",
        json={
            "question": "Hur testar man ett nytt system?",
            "enable_synthesis": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_model"] == "Qwen/Qwen3-32B"
    assert payload["retrieval"]["llm_synthesis_model"] == "Qwen/Qwen3-32B"


def test_local_api_ask_honors_enable_synthesis_true_without_env_flag(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)

    def _fake_final_grounded_answer(query, chunks, *, enable_synthesis, llm_model, llm_rewrite):
        return {
            "extractive_answer": "Extraktivt svar från källmaterialet.",
            "final_answer": "Omskrivet källbundet svar från källmaterialet.",
            "synthesis_enabled": enable_synthesis,
            "synthesis_used": True,
            "llm_model": llm_model,
            "llm_status": "rewrite_applied",
            "synthesis_prompt": "",
        }

    monkeypatch.setattr(app_module, "build_final_grounded_answer", _fake_final_grounded_answer)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask",
        json={
            "question": "Hur testar man ett nytt system?",
            "enable_synthesis": True,
            "llm_model": "mistralai/Mistral-Small-4-119B-2603",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["enable_synthesis"] is True
    assert payload["llm_model"] == "mistralai/Mistral-Small-4-119B-2603"
    assert payload["retrieval"]["llm_synthesis_enabled"] is True
    assert payload["retrieval"]["llm_synthesis_used"] is True
    assert payload["retrieval"]["llm_synthesis_model"] == "mistralai/Mistral-Small-4-119B-2603"


def test_local_launch_mounts_custom_api_app(monkeypatch):
    launched_kwargs = []
    mounted_apps = []

    import gradio as gr
    import uvicorn

    def _capture_launch(self, *args, **kwargs):
        launched_kwargs.append(kwargs)
        return None

    def _capture_mount(app, *args, **kwargs):
        mounted_apps.append(app)
        return app

    monkeypatch.setattr(gr.Blocks, "launch", _capture_launch)
    monkeypatch.setattr(gr, "mount_gradio_app", _capture_mount)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    if launched_kwargs:
        assert launched_kwargs[-1]["_app"] is app_module.API_APP
    else:
        assert mounted_apps[-1] is app_module.API_APP


def test_local_health_and_ready_routes_are_on_mounted_api_app(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    client = TestClient(app_module.API_APP)

    health_response = client.get("/health")
    ready_response = client.get("/ready")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert ready_response.json()["status"] == "ok"


def test_usage_log_record_includes_huggingface_token_fields_when_present(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMINFORANDE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_LOGGING", "true")
    app_module = _load_local_app_without_launch(monkeypatch)
    llm_usage = app_module.summarize_llm_usage(
        [
            app_module._llm_usage_call_record(
                purpose="synthesis",
                model="Qwen/Qwen3-32B",
                usage={
                    "prompt_tokens": 13,
                    "completion_tokens": 8,
                    "total_tokens": 21,
                },
                status="ok",
            )
        ],
        "Qwen/Qwen3-32B",
    )

    app_module.log_usage_event(
        "api_question",
        question="Hur testar man ett nytt system?",
        answer="Med acceptanstest och verifiering.",
        route="rag",
        llm_model="Qwen/Qwen3-32B",
        metadata={"llm_usage": llm_usage},
    )

    records = [
        record
        for record in _jsonl_records(tmp_path / "logs")
        if record["event_type"] == "api_question"
    ]
    assert len(records) == 1
    logged_usage = records[0]["metadata"]["llm_usage"]
    assert logged_usage["provider"] == "huggingface_hub.InferenceClient.chat_completion"
    assert logged_usage["model"] == "Qwen/Qwen3-32B"
    assert logged_usage["prompt_tokens"] == 13
    assert logged_usage["completion_tokens"] == 8
    assert logged_usage["total_tokens"] == 21
    assert logged_usage["calls"] == 1
    assert logged_usage["missing"] is False
    assert logged_usage["calls_detail"] == [
        {
            "purpose": "synthesis",
            "provider": "huggingface_hub.InferenceClient.chat_completion",
            "model": "Qwen/Qwen3-32B",
            "prompt_tokens": 13,
            "completion_tokens": 8,
            "total_tokens": 21,
            "missing": False,
            "status": "ok",
            "error": None,
        }
    ]


def test_usage_log_record_tolerates_absent_and_partially_missing_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("SYSTEMINFORANDE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_LOGGING", "true")
    app_module = _load_local_app_without_launch(monkeypatch)
    llm_usage = app_module.summarize_llm_usage(
        [
            app_module._llm_usage_call_record(
                purpose="synthesis",
                model="Qwen/Qwen3-32B",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                status="ok",
            ),
            app_module._llm_usage_call_record(
                purpose="debug_comparison",
                model="Qwen/Qwen3-32B",
                usage=None,
                status="ok",
            ),
        ],
        "Qwen/Qwen3-32B",
    )

    app_module.log_usage_event(
        "api_question",
        question="Hur testar man ett nytt system?",
        answer="Med acceptanstest och verifiering.",
        route="rag",
        llm_model="Qwen/Qwen3-32B",
        metadata={"llm_usage": llm_usage},
    )

    records = [
        record
        for record in _jsonl_records(tmp_path / "logs")
        if record["event_type"] == "api_question"
    ]
    assert len(records) == 1
    logged_usage = records[0]["metadata"]["llm_usage"]
    assert logged_usage["calls"] == 2
    assert logged_usage["prompt_tokens"] is None
    assert logged_usage["completion_tokens"] is None
    assert logged_usage["total_tokens"] is None
    assert logged_usage["missing"] is True
    assert logged_usage["calls_detail"][0]["prompt_tokens"] == 10
    assert logged_usage["calls_detail"][0]["completion_tokens"] is None
    assert logged_usage["calls_detail"][1]["missing"] is True


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

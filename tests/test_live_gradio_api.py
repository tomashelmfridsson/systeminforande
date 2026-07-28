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
UNSUPPORTED_FALLBACK_KEYWORDS = (
    "inte tillräckligt underlag",
    "inte relevant stöd",
    "källmaterialet",
    "kan inte verifiera",
    "tillgängliga källorna",
)


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


def _agentic_pipeline_chunk() -> dict:
    return {
        "id": "chunk-1",
        "source": "Utbildningsstrategi.pdf",
        "source_type": "pdf",
        "title": "Utbildningsstrategi",
        "text": "Utbildningsstrategin ska beskriva syfte, målgrupper, utbildningsbehov och genomförande.",
        "pages": [1],
    }


def _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls: list[str]) -> None:
    chunk = _agentic_pipeline_chunk()

    def fake_search(query, top_k=5, retrieval_rewrite=None):
        calls.append("retrieval_with_rewrite" if retrieval_rewrite else "retrieval")
        return [(12.0, chunk)]

    def fake_explain_search(query, top_k=5, retrieval_rewrite=None):
        return {
            "query": query,
            "query_terms": ["utbildningsstrategi"],
            "expanded_query_terms": ["utbildningsstrategi"],
            "intent": "definition",
            "top_results": [
                {
                    "score": 12.0,
                    "parts": {"bm25": 10, "title_overlap": 1, "definition_boost": 1, "domain_boost": 0, "intent_boost": 0},
                    "chunk": chunk,
                    "matched_terms": ["utbildningsstrategi"],
                }
            ],
            "agentic_retrieval": {
                "accepted_variants": [
                    {"query": query, "purpose": "literal", "weight": 1.0},
                    {"query": "utbildningsstrategi målgrupper", "purpose": "synonym", "weight": 0.88},
                ],
                "rejected_variants": [{"query": "generiska utbildningsråd", "reason": "semantic_drift"}],
                "merged_ranking": [],
            } if retrieval_rewrite else {},
        }

    monkeypatch.setattr(app_module, "search", fake_search)
    monkeypatch.setattr(app_module, "explain_search", fake_explain_search)
    monkeypatch.setattr(app_module, "filter_allowed_results", lambda results: results)
    monkeypatch.setattr(app_module, "_has_relevant_rag_support", lambda search_debug: (True, ""))
    monkeypatch.setattr(app_module, "build_sources_md", lambda results: "\n\n### Källor\n- Utbildningsstrategi.pdf")


def test_agentic_rag_feature_flag_disabled_preserves_non_agentic_retrieval_path(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "false")
    calls: list[str] = []
    _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls)

    def fail_if_agent1_runs(*args, **kwargs):
        raise AssertionError("Agent 1 should not run when agentic RAG is disabled")

    monkeypatch.setattr(app_module, "generate_retrieval_rewrite", fail_if_agent1_runs)
    monkeypatch.setattr(
        app_module,
        "build_final_grounded_answer",
        lambda *args, **kwargs: {
            "extractive_answer": "Källbundet bassvar om utbildningsstrategi.",
            "final_answer": "Källbundet bassvar om utbildningsstrategi.",
            "synthesis_used": False,
            "llm_status": "disabled",
        },
    )

    response = app_module.build_rag_response("Vad är en utbildningsstrategi?", debug=False, llm_model="model-a", enable_synthesis=False)

    assert calls == ["retrieval"]
    assert response["retrieval"]["agentic_rag_enabled"] is False
    assert response["retrieval"].get("agentic_pipeline") is None
    assert response["retrieval"]["agentic_retrieval"] == {}


def test_agentic_rag_feature_flag_enabled_calls_three_roles_in_order_uses_reviewed_answer_and_skips_debug_comparison(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")
    calls: list[str] = []
    _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls)

    def fake_agent1(question, llm_rewrite, **kwargs):
        calls.append("agent1")
        return {
            "status": "ok",
            "model": "agent1-model",
            "retrieval_queries": [{"query": question, "purpose": "literal", "weight": 1.0}],
            "debug": {"fallback_reason": None, "dropped_queries": []},
        }

    def fake_agent2(original_question, chunks, rewrite_metadata, llm_answer, **kwargs):
        calls.append("agent2")
        return {
            "status": "ok",
            "model": "agent2-model",
            "answer": "Utbildningsstrategin beskriver syfte, målgrupper, utbildningsbehov och genomförande.",
            "answer_scope": "direct",
            "evidence_ids_used": ["chunk-1"],
            "debug": {"fallback_reason": None},
        }

    def fake_agent3(original_question, draft_answer, evidence_snippets, evidence_ids, llm_review, **kwargs):
        calls.append("agent3")
        assert original_question == "Vad är en utbildningsstrategi?"
        assert draft_answer == "Utbildningsstrategin beskriver syfte, målgrupper, utbildningsbehov och genomförande."
        assert evidence_ids == ["chunk-1"]
        return {
            "status": "approved",
            "model": "agent3-model",
            "reason": "Svaret är källbundet.",
            "revision": "",
            "evidence_ids_used": ["chunk-1"],
            "debug": {"fallback_reason": None},
        }

    def fake_fallback_answer(query, chunks, **kwargs):
        return {
            "extractive_answer": "Extraktivt reservsvar.",
            "final_answer": "Extraktivt reservsvar.",
            "synthesis_used": False,
            "llm_status": "disabled",
        }

    monkeypatch.setattr(app_module, "generate_retrieval_rewrite", fake_agent1)
    monkeypatch.setattr(app_module, "generate_evidence_answer", fake_agent2)
    monkeypatch.setattr(app_module, "generate_answer_review", fake_agent3)
    monkeypatch.setattr(app_module, "build_final_grounded_answer", fake_fallback_answer)

    response = app_module.build_rag_response("Vad är en utbildningsstrategi?", debug=True, llm_model="model-a", enable_synthesis=True)

    assert calls[:4] == ["agent1", "retrieval_with_rewrite", "agent2", "agent3"]
    assert "debug_comparison" not in [item["purpose"] for item in response["llm_usage"]["calls_detail"]]
    assert response["answer_markdown"].startswith("Utbildningsstrategin beskriver syfte")
    assert response["retrieval"]["agentic_rag_enabled"] is True
    assert response["retrieval"]["agentic_pipeline"]["review_status"] == "approved"


def test_agentic_rag_usage_metadata_includes_per_agent_tokens_latency_counts_and_fallback(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")
    calls: list[str] = []
    _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls)

    def fake_agent1(question, llm_rewrite, **kwargs):
        llm_rewrite("agent1 prompt", "agent1-model")
        return {"status": "ok", "model": "agent1-model", "retrieval_queries": [{"query": question, "purpose": "literal", "weight": 1.0}], "debug": {"fallback_reason": None, "dropped_queries": []}}

    def fake_agent2(original_question, chunks, rewrite_metadata, llm_answer, **kwargs):
        llm_answer("agent2 prompt", "agent2-model")
        return {"status": "ok", "model": "agent2-model", "answer": "Utbildningsstrategin beskriver syfte, målgrupper, utbildningsbehov och genomförande.", "answer_scope": "direct", "evidence_ids_used": ["chunk-1"], "debug": {"fallback_reason": None}}

    def fake_agent3(original_question, draft_answer, evidence_snippets, evidence_ids, llm_review, **kwargs):
        llm_review("agent3 prompt", "agent3-model")
        return {"status": "approved", "model": "agent3-model", "revision": "", "evidence_ids_used": ["chunk-1"], "debug": {"fallback_reason": None}}

    def fake_fallback_answer(query, chunks, **kwargs):
        return {"extractive_answer": "Utbildningsstrategin beskriver syfte.", "final_answer": "Utbildningsstrategin beskriver syfte.", "synthesis_used": False, "llm_status": "disabled"}

    def fake_llm(prompt, model, *, purpose, usage_records):
        usage_records.append({
            "purpose": purpose,
            "provider": "test",
            "model": model,
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "latency_ms": 12.5,
            "missing": False,
            "status": "ok",
            "error": None,
        })
        return "{}"

    monkeypatch.setattr(app_module, "generate_retrieval_rewrite", fake_agent1)
    monkeypatch.setattr(app_module, "generate_evidence_answer", fake_agent2)
    monkeypatch.setattr(app_module, "generate_answer_review", fake_agent3)
    monkeypatch.setattr(app_module, "build_final_grounded_answer", fake_fallback_answer)
    monkeypatch.setattr(app_module, "safe_generate_reasoning_from_prompt_with_usage_records", fake_llm)

    response = app_module.build_rag_response("Vad är en utbildningsstrategi?", debug=False, llm_model="model-a", enable_synthesis=True)
    metadata = response["retrieval"]["agentic_pipeline"]

    assert [agent["role"] for agent in metadata["agents"]] == ["agent1_retrieval_rewrite", "agent2_evidence_answer", "agent3_grounded_review"]
    assert metadata["accepted_rewrites"] == 2
    assert metadata["rejected_rewrites"] == 1
    assert metadata["fallback_reason"] is None
    assert metadata["usage"]["prompt_tokens"] == 300
    assert metadata["usage"]["completion_tokens"] == 75
    assert metadata["usage"]["total_tokens"] == 375
    assert metadata["usage"]["calls"] == 3
    assert metadata["usage"]["calls_detail"][0]["latency_ms"] == 12.5
    agent_usage = {agent["role"]: agent["usage"] for agent in metadata["agents"]}
    assert agent_usage["agent1_retrieval_rewrite"]["total_tokens"] == 125
    assert agent_usage["agent2_evidence_answer"]["total_tokens"] == 125
    assert agent_usage["agent3_grounded_review"]["model"] == "agent3-model"
    assert agent_usage["agent3_grounded_review"]["latency_ms"] == 12.5


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

    actual_sources = _extract_sources(answer)
    if scenario.get("source_policy") == "any":
        assert actual_sources, f"Expected at least one cited source, got answer: {answer!r}"
    elif scenario.get("source_policy") == "none":
        assert not actual_sources, f"Expected no cited sources for unsupported question, got {sorted(actual_sources)}"
    else:
        expected_sources = set(scenario["expected_source_any"])
        if not expected_sources:
            return
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
    assert any(keyword in answer_lower for keyword in UNSUPPORTED_FALLBACK_KEYWORDS)


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

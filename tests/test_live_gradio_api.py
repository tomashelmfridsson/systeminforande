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
from rag.agentic_answer import parse_evidence_answer_response

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
        lambda question, llm_rewrite, **kwargs: {
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
    evidence_filter_calls: list[list[str]] = []
    agent_answer = (
        "Utbildningsstrategin beskriver syfte, målgrupper, utbildningsbehov och genomförande."
        "\n\n**Källor**\n- Modellgenererad_källa.pdf"
    )
    _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls)
    original_filter = app_module.filter_results_by_evidence_ids

    def tracking_filter(results, evidence_ids):
        evidence_filter_calls.append(list(evidence_ids))
        return original_filter(results, evidence_ids)

    monkeypatch.setattr(app_module, "filter_results_by_evidence_ids", tracking_filter)

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
            "answer": agent_answer,
            "answer_scope": "direct",
            "evidence_ids_used": ["chunk-1"],
            "debug": {"fallback_reason": None},
        }

    def fake_agent3(original_question, draft_answer, evidence_snippets, evidence_ids, llm_review, **kwargs):
        calls.append("agent3")
        assert original_question == "Vad är en utbildningsstrategi?"
        assert draft_answer == agent_answer
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
    assert evidence_filter_calls == [["chunk-1"]]
    assert response["answer_markdown"].count("Källor") == 1
    assert "Modellgenererad_källa.pdf" not in response["answer_markdown"]


def test_filter_results_by_evidence_ids_keeps_only_agent3_approved_chunks(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    results = [
        (9.0, {"id": "used:1", "source": "Used.pdf"}),
        (8.0, {"id": "retrieved:2", "source": "RetrievedButUnused.pdf"}),
    ]

    filtered = app_module.filter_results_by_evidence_ids(results, ["used:1"])

    assert filtered == [results[0]]


def test_relevance_gate_rejects_high_scoring_agent_drift_that_only_matches_generic_best_term(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    search_debug = {
        "query_terms": ["bodtennis", "gummi", "bast"],
        "expanded_query_terms": ["bodtennis", "gummi", "bast"],
        "top_results": [
            {
                "score": 15.0,
                "chunk": {
                    "id": "irrelevant-1",
                    "source": "Mallar_acceptanstest_testrapport.pdf",
                    "title": "Sammanfattning och rekommendation",
                    "text": "Välj den bästa rekommendationen för systeminförandet.",
                },
            }
        ],
        "agentic_retrieval": {
            "accepted_variants": [
                {"query": "Vilket bodtennis gummi är bäst?"},
                {"query": "bäst kravmall testrapport sammanfattning rekommendation"},
            ],
            "merged_ranking": [
                {
                    "chunk_id": "irrelevant-1",
                    "original_query_match": False,
                }
            ],
        },
    }

    supported, reason = app_module._has_relevant_rag_support(search_debug)

    assert supported is False
    assert "originalfrågans ämne" in reason


def test_relevance_gate_preserves_supported_domain_questions(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)

    for question in [
        "Hur planerar man implementation av system?",
        "Vad är en utbildningsstrategi?",
        "Vilka etapper finns?",
    ]:
        supported, reason = app_module._has_relevant_rag_support(
            app_module.explain_search(question, top_k=5)
        )
        assert supported is True, f"{question}: {reason}"


def test_out_of_domain_table_tennis_question_returns_no_answer_sources_or_agent2_call(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")

    drifted_rewrite = {
        "status": "ok",
        "model": app_module.AGENT1_MODEL,
        "original_question": "Vilket bodtennis gummi är bäst?",
        "retrieval_queries": [
            {
                "query": "Vilket bodtennis gummi är bäst?",
                "purpose": "literal",
                "weight": 1.0,
            },
            {
                "query": "bäst kravmall testrapport sammanfattning rekommendation",
                "purpose": "broader_context",
                "weight": 0.8,
            },
        ],
        "debug": {"fallback_reason": None, "dropped_queries": []},
    }
    monkeypatch.setattr(
        app_module,
        "generate_retrieval_rewrite",
        lambda *args, **kwargs: drifted_rewrite,
    )

    def fail_if_agent2_runs(*args, **kwargs):
        raise AssertionError("Agent 2 must not run for an out-of-domain question")

    monkeypatch.setattr(app_module, "generate_evidence_answer", fail_if_agent2_runs)

    response = app_module.build_rag_response(
        "Vilket bodtennis gummi är bäst?",
        debug=False,
        llm_model="openai/gpt-oss-20b",
        enable_synthesis=True,
        enable_agentic_rag=True,
    )

    assert response["answer_markdown"] == app_module.grounded_answer_or_fallback("")
    assert response["sources"] == []
    assert response["homepage_links"] == []
    assert response["retrieval"]["relevance_supported"] is False
    assert "Källor" not in response["answer_markdown"]


def test_gradio_request_url_can_disable_default_agentic_rag(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")
    received = {}

    def fake_build_rag_response(query, debug, llm_model, enable_synthesis, *, enable_agentic_rag):
        received["enable_agentic_rag"] = enable_agentic_rag
        return {
            "route": "rag",
            "answer_markdown": "Äldre RAG-väg.",
            "sources": [],
            "homepage_links": [],
            "retrieval": {"llm_usage": {}},
            "llm_usage": {},
        }

    class FakeGradioRequest:
        query_params = {"enable_agentic_rag": "false"}

    monkeypatch.setattr(app_module, "build_rag_response", fake_build_rag_response)

    response = app_module.answer_question(
        "Vilka etapper finns?",
        llm_model="model-a",
        request=FakeGradioRequest(),
    )

    assert received["enable_agentic_rag"] is False
    assert response["enable_agentic_rag"] is False


def test_agentic_rag_usage_metadata_includes_per_agent_tokens_latency_counts_and_fallback(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")
    calls: list[str] = []
    max_tokens_by_purpose: dict[str, int] = {}
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

    def fake_llm(prompt, model, *, purpose, usage_records, max_tokens=1200):
        max_tokens_by_purpose[purpose] = max_tokens
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
    assert metadata["final_status"] == "approved"
    assert metadata["escalation"]["attempted"] is False
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
    assert max_tokens_by_purpose == {
        "agent1_retrieval_rewrite": 1200,
        "agent2_evidence_answer": 1800,
        "agent3_grounded_review": 1000,
    }


def test_agentic_rag_rejected_agent_answer_uses_one_120b_correction(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")
    calls: list[str] = []
    correction_calls = []
    _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls)

    monkeypatch.setattr(
        app_module,
        "generate_retrieval_rewrite",
        lambda question, llm_rewrite, **kwargs: {
            "status": "ok",
            "model": app_module.AGENT1_MODEL,
            "retrieval_queries": [{"query": question, "purpose": "literal", "weight": 1.0}],
            "debug": {"fallback_reason": None, "dropped_queries": []},
        },
    )
    monkeypatch.setattr(
        app_module,
        "generate_evidence_answer",
        lambda *args, **kwargs: {
            "status": "ok",
            "model": app_module.AGENT2_MODEL,
            "answer": "Ett agentsvar som Agent 3 underkänner.",
            "answer_scope": "direct",
            "evidence_ids_used": ["chunk-1"],
            "debug": {"fallback_reason": None},
        },
    )
    monkeypatch.setattr(
        app_module,
        "generate_answer_review",
        lambda *args, **kwargs: {
            "status": "rejected",
            "model": app_module.AGENT3_MODEL,
            "reason": "Svaret är inte tillräckligt källbundet.",
            "revision": "",
            "evidence_ids_used": [],
            "debug": {"fallback_reason": "agent3_grounding_failed"},
        },
    )

    def fake_correction(
        original_question,
        draft_answer,
        review_reason,
        chunks,
        rewrite_metadata,
        llm_answer,
        *,
        model,
    ):
        correction_calls.append((review_reason, model))
        llm_answer("compact correction prompt", model)
        return {
            "status": "ok",
            "model": model,
            "answer": "Det korrigerade svaret beskriver utbildningsstrategins syfte och målgrupper.",
            "answer_scope": "direct",
            "evidence_ids_used": ["chunk-1"],
            "debug": {"fallback_reason": None},
        }

    def fake_llm(prompt, model, *, purpose, usage_records, max_tokens=1200):
        usage_records.append({
            "purpose": purpose,
            "provider": "test",
            "model": model,
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "latency_ms": 10.0,
            "missing": False,
            "status": "ok",
            "error": None,
        })
        return "{}"

    def corrected_answer(query, chunks, **kwargs):
        assert kwargs["enable_synthesis"] is False
        assert kwargs["fallback_answer"].startswith("Det korrigerade svaret")
        return {
            "extractive_answer": kwargs["fallback_answer"],
            "final_answer": kwargs["fallback_answer"],
            "synthesis_used": False,
            "llm_status": "disabled",
        }

    monkeypatch.setattr(app_module, "generate_corrected_evidence_answer", fake_correction)
    monkeypatch.setattr(app_module, "safe_generate_reasoning_from_prompt_with_usage_records", fake_llm)
    monkeypatch.setattr(app_module, "build_final_grounded_answer", corrected_answer)

    response = app_module.build_rag_response(
        "Vad är en utbildningsstrategi?",
        debug=False,
        llm_model="model-a",
        enable_synthesis=True,
    )

    assert response["answer_markdown"].startswith("Det korrigerade svaret")
    assert correction_calls == [
        ("Svaret är inte tillräckligt källbundet.", "openai/gpt-oss-120b")
    ]
    assert calls == ["retrieval_with_rewrite"]
    assert response["retrieval"]["llm_status"] == "agentic_120b_correction"
    assert response["retrieval"]["agentic_pipeline"]["final_status"] == "corrected"
    assert response["retrieval"]["agentic_pipeline"]["fallback_reason"] is None
    escalation = response["retrieval"]["agentic_pipeline"]["escalation"]
    assert escalation["attempted"] is True
    assert escalation["status"] == "ok"
    assert escalation["usage"]["calls"] == 1
    assert [call["purpose"] for call in response["llm_usage"]["calls_detail"]] == [
        "agent2_120b_correction"
    ]


def test_agentic_rag_failed_correction_preserves_extractive_fallback_without_large_synthesis(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_AGENTIC_RAG", "true")
    calls: list[str] = []
    _stub_common_agentic_pipeline_rag(app_module, monkeypatch, calls)

    monkeypatch.setattr(
        app_module,
        "generate_retrieval_rewrite",
        lambda question, llm_rewrite, **kwargs: {
            "status": "ok",
            "model": "agent1-model",
            "retrieval_queries": [{"query": question, "purpose": "literal", "weight": 1.0}],
            "debug": {"fallback_reason": None, "dropped_queries": []},
        },
    )
    monkeypatch.setattr(
        app_module,
        "generate_evidence_answer",
        lambda *args, **kwargs: {
            "status": "ok",
            "model": "agent2-model",
            "answer": "Ett agentsvar som Agent 3 underkänner.",
            "answer_scope": "direct",
            "evidence_ids_used": ["chunk-1"],
            "debug": {"fallback_reason": None},
        },
    )
    monkeypatch.setattr(
        app_module,
        "generate_answer_review",
        lambda *args, **kwargs: {
            "status": "rejected",
            "model": "agent3-model",
            "revision": "",
            "evidence_ids_used": [],
            "debug": {"fallback_reason": "agent3_grounding_failed"},
        },
    )
    monkeypatch.setattr(
        app_module,
        "generate_corrected_evidence_answer",
        lambda *args, **kwargs: {
            "status": "fallback",
            "model": app_module.AGENT_CORRECTION_MODEL,
            "answer": "",
            "evidence_ids_used": [],
            "debug": {"fallback_reason": "agent2_invalid_json"},
        },
    )

    def _extractive_fallback(query, chunks, **kwargs):
        assert "fallback_answer" not in kwargs
        assert kwargs["enable_synthesis"] is False
        assert kwargs["llm_rewrite"] is None
        return {
            "extractive_answer": "Det källbundna extraktiva reservsvaret.",
            "final_answer": "Det källbundna extraktiva reservsvaret.",
            "synthesis_used": False,
            "llm_status": "disabled",
        }

    monkeypatch.setattr(app_module, "build_final_grounded_answer", _extractive_fallback)

    response = app_module.build_rag_response(
        "Vad är en utbildningsstrategi?",
        debug=False,
        llm_model="model-a",
        enable_synthesis=False,
    )

    assert response["answer_markdown"].startswith("Det källbundna extraktiva reservsvaret.")
    assert "agentsvar som Agent 3 underkänner" not in response["answer_markdown"]
    assert response["retrieval"]["agentic_pipeline"]["review_status"] == "rejected"
    assert response["retrieval"]["agentic_pipeline"]["final_status"] == "fallback"
    assert response["retrieval"]["agentic_pipeline"]["escalation"]["attempted"] is True
    assert response["retrieval"]["agentic_fallback_retrieval_used"] is True
    assert calls == ["retrieval_with_rewrite", "retrieval"]
    assert "agentic_fallback_synthesis" not in [
        call["purpose"] for call in response["llm_usage"]["calls_detail"]
    ]


def test_agent2_accepts_live_model_scope_alias_and_minimal_evidence_objects():
    question = "Hur lämnas systemet över till förvaltning efter införandet?"
    chunks = [
        {
            "id": "q22-1",
            "source": "Forvaltningsoverlamnande.pdf",
            "source_type": "pdf",
            "title": "Förvaltningsöverlämnande",
            "text": "Förvaltningsöverlämnandet beskriver mottagare, förvaltningsobjekt och en tidplan för överlämnandet.",
            "pages": [3],
        }
    ]
    raw = json.dumps(
        {
            "original_question": question,
            "answer": "Förvaltningsöverlämnandet beskriver mottagare, förvaltningsobjekt och en tidplan för överlämnandet.",
            "answer_scope": "sufficient",
            "evidence_used": [{"chunk_id": "q22-1"}],
            "evidence_ids_used": ["q22-1"],
            "unsupported_or_uncertain": [],
            "source_coverage": {
                "uses_retrieved_chunks": True,
                "answers_original_question": True,
                "ignores_metadata_as_facts": True,
            },
            "grounding_notes": "Svaret använder q22-1.",
        },
        ensure_ascii=False,
    )

    result = parse_evidence_answer_response(
        question,
        chunks,
        raw,
        rewrite_metadata={
            "semantic_terms": [
                {
                    "surface": "förvaltningsöverlämnande",
                    "normalized_family": "överlämna förvaltning",
                }
            ]
        },
    )

    assert result["status"] == "ok"
    assert result["answer_scope"] == "direct"
    assert result["evidence_ids_used"] == ["q22-1"]
    assert result["evidence_used"][0]["source"] == "Forvaltningsoverlamnande.pdf"
    assert result["evidence_used"][0]["pages"] == [3]


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


def test_local_api_ask_can_override_agentic_rag_per_request(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    received = {}

    def _fake_answer_question(**kwargs):
        received.update(kwargs)
        return {
            "normalized_question": kwargs["message"],
            "answer_markdown": "Ett strukturerat testsvar.",
            "route": "rag",
            "llm_model": kwargs["llm_model"],
            "timing_ms": 1.0,
            "sources": [],
            "llm_usage": {"calls": 0},
            "retrieval": {
                "agentic_rag_enabled": kwargs["enable_agentic_rag"],
                "agentic_pipeline": {"enabled": kwargs["enable_agentic_rag"]},
            },
        }

    monkeypatch.setattr(app_module, "answer_question", _fake_answer_question)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask",
        json={
            "question": "Hur testar man ett nytt system?",
            "enable_agentic_rag": True,
            "enable_synthesis": False,
            "llm_model": "openai/gpt-oss-120b",
        },
    )

    assert response.status_code == 200
    assert received["enable_agentic_rag"] is True
    assert response.json()["retrieval"]["agentic_rag_enabled"] is True


def test_local_api_ask_accepts_agentic_and_debug_url_overrides(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    received = {}

    def _fake_answer_question(**kwargs):
        received.update(kwargs)
        return {
            "normalized_question": kwargs["message"],
            "answer_markdown": "Ett strukturerat testsvar.",
            "route": "rag",
            "llm_model": kwargs["llm_model"],
            "timing_ms": 1.0,
            "sources": [],
            "llm_usage": {"calls": 0},
            "retrieval": {
                "agentic_rag_enabled": kwargs["enable_agentic_rag"],
                "agentic_pipeline": None,
            },
        }

    monkeypatch.setattr(app_module, "answer_question", _fake_answer_question)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask?enable_agentic_rag=false&debug=true",
        json={
            "question": "Hur testar man ett nytt system?",
            "llm_model": "openai/gpt-oss-120b",
        },
    )

    assert response.status_code == 200
    assert received["enable_agentic_rag"] is False
    assert received["debug_mode"] is True


def test_local_api_ask_body_override_has_priority_over_url(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    received = {}

    def _fake_answer_question(**kwargs):
        received.update(kwargs)
        return {
            "normalized_question": kwargs["message"],
            "answer_markdown": "Ett strukturerat testsvar.",
            "route": "rag",
            "llm_model": kwargs["llm_model"],
            "timing_ms": 1.0,
            "sources": [],
            "llm_usage": {"calls": 0},
            "retrieval": {
                "agentic_rag_enabled": kwargs["enable_agentic_rag"],
                "agentic_pipeline": None,
            },
        }

    monkeypatch.setattr(app_module, "answer_question", _fake_answer_question)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask?enable_agentic_rag=false&debug=false",
        json={
            "question": "Hur testar man ett nytt system?",
            "enable_agentic_rag": True,
            "debug_mode": True,
            "llm_model": "openai/gpt-oss-120b",
        },
    )

    assert response.status_code == 200
    assert received["enable_agentic_rag"] is True
    assert received["debug_mode"] is True


def test_local_api_ask_rejects_non_boolean_agentic_override(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask",
        json={
            "question": "Hur testar man ett nytt system?",
            "enable_agentic_rag": "true",
        },
    )

    assert response.status_code == 400
    assert "enable_agentic_rag" in response.json()["detail"]


def test_local_api_ask_rejects_invalid_boolean_url_override(monkeypatch):
    app_module = _load_local_app_without_launch(monkeypatch)
    client = TestClient(app_module.API_APP)

    response = client.post(
        "/api/ask?debug=maybe",
        json={"question": "Hur testar man ett nytt system?"},
    )

    assert response.status_code == 400
    assert "debug" in response.json()["detail"]


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


def test_local_launch_mounts_gradio_on_fastapi_parent(monkeypatch):
    mounted_apps = []

    import gradio as gr
    import uvicorn

    def _capture_mount(app, *args, **kwargs):
        mounted_apps.append(app)
        return app

    monkeypatch.setattr(gr, "mount_gradio_app", _capture_mount)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")

    assert mounted_apps[-1] is app_module.API_APP
    assert app_module.ASGI_APP is app_module.API_APP


def test_mounted_asgi_app_serves_fastapi_and_gradio_routes(monkeypatch):
    import gradio as gr

    original_mount = gr.mount_gradio_app
    monkeypatch.setattr(gr, "mount_gradio_app", original_mount)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    client = TestClient(app_module.ASGI_APP)

    health_response = client.get("/health")
    api_response = client.post("/api/ask", json={})
    gradio_response = client.get("/gradio_api/info")

    assert health_response.status_code == 200
    assert api_response.status_code == 400
    assert gradio_response.status_code == 200
    assert "/submit" in gradio_response.json()["named_endpoints"]


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

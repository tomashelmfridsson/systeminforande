from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.reasoning import extract_hf_usage
from rag.agentic_rewrite import (
    DEFAULT_REWRITE_MODEL,
    build_retrieval_rewrite_prompt,
    generate_retrieval_rewrite,
    parse_retrieval_rewrite_response,
)
from rag.agentic_answer import (
    build_evidence_answer_prompt,
    generate_evidence_answer,
)
from rag.grounding import INSUFFICIENT_EVIDENCE_ANSWER
from rag.synthesis import (
    DEFAULT_SYNTHESIS_MODEL,
    SUPPORTED_EXPERIMENT_MODELS,
    build_synthesis_prompt,
    build_final_grounded_answer,
    resolve_synthesis_settings,
)


CHUNKS = [
    {
        "source": "Mallar_utbildningsstrategi.pdf",
        "source_type": "pdf",
        "title": "Utbildningsstrategi",
        "text": (
            "Utbildningsstrategin ska beskriva syftet med dokumentet, utbildningsstrategins huvudresultat, "
            "målgrupper, utbildningarnas innehåll, utbildningsmål och ett grovt uppskattat utbildningsbehov. "
            "Den ska också beskriva bakgrund, förutsättningar och hur utbildningen ska genomföras."
        ),
        "pages": [1],
    }
]

OBSTACLE_CHUNKS = [
    {
        "source": "Arbetsomraden_checklista.pdf",
        "source_type": "pdf",
        "title": "Arbetsområden vid systeminförande",
        "text": (
            "Checklistan delar upp systeminförande i arbetsområden som acceptanstest, utbildning "
            "och information, IT-miljöer, konvertering och laddning samt driftsättning. "
            "Arbetsområdena används för att hålla ihop aktiviteter, ansvar och uppföljning."
        ),
        "pages": [1],
    },
    {
        "source": "Systemfaser.pdf",
        "source_type": "pdf",
        "title": "Systemfaser",
        "text": (
            "Införandet behöver planeras genom förberedelser, genomförande och uppföljning. "
            "Varje fas kräver att beroenden mellan verksamhet, teknik och drift hanteras så att "
            "systemet kan användas i den skarpa verksamheten."
        ),
        "pages": [2],
    },
]

UNDERVISNING_RELATION_CHUNKS = [
    {
        "id": "kursplan:1",
        "source": "Utbildningsplan.pdf",
        "source_type": "pdf",
        "title": "Planering av lärarledd utbildning",
        "text": (
            "Lärarna undervisade pilotgruppen i de nya arbetssätten innan införandet. "
            "Erfarenheterna användes för att justera kursplanen och göra innehållet mer praktiskt."
        ),
        "pages": [4],
    },
    {
        "id": "uppfoljning:2",
        "source": "Utbildningsuppfoljning.pdf",
        "source_type": "pdf",
        "title": "Uppföljning efter genomförd utbildning",
        "text": (
            "Efter att superanvändarna hade undervisat sina grupper samlades frågor in. "
            "Frågorna användes som underlag för korta repetitionstillfällen."
        ),
        "pages": [2],
    },
]

UNDERVISNING_REWRITE_METADATA = {
    "status": "ok",
    "original_question": "Hur ska undervisning följas upp efter införandet?",
    "semantic_terms": [
        {"surface": "undervisning", "normalized_family": "undervisa", "kind": "derivation"},
        {"surface": "undervisade", "normalized_family": "undervisa", "kind": "inflection"},
        {"surface": "undervisat", "normalized_family": "undervisa", "kind": "inflection"},
    ],
    "negative_constraints": ["lägg inte till generiska pedagogiska råd"],
}


def test_agent2_prompt_teaches_semantic_relations_between_undervisning_and_undervisade():
    prompt = build_evidence_answer_prompt(
        "Hur ska undervisning följas upp efter införandet?",
        UNDERVISNING_RELATION_CHUNKS,
        UNDERVISNING_REWRITE_METADATA,
    )
    prompt_lower = prompt.lower()

    assert "undervisning" in prompt_lower
    assert "undervisade" in prompt_lower
    assert "undervisat" in prompt_lower
    assert "ordformer" in prompt_lower
    assert "samma begreppsfamilj" in prompt_lower
    assert "inte som egna fakta" in prompt_lower


def test_agent2_rejects_semantic_relation_answer_when_an_answer_point_lacks_evidence_id():
    question = "Hur ska undervisning följas upp efter införandet?"

    def fake_llm(prompt: str, model: str | None = None) -> str:
        assert "chunk_id=kursplan:1" in prompt
        assert "chunk_id=uppfoljning:2" in prompt
        return json.dumps(
            {
                "original_question": question,
                "answer": (
                    "Undervisningen bör följas upp genom att erfarenheter från när lärarna undervisade "
                    "pilotgruppen används för att justera kursplanen och göra innehållet mer praktiskt. "
                    "Efter att superanvändarna hade undervisat sina grupper behöver frågor samlas in och "
                    "användas som underlag för korta repetitionstillfällen."
                ),
                "answer_scope": "direct",
                "evidence_used": [
                    {
                        "chunk_id": "kursplan:1",
                        "source": "Utbildningsplan.pdf",
                        "pages": [4],
                        "claim_supported": "Erfarenheter från undervisade pilotgrupper används för att justera kursplanen.",
                    }
                ],
                "unsupported_or_uncertain": [],
                "source_coverage": {
                    "uses_retrieved_chunks": True,
                    "answers_original_question": True,
                    "ignores_metadata_as_facts": True,
                },
                "grounding_notes": "Svaret använder ordformen undervisning i frågan och undervisade/undervisat i evidensen.",
            },
            ensure_ascii=False,
        )

    result = generate_evidence_answer(
        question,
        UNDERVISNING_RELATION_CHUNKS,
        UNDERVISNING_REWRITE_METADATA,
        fake_llm,
    )

    assert result["status"] == "fallback"
    assert result["debug"]["fallback_reason"] == "agent2_missing_evidence"
    assert result["evidence_ids_used"] == []


def test_synthesis_prompt_asks_for_fuller_source_grounded_obstacle_reasoning():
    prompt = build_synthesis_prompt(
        "Vilka hinder finns i systeminförande?",
        OBSTACLE_CHUNKS,
        "Frågan verkar beröra flera återkommande områden eller delar.",
    )
    prompt_lower = prompt.lower()

    assert "6 till 10 meningar" in prompt
    assert "hinder" in prompt_lower
    assert "förklara vad punkterna innebär" in prompt_lower
    assert "varför de spelar roll" in prompt_lower
    assert "hur de hänger ihop" in prompt_lower
    assert "inga dokument- eller sidreferenser" in prompt_lower
    assert "frågan verkar beröra" in prompt_lower
    assert "undvik" in prompt_lower


def test_synthesis_stage_rejects_fragan_verkar_template_phrase():
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        return (
            "Frågan verkar beröra flera återkommande områden eller delar. De hinder som framträder "
            "tydligast är acceptanstest, utbildning och information, IT-miljöer, konvertering och "
            "laddning samt driftsättning. Arbetsområdena används för aktiviteter, ansvar och uppföljning. "
            "Införandet behöver planeras genom förberedelser, genomförande och uppföljning."
        )

    result = build_final_grounded_answer(
        "Vilka hinder finns i systeminförande?",
        OBSTACLE_CHUNKS,
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

    assert result["synthesis_used"] is False
    assert result["final_answer"] == result["extractive_answer"]
    assert result["llm_status"] == "fallback_to_extractive_due_to_grounding_check"


def test_synthesis_stage_accepts_fuller_source_bound_obstacle_reasoning():
    rewritten = (
        "Ett systeminförande kan hindras av att flera arbetsområden måste fungera samtidigt, inte av en enda isolerad aktivitet. "
        "Underlaget pekar på acceptanstest, utbildning och information, IT-miljöer, konvertering och laddning samt driftsättning. "
        "Det betyder att hinder kan uppstå när ansvar, aktiviteter eller uppföljning inom dessa områden inte hålls ihop. "
        "Acceptanstest och utbildning påverkar verksamhetens möjlighet att börja använda systemet på ett kontrollerat sätt. "
        "IT-miljöer, konvertering och driftsättning är samtidigt tekniska och praktiska förutsättningar för att lösningen ska fungera i skarp drift. "
        "Faserna i införandet gör också att beroenden mellan verksamhet, teknik och drift behöver planeras, genomföras och följas upp i rätt ordning. "
        "Hindren ska därför förstås som samordnings- och genomförandeproblem inom de arbetsområden som källorna räknar upp."
    )

    result = build_final_grounded_answer(
        "Vilka hinder finns i systeminförande?",
        OBSTACLE_CHUNKS,
        enable_synthesis=True,
        llm_rewrite=lambda prompt, model=None: rewritten,
    )

    assert result["synthesis_used"] is True
    assert result["final_answer"] == rewritten


def test_synthesis_stage_strips_llm_generated_sources_before_app_sources_are_appended():
    rewritten = (
        "Ett systeminförande kan hindras av att flera arbetsområden måste fungera samtidigt, inte av en enda isolerad aktivitet. "
        "Underlaget pekar på acceptanstest, utbildning och information, IT-miljöer, konvertering och laddning samt driftsättning. "
        "Det betyder att hinder kan uppstå när ansvar, aktiviteter eller uppföljning inom dessa områden inte hålls ihop. "
        "Acceptanstest och utbildning påverkar verksamhetens möjlighet att börja använda systemet på ett kontrollerat sätt. "
        "IT-miljöer, konvertering och driftsättning är samtidigt tekniska och praktiska förutsättningar för att lösningen ska fungera i skarp drift."
        "\n\n---\n\n### Källor\n- [Felaktig extra källista](https://example.com)"
    )

    result = build_final_grounded_answer(
        "Vilka hinder finns i systeminförande?",
        OBSTACLE_CHUNKS,
        enable_synthesis=True,
        llm_rewrite=lambda prompt, model=None: rewritten,
    )

    final_answer = str(result["final_answer"])

    assert result["synthesis_used"] is True
    assert final_answer.count("### Källor") == 0
    assert "Felaktig extra källista" not in final_answer


def test_synthesis_stage_is_off_by_default():
    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=False,
    )

    assert result["synthesis_enabled"] is False
    assert result["synthesis_used"] is False
    assert result["final_answer"] == result["extractive_answer"]
    assert "utbildningsstrategi" in result["final_answer"].lower()


@pytest.mark.parametrize("model_name", ["openai/gpt-oss-120b", "zai-org/GLM-5.2"])
def test_synthesis_stage_can_replace_extractive_answer_when_feature_flag_is_enabled(model_name: str):
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        assert "Utbildningsstrategi" in prompt
        assert model == model_name
        assert "4 till 8 meningar" in prompt
        assert "Svara självständigt" in prompt
        assert "Använd inte ett förbyggt bassvar som struktur" in prompt
        assert "Fallback-svar om LLM-svaret inte blir källbundet" in prompt
        assert "Bassvar:" not in prompt
        return (
            "En utbildningsstrategi behöver göra det tydligt varför dokumentet finns och vilket huvudresultat "
            "strategin ska leda till. Den ska också beskriva vilka målgrupper som omfattas, vilket innehåll "
            "utbildningarna har och vilka mål utbildningen ska uppfylla. Utifrån underlaget behöver strategin dessutom "
            "ta upp ett grovt uppskattat utbildningsbehov, bakgrund, förutsättningar och hur utbildningen ska genomföras."
        )

    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=True,
        llm_model=model_name,
        llm_rewrite=llm_rewrite,
    )

    assert result["synthesis_enabled"] is True
    assert result["synthesis_used"] is True
    assert result["llm_model"] == model_name
    assert result["final_answer"] != result["extractive_answer"]
    assert "målgrupper" in result["final_answer"].lower()
    assert "utbildningsbehov" in result["final_answer"].lower()


def test_synthesis_stage_falls_back_to_extractive_answer_when_rewrite_is_unsupported():
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        return "Projektet bör ha en styrgrupp och veckovisa statusmöten för att lyckas."

    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

    assert result["synthesis_enabled"] is True
    assert result["synthesis_used"] is False
    assert result["final_answer"] == result["extractive_answer"]
    assert result["llm_status"] == "fallback_to_extractive_due_to_grounding_check"


def test_synthesis_stage_rejects_formulaic_template_phrases():
    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        return (
            "Materialet visar att utbildningsstrategin bör beskriva syfte, huvudresultat, målgrupper, "
            "utbildningarnas innehåll, utbildningsmål och utbildningsbehov. Den bör också beskriva "
            "bakgrund, förutsättningar och genomförande."
        )

    result = build_final_grounded_answer(
        "Vad ska en utbildningsstrategi innehålla",
        CHUNKS,
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

    assert result["synthesis_used"] is False
    assert result["final_answer"] == result["extractive_answer"]
    assert result["llm_status"] == "fallback_to_extractive_due_to_grounding_check"


def test_synthesis_stage_does_not_call_llm_when_extractive_path_has_insufficient_evidence():
    called = False

    def llm_rewrite(prompt: str, model: str | None = None) -> str:
        nonlocal called
        called = True
        return "Det här borde aldrig användas."

    result = build_final_grounded_answer(
        "Vilken färg har månen i projektmodellen?",
        [],
        enable_synthesis=True,
        llm_rewrite=llm_rewrite,
    )

    assert result["extractive_answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["synthesis_used"] is False
    assert called is False


def test_resolve_synthesis_settings_uses_llm_synthesis_by_default(monkeypatch):
    monkeypatch.delenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", raising=False)
    monkeypatch.delenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", raising=False)

    settings = resolve_synthesis_settings()

    assert settings["enabled"] is True
    assert settings["model"] == DEFAULT_SYNTHESIS_MODEL
    assert SUPPORTED_EXPERIMENT_MODELS == ("openai/gpt-oss-120b", "zai-org/GLM-5.2")


def test_resolve_synthesis_settings_supports_environment_default_model(monkeypatch):
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", "true")
    monkeypatch.setenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", "zai-org/GLM-5.2")

    settings = resolve_synthesis_settings()

    assert settings["enabled"] is True
    assert settings["model"] == "zai-org/GLM-5.2"
    assert settings["enabled_source"] == "environment"


def test_resolve_synthesis_settings_supports_per_request_override(monkeypatch):
    monkeypatch.setenv("SYSTEMINFORANDE_ENABLE_LLM_SYNTHESIS", "false")
    monkeypatch.setenv("SYSTEMINFORANDE_LLM_SYNTHESIS_MODEL", "openai/gpt-oss-120b")

    settings = resolve_synthesis_settings(
        enable_synthesis=True,
        llm_model="zai-org/GLM-5.2",
        default_model="openai/gpt-oss-120b",
    )

    assert settings["enabled"] is True
    assert settings["model"] == "zai-org/GLM-5.2"
    assert settings["requested_model"] == "zai-org/GLM-5.2"
    assert settings["enabled_source"] == "override"


def test_extract_hf_usage_reads_huggingface_usage_fields_defensively():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens="12",
            completion_tokens=5,
            total_tokens=17,
        )
    )

    assert extract_hf_usage(response) == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


def test_extract_hf_usage_tolerates_missing_usage():
    assert extract_hf_usage(SimpleNamespace()) == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_huggingface_provider_call_returns_usage_when_present(monkeypatch):
    from llm import reasoning

    class FakeClient:
        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message={"content": "Källbundet svar"})],
                usage=SimpleNamespace(
                    prompt_tokens=13,
                    completion_tokens=8,
                    total_tokens=21,
                ),
            )

    monkeypatch.setattr(reasoning, "get_llm_client", lambda model=None: FakeClient())

    result = reasoning.generate_reasoning_from_prompt_with_usage(
        "Svara bara utifrån källmaterialet.",
        model="Qwen/Qwen3-32B",
    )

    assert result.text == "Källbundet svar"
    assert result.usage == {
        "prompt_tokens": 13,
        "completion_tokens": 8,
        "total_tokens": 21,
    }


def test_huggingface_provider_call_does_not_fail_when_usage_is_missing(monkeypatch):
    from llm import reasoning

    class FakeClient:
        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message={"content": "Svar utan usage-metadata"})]
            )

    monkeypatch.setattr(reasoning, "get_llm_client", lambda model=None: FakeClient())

    result = reasoning.generate_reasoning_from_prompt_with_usage(
        "Svara bara utifrån källmaterialet.",
        model="Qwen/Qwen3-32B",
    )

    assert result.text == "Svar utan usage-metadata"
    assert result.usage == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_agent1_q22_style_question_keeps_original_highest_weight_and_allows_useful_pdf_wording_variants():
    question = "Hur lämnas systemet över till förvaltning efter införandet?"

    def fake_llm(prompt: str, model: str | None = None) -> str:
        assert model == DEFAULT_REWRITE_MODEL
        assert question in prompt
        assert "Svara inte på frågan" in prompt
        assert "strikt JSON" in prompt
        assert len(prompt.split()) <= 600
        return json.dumps(
            {
                "original_question": question,
                "retrieval_queries": [
                    {"query": "förvaltningsöverlämnande efter införande", "purpose": "compound"},
                    {"query": "överlämning till förvaltning", "purpose": "swedish_inflection"},
                    {"query": "mottagare förvaltningsobjekt", "purpose": "synonym"},
                ],
                "semantic_terms": [
                    {"surface": "lämnas över", "normalized_family": "överlämna", "kind": "lemma"},
                    {"surface": "överlämning", "normalized_family": "överlämna", "kind": "inflection"},
                    {"surface": "förvaltningsöverlämnande", "normalized_family": "överlämna förvaltning", "kind": "compound"},
                    {"surface": "förvaltning", "normalized_family": "förvaltning", "kind": "lemma"},
                ],
                "negative_constraints": ["ändra inte frågan till driftsättning"],
                "confidence": 0.82,
            },
            ensure_ascii=False,
        )

    result = generate_retrieval_rewrite(question, fake_llm)

    assert result["status"] == "ok"
    assert result["model"] == DEFAULT_REWRITE_MODEL
    assert result["retrieval_queries"][0] == {
        "query": question,
        "purpose": "literal",
        "weight": 1.0,
    }
    assert len(result["retrieval_queries"]) <= 5
    assert all(item["weight"] < 1.0 for item in result["retrieval_queries"][1:])
    variant_text = " ".join(item["query"] for item in result["retrieval_queries"][1:])
    assert "förvaltningsöverlämnande" in variant_text
    assert "överlämning" in variant_text
    assert result["debug"]["dropped_queries"] == []


def test_agent1_generic_swedish_grammar_variants_are_accepted_without_domain_terms():
    question = "Hur ska undervisning planeras när lärare har undervisat olika grupper?"
    payload = {
        "original_question": question,
        "retrieval_queries": [
            {"query": "planera undervisning för olika lärargrupper", "purpose": "swedish_inflection"},
            {"query": "undervisade grupper och utbildningsplanering", "purpose": "swedish_inflection"},
            {"query": "undervisat olika grupper planering", "purpose": "swedish_inflection"},
        ],
        "semantic_terms": [
            {"surface": "undervisning", "normalized_family": "undervisa", "kind": "lemma"},
            {"surface": "undervisade", "normalized_family": "undervisa", "kind": "inflection"},
            {"surface": "undervisat", "normalized_family": "undervisa", "kind": "inflection"},
            {"surface": "lärare", "normalized_family": "lärare", "kind": "lemma"},
        ],
        "negative_constraints": [],
        "confidence": 0.76,
    }

    result = parse_retrieval_rewrite_response(question, json.dumps(payload, ensure_ascii=False))

    assert result["status"] == "ok"
    queries = [item["query"] for item in result["retrieval_queries"]]
    assert queries[0] == question
    assert any("undervisade" in query for query in queries)
    assert any("undervisat" in query for query in queries)
    assert result["debug"]["accepted_query_count"] >= 3


def test_agent1_drifted_rewrites_are_dropped_before_retrieval():
    question = "Hur lämnas systemet över till förvaltning?"
    payload = {
        "original_question": question,
        "retrieval_queries": [
            {"query": "driftsättning och driftstart av systemet", "purpose": "broader_context"},
            {"query": "överlämning till förvaltning", "purpose": "swedish_inflection"},
        ],
        "semantic_terms": [
            {"surface": "överlämning", "normalized_family": "överlämna", "kind": "inflection"},
            {"surface": "förvaltning", "normalized_family": "förvaltning", "kind": "lemma"},
            {"surface": "driftsättning", "normalized_family": "driftsättning", "kind": "lemma"},
        ],
        "negative_constraints": ["driftsättning är inte samma scope som överlämning"],
        "confidence": 0.9,
    }

    result = parse_retrieval_rewrite_response(question, json.dumps(payload, ensure_ascii=False))

    assert result["status"] == "ok"
    queries = [item["query"] for item in result["retrieval_queries"]]
    assert "överlämning till förvaltning" in queries
    assert "driftsättning och driftstart av systemet" not in queries
    assert result["debug"]["dropped_queries"] == [
        {
            "query": "driftsättning och driftstart av systemet",
            "reason": "semantic_drift",
        }
    ]


def test_agent1_response_must_be_strict_json_and_preserve_original_question():
    question = "Vad är ett arbetsområde?"

    invalid_json = parse_retrieval_rewrite_response(question, "Här är JSON:\n{}")
    changed_question = parse_retrieval_rewrite_response(
        question,
        json.dumps(
            {
                "original_question": "Vad är en arbetsmodell?",
                "retrieval_queries": [{"query": "arbetsområde definition", "purpose": "literal"}],
                "semantic_terms": [],
                "negative_constraints": [],
                "confidence": 0.8,
            },
            ensure_ascii=False,
        ),
    )

    assert invalid_json["status"] == "fallback"
    assert invalid_json["debug"]["fallback_reason"] == "agent1_invalid_json"
    assert changed_question["status"] == "fallback"
    assert changed_question["debug"]["fallback_reason"] == "original_question_mismatch"


def test_agent1_low_confidence_or_answer_like_rewrites_fall_back_or_are_dropped():
    question = "Finns det en arbetsmodell för införande?"
    low_confidence = parse_retrieval_rewrite_response(
        question,
        json.dumps(
            {
                "original_question": question,
                "retrieval_queries": [{"query": "införandemodell", "purpose": "synonym"}],
                "semantic_terms": [],
                "negative_constraints": [],
                "confidence": 0.2,
            },
            ensure_ascii=False,
        ),
    )
    answer_like = parse_retrieval_rewrite_response(
        question,
        json.dumps(
            {
                "original_question": question,
                "retrieval_queries": [
                    {"query": "Ja, det finns en arbetsmodell som används i införandet.", "purpose": "broader_context"},
                    {"query": "införandemodell arbetsmodell", "purpose": "synonym"},
                ],
                "semantic_terms": [
                    {"surface": "arbetsmodell", "normalized_family": "arbetsmodell", "kind": "lemma"},
                    {"surface": "införandemodell", "normalized_family": "arbetsmodell", "kind": "synonym"},
                ],
                "negative_constraints": [],
                "confidence": 0.8,
            },
            ensure_ascii=False,
        ),
    )

    assert low_confidence["status"] == "fallback"
    assert low_confidence["debug"]["fallback_reason"] == "low_confidence"
    assert "Ja, det finns" not in " ".join(
        item["query"] for item in answer_like["retrieval_queries"]
    )
    assert answer_like["debug"]["dropped_queries"][0]["reason"] == "answers_question"


def test_agent1_rewrite_prompt_requests_strict_json_and_small_output_budget():
    prompt = build_retrieval_rewrite_prompt("Hur används acceptanstest i införandet?")

    assert "openai/gpt-oss-20b" in prompt
    assert "max 5" in prompt
    assert "strikt JSON" in prompt
    assert "Svara inte på frågan" in prompt
    assert "200 output tokens" in prompt
    assert len(prompt.split()) <= 600

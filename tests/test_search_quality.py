from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.search as search_module
from rag.search import explain_search


def _install_synthetic_chunks(monkeypatch, chunks: list[dict]) -> None:
    monkeypatch.setattr(search_module, "_CHUNKS_CACHE", None)
    monkeypatch.setattr(search_module, "_INDEX_CACHE", None)
    monkeypatch.setattr(search_module, "load_chunks", lambda: chunks)


def _rewrite_result(question: str, accepted: list[str], rejected: list[dict] | None = None) -> dict:
    return {
        "status": "ok",
        "original_question": question,
        "retrieval_queries": [
            {"query": query, "purpose": "literal" if index == 0 else "synonym", "weight": 1.0 if index == 0 else 0.8}
            for index, query in enumerate(accepted)
        ],
        "debug": {"dropped_queries": rejected or []},
    }


def _sources(search_debug: dict) -> list[str]:
    return [item["chunk"]["source"] for item in search_debug["top_results"]]


def test_multi_query_hits_that_match_multiple_variants_outrank_single_variant_hits(monkeypatch):
    question = "Hur beskrivs ansvar?"
    _install_synthetic_chunks(
        monkeypatch,
        [
            {
                "id": "single-original",
                "source": "Ansvar_planering.pdf",
                "source_type": "pdf",
                "title": "Ansvar",
                "text": "Ansvar beskrivs med roller och tidplan.",
                "pages": [1],
            },
            {
                "id": "multi-variant",
                "source": "Utbildningsstrategi.pdf",
                "source_type": "pdf",
                "title": "Utbildningsstrategi",
                "text": "Ansvar beskrivs i en utbildningsstrategi med målgrupper och utbildningsbehov.",
                "pages": [2],
            },
            {
                "id": "single-variant",
                "source": "Malgrupper.pdf",
                "source_type": "pdf",
                "title": "Målgrupper",
                "text": "Målgrupper beskrivs separat i underlaget.",
                "pages": [3],
            },
        ],
    )

    result = explain_search(
        question,
        top_k=3,
        retrieval_rewrite=_rewrite_result(
            question,
            [question, "utbildningsstrategi utbildningsbehov", "målgrupper utbildningsstrategi"],
        ),
    )

    assert result["top_results"][0]["chunk"]["id"] == "multi-variant"
    assert result["agentic_retrieval"]["merged_ranking"][0]["matched_query_count"] >= 2


def test_multi_query_rerank_keeps_original_query_hits_above_weak_drifted_variant(monkeypatch):
    question = "Hur planeras utbildning?"
    _install_synthetic_chunks(
        monkeypatch,
        [
            {
                "id": "original-topic",
                "source": "Utbildning_planering.pdf",
                "source_type": "pdf",
                "title": "Planera utbildning",
                "text": "Planeras utbildning med ansvar, målgrupper och tidplan.",
                "pages": [1],
            },
            {
                "id": "weak-drift",
                "source": "Acceptanstest_testplan.pdf",
                "source_type": "pdf",
                "title": "Acceptanstest testplan",
                "text": "Acceptanstest och testplan beskriver verifiering före godkännande.",
                "pages": [5],
            },
        ],
    )

    result = explain_search(
        question,
        top_k=2,
        retrieval_rewrite=_rewrite_result(
            question,
            [question, "acceptanstest testplan"],
            rejected=[{"query": "driftsättning", "reason": "semantic_drift"}],
        ),
    )

    assert result["top_results"][0]["chunk"]["id"] == "original-topic"
    assert result["agentic_retrieval"]["rejected_variants"] == [
        {"query": "driftsättning", "reason": "semantic_drift"}
    ]


def test_multi_query_retrieval_deduplicates_chunks_before_answer_generation(monkeypatch):
    question = "Hur planeras utbildning?"
    _install_synthetic_chunks(
        monkeypatch,
        [
            {
                "id": "same-chunk",
                "source": "Utbildning_planering.pdf",
                "source_type": "pdf",
                "title": "Planera utbildning",
                "text": "Planeras utbildning genom utbildningsstrategi och utbildningsbehov.",
                "pages": [1],
            }
        ],
    )

    result = explain_search(
        question,
        top_k=5,
        retrieval_rewrite=_rewrite_result(
            question,
            [question, "utbildningsstrategi", "utbildningsbehov"],
        ),
    )

    ids = [item["chunk"]["id"] for item in result["top_results"]]
    assert ids == ["same-chunk"]
    assert result["agentic_retrieval"]["per_query_hits"]
    assert result["agentic_retrieval"]["merged_ranking"][0]["matched_query_count"] >= 2


def test_acceptance_test_query_prefers_acceptance_documents():
    result = explain_search("Hur används acceptanstest i införandet?", top_k=5)

    assert "acceptanstest" in result["expanded_query_terms"]
    assert any(
        source in {
            "Verktyget_aktiviteter.pdf",
            "Mallar_acceptanstest_testplan.pdf",
            "Mallar_acceptanstest_testrapport.pdf",
            "Arbetsomraden_checklista.pdf",
            "Inforandekrav_checklista.pdf",
        }
        for source in _sources(result)
    )


def test_work_area_definition_prefers_work_area_checklist():
    result = explain_search("Vad är ett arbetsområde?", top_k=5)

    assert result["top_results"]
    assert result["top_results"][0]["chunk"]["source"] in {
        "Arbetsomraden_checklista.pdf",
        "Verktyget_och_systeminforandet.pdf",
    }


def test_common_acceptance_test_typos_are_mapped_to_canonical_term():
    result = explain_search("Hur anvnds acceptanstst i införandet?", top_k=5)

    assert "acceptanstest" in result["expanded_query_terms"]
    assert any(
        source in {
            "Verktyget_aktiviteter.pdf",
            "Mallar_acceptanstest_testplan.pdf",
            "Mallar_acceptanstest_testrapport.pdf",
        }
        for source in _sources(result)
    )


def test_implementation_followup_prefers_project_pdfs_over_web_content():
    result = explain_search("På vilket sätt bör implmenteringen planeras och följas up?", top_k=5)

    sources = _sources(result)
    assert any(
        source in {
            "Verktyget_projektstyrning.pdf",
            "Mallar_acceptanstest_testplan.pdf",
            "Verktyget_aktiviteter.pdf",
            "Mallar_omlaggningsstrategi.pdf",
        }
        for source in sources
    )
    assert sources[0].endswith(".pdf")


def test_checklist_for_requirement_introduction_prefers_requirement_checklist():
    result = explain_search("Finns det en checklista för införandekrav", top_k=5)

    assert result["top_results"]
    assert result["top_results"][0]["chunk"]["source"] == "Inforandekrav_checklista.pdf"


def test_project_library_definition_prefers_project_steering_document():
    result = explain_search("Vad är ett projektbibliotek", top_k=5)

    assert result["top_results"]
    assert result["top_results"][0]["chunk"]["source"] == "Verktyget_projektstyrning.pdf"


def test_project_organization_definition_prefers_project_organization_sections():
    result = explain_search("Vad är en projektorganisation", top_k=5)

    assert result["top_results"]
    assert result["top_results"][0]["chunk"]["source"] in {
        "Verktyget_projektstyrning.pdf",
        "Mallar_delprojektplan.pdf",
    }

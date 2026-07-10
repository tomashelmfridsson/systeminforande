from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.search import explain_search


def _sources(search_debug: dict) -> list[str]:
    return [item["chunk"]["source"] for item in search_debug["top_results"]]


def test_acceptance_test_query_prefers_acceptance_documents():
    result = explain_search("Hur används acceptanstest i införandet?", top_k=5)

    assert "acceptanstest" in result["expanded_query_terms"]
    assert any(
        source in {
            "210_Acceptanstest_testplan.pdf",
            "222_Acceptanstest_testplan.pdf",
            "231_Acceptanstest_krav_leveransgodkannande.pdf",
        }
        for source in _sources(result)
    )


def test_work_area_definition_prefers_work_area_checklist():
    result = explain_search("Vad är ett arbetsområde?", top_k=5)

    assert result["top_results"]
    assert result["top_results"][0]["chunk"]["source"] == "Checklista_Arbetsomraden.pdf"


def test_common_acceptance_test_typos_are_mapped_to_canonical_term():
    result = explain_search("Hur anvnds acceptanstst i införandet?", top_k=5)

    assert "acceptanstest" in result["expanded_query_terms"]
    assert any(source.startswith("210_Acceptanstest") or source.startswith("222_Acceptanstest") for source in _sources(result))


def test_implementation_followup_prefers_project_pdfs_over_web_content():
    result = explain_search("På vilket sätt bör implmenteringen planeras och följas up?", top_k=5)

    sources = _sources(result)
    assert any(
        source in {
            "210_Acceptanstest_testplan.pdf",
            "350_Konvertering_strategi.pdf",
            "354_Konvertering_verifiering_kontroller.pdf",
        }
        for source in sources
    )
    assert sources[0].endswith(".pdf")

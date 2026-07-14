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

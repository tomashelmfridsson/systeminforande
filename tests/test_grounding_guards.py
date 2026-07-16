from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.search as search_module
from rag.grounding import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    filter_allowed_results,
    grounded_answer_or_fallback,
    is_allowed_chunk_source,
)


def test_grounded_answer_or_fallback_returns_explicit_source_limited_message():
    assert grounded_answer_or_fallback("") == INSUFFICIENT_EVIDENCE_ANSWER
    assert "PDF" in INSUFFICIENT_EVIDENCE_ANSWER
    assert "hemsidans innehåll" in INSUFFICIENT_EVIDENCE_ANSWER


def test_allowed_chunk_source_accepts_pdfs_and_systeminforande_homepage_urls_only():
    assert is_allowed_chunk_source({"source_type": "pdf", "source": "Dokument.pdf"})
    assert is_allowed_chunk_source(
        {"source_type": "web", "source": "https://www.systeminforande.se/arbetsmodell"}
    )

    assert not is_allowed_chunk_source(
        {"source_type": "web", "source": "https://example.com/arbetsmodell"}
    )
    assert not is_allowed_chunk_source({"source_type": "txt", "source": "notes.txt"})


def test_filter_allowed_results_removes_non_homepage_web_sources():
    filtered = filter_allowed_results(
        [
            (9.0, {"source_type": "pdf", "source": "Dokument.pdf"}),
            (8.0, {"source_type": "web", "source": "https://www.systeminforande.se/verktyg"}),
            (7.0, {"source_type": "web", "source": "https://other.example.com/page"}),
        ]
    )

    assert len(filtered) == 2
    assert all(item[1]["source"] != "https://other.example.com/page" for item in filtered)


def test_search_index_excludes_disallowed_web_chunks(monkeypatch):
    custom_chunks = [
        {
            "id": "allowed_pdf_1",
            "title": "Projektbibliotek",
            "section": "1",
            "text": "Projektbibliotek används för att lagra projektets dokument och filer.",
            "pages": [1],
            "source": "Verktyget_projektstyrning.pdf",
            "source_type": "pdf",
        },
        {
            "id": "disallowed_web_1",
            "title": "Projektbibliotek",
            "section": "web-1",
            "text": "Projektbibliotek betyder något helt annat på en extern webbplats.",
            "pages": [1],
            "source": "https://example.com/projektbibliotek",
            "source_type": "web",
        },
    ]

    monkeypatch.setattr(search_module, "load_chunks", lambda: custom_chunks)
    search_module._INDEX_CACHE = None
    search_module._CHUNKS_CACHE = None

    results = search_module.search("Vad är ett projektbibliotek", top_k=5)

    assert results
    assert all(chunk["source"] != "https://example.com/projektbibliotek" for _, chunk in results)

    search_module._INDEX_CACHE = None

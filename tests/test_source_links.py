from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.search import search
from rag.source_links import build_sources_md, collect_homepage_links



def test_build_sources_md_adds_related_homepage_links_for_pdf_results():
    results = [
        (
            9.2,
            {
                "source": "Arbetsomraden_checklista.pdf",
                "source_type": "pdf",
                "title": "Arbetsområden",
                "pages": [2, 3],
            },
        ),
        (
            8.1,
            {
                "source": "Mallar_acceptanstest_testplan.pdf",
                "source_type": "pdf",
                "title": "Acceptanstest testplan",
                "pages": [1],
            },
        ),
    ]

    markdown = build_sources_md(results)

    assert "### Källor" in markdown
    assert "### Relaterade hemsidor" in markdown
    assert "https://www.systeminforande.se/implementering2" in markdown
    assert "https://www.systeminforande.se/verktyg" in markdown
    assert "https://www.systeminforande.se/checklistor-och-mallar-till-verktyget-1" in markdown



def test_real_acceptance_test_results_collect_related_homepage_links():
    results = search("Hur används acceptanstest i införandet?", top_k=5)

    homepage_urls = {link["url"] for link in collect_homepage_links(results)}

    assert "https://www.systeminforande.se/verktyg" in homepage_urls or "https://www.systeminforande.se/checklistor-och-mallar-till-verktyget-1" in homepage_urls
